"""Tests for the LTX Alpha Gen helper nodes. Planning maths needs nothing; the tensor tests run only if torch is installed."""

import pytest

from src.creaturenodes import alphagen as ag
from src.creaturenodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


# ---------------- planning maths ----------------

@pytest.mark.parametrize("length, expected", [(1, 1), (2, 9), (9, 9), (10, 17), (120, 121), (121, 121), (122, 129)])
def test_frames_8n1(length, expected):
    assert ag.frames_8n1(length) == expected
    assert (ag.frames_8n1(length) - 1) % 8 == 0


@pytest.mark.parametrize("w, h, size, expected", [
    (1920, 1080, ag.SIZE_FULL, (1920, 1080, 1920, 1088)),   # docs example: pad 1080 -> 1088, no scaling
    (1280, 720, ag.SIZE_FULL, (1280, 720, 1280, 736)),      # small clips are not scaled up
    (3840, 2160, ag.SIZE_FULL, (1920, 1080, 1920, 1088)),   # 4K scaled to fit
    (2048, 1152, ag.SIZE_FULL, (1920, 1080, 1920, 1088)),   # 2K 16:9
    (1080, 1920, ag.SIZE_FULL, (1080, 1920, 1088, 1920)),   # portrait
    (1920, 1080, ag.SIZE_HALF, (960, 540, 960, 544)),
])
def test_fit_size(w, h, size, expected):
    assert ag.fit_size(w, h, size) == expected


def test_fit_size_never_exceeds_limits():
    for w in range(200, 5000, 97):
        for h in (360, 720, 1080, 1440, 2160, 2900):
            ww, wh, pw, ph = ag.fit_size(w, h)
            assert pw % 32 == 0 and ph % 32 == 0
            assert max(pw, ph) <= 1920 and min(pw, ph) <= 1088
            assert ww <= pw and wh <= ph


def test_short_clip_is_one_window():
    assert ag.plan_windows(81) == ([(0, 81)], 81)
    assert ag.plan_windows(121) == ([(0, 121)], 121)


def test_windows_cover_clip_with_minimum_overlap():
    for frames in range(122, 461):  # 4 windows cover at most 121 + 3 * 113 = 460 frames
        windows, used = ag.plan_windows(frames)
        assert used == frames
        assert windows[0][0] == 0 and windows[-1][0] + windows[-1][1] == frames
        assert all(n == 121 for _, n in windows)
        for (s1, n1), (s2, _) in zip(windows, windows[1:]):
            assert s1 + n1 - s2 >= 8  # at least the requested overlap


def test_windows_capped_at_max():
    windows, used = ag.plan_windows(1000)
    assert len(windows) == 4
    assert used == 121 + 3 * 113
    assert windows == [(0, 121), (113, 121), (226, 121), (339, 121)]


def test_blend_weights_sum_to_one_and_crossfade():
    for frames in (121, 200, 235, 300, 460):
        windows, used = ag.plan_windows(frames)
        weights = ag.blend_weights(windows, used)
        assert len(weights) == used
        for parts in weights:
            assert parts and abs(sum(p[2] for p in parts) - 1.0) < 1e-9
        assert weights[0] == [(0, 0, 1.0)]
        assert weights[-1][-1][0] == len(windows) - 1
    # two windows: weight of the first falls monotonically across the overlap
    windows, used = ag.plan_windows(230)
    w = ag.blend_weights(windows, used)
    first = [dict((p[0], p[2]) for p in parts).get(0, 0.0) for parts in w]
    assert all(a >= b for a, b in zip(first, first[1:]))


def test_settings_plan_and_info():
    class Clip:
        shape = (300, 2160, 3840, 3)
    plan, windows, used, seed, info = ag.AlphaGenSettings().plan(Clip(), ag.SIZE_FULL, 121, 8, 4, 1234, False, fps=24.0)
    assert windows == 3 and used == 300 and seed == 1234
    assert plan["work"] == (1920, 1080) and plan["pad"] == (1920, 1088) and plan["source_size"] == (3840, 2160)
    assert "scaled down" in info and "3 window" in info


def test_lazy_status_only_needs_planned_windows():
    node = ag.AlphaGenMatteStitch()
    plan = {"windows": [(0, 121), (100, 121)]}
    assert node.check_lazy_status(plan) == ["matte_1", "matte_2"]
    assert node.check_lazy_status(plan, matte_1=object()) == ["matte_2"]


def test_registered_in_both_mappings():
    for name in ag.ALPHAGEN_NODES:
        assert name in NODE_CLASS_MAPPINGS and name in NODE_DISPLAY_NAME_MAPPINGS


# ---------------- tensor nodes ----------------

def test_window_and_stitch_roundtrip():
    torch = pytest.importorskip("torch")
    frames, h, w = 230, 270, 480
    clip = torch.rand(frames, h, w, 3)
    plan, n, used, *_ = ag.AlphaGenSettings().plan(clip, ag.SIZE_FULL, 121, 8, 4, 1, False)
    assert plan["pad"] == (480, 288)
    mattes = {}
    for i in range(1, n + 1):
        win, pw, ph, length = ag.AlphaGenWindow().window(clip, plan, i)
        assert win.shape == (length, 288, 480, 3) and (length - 1) % 8 == 0
        assert torch.all(win[:, 270:] == 0)  # black padding
        mattes[f"matte_{i}"] = win  # identity "model": matte = its own input
    matte, mask = ag.AlphaGenMatteStitch().stitch(plan, **mattes)
    assert matte.shape == (frames, h, w, 3) and mask.shape == (frames, h, w)
    assert torch.allclose(mask, clip.mean(dim=-1), atol=1e-5)  # windows agree, so the blend reproduces the input


def test_stitch_scales_back_and_inverts():
    torch = pytest.importorskip("torch")
    clip = torch.ones(17, 2160, 3840, 3)
    plan, *_ = ag.AlphaGenSettings().plan(clip, ag.SIZE_HALF, 121, 8, 4, 1, True)
    win, *_ = ag.AlphaGenWindow().window(clip, plan, 1)
    assert win.shape[1:3] == (544, 960)
    matte, mask = ag.AlphaGenMatteStitch().stitch(plan, matte_1=win)
    assert mask.shape == (17, 2160, 3840)
    assert float(mask.max()) < 1e-4  # white clip, inverted -> black
