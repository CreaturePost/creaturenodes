"""Tests for the LongCat helper nodes and the LongCat switch options (no torch or ComfyUI needed)."""

import pytest

from src.creaturenodes import longcat as lc
from src.creaturenodes.switches import I2VModelSwitch, ImageEditingModelSwitch

FAST = "Fast (distill LoRA, 16 steps, CFG 1)"
FULL = "Full quality (50 steps, CFG 4)"


class FakeBatch(list):
    """List that supports tensor-style list indexing and a .shape, like an IMAGE tensor [N, H, W, C]."""

    def __init__(self, items, h=1080, w=1920):
        super().__init__(items)
        self.shape = (len(items), h, w, 3)

    def __getitem__(self, key):
        if isinstance(key, list):
            return FakeBatch([list.__getitem__(self, k) for k in key], self.shape[1], self.shape[2])
        return list.__getitem__(self, key)


# ---------------- settings and segment planning ----------------

def test_modes_match_official_values():
    node = lc.LongCatSettings()
    steps, cfg, lora, *_ = node.settings(FAST, lc.TASK_I2V, 5.0, "13")
    assert (steps, cfg, lora) == (16, 1.0, 1.0)
    steps, cfg, lora, *_ = node.settings(FULL, lc.TASK_I2V, 5.0, "13")
    assert (steps, cfg, lora) == (50, 4.0, 0.0)


@pytest.mark.parametrize("seconds, segments, frames", [
    (5.0, 1, 76),     # 76 frames fit in one 93-frame segment
    (6.1, 1, 93),     # 93 frames exactly
    (6.2, 2, 94),     # one frame more needs a second segment
    (15.0, 3, 226),   # 93 + 80 + 80 = 253 >= 226
    (60.0, 6, 493),   # capped at 6 segments: 93 + 5 * 80
])
def test_i2v_plan(seconds, segments, frames):
    assert lc.plan_segments(lc.TASK_I2V, seconds, 13) == (segments, frames)


@pytest.mark.parametrize("seconds, segments, frames", [
    (5.0, 1, 75),
    (5.4, 2, 81),     # 81 new frames > 80 per segment
    (10.0, 2, 150),
    (60.0, 6, 480),   # capped: 6 * 80
])
def test_extend_plan(seconds, segments, frames):
    assert lc.plan_segments(lc.TASK_EXTEND, seconds, 13) == (segments, frames)


def test_context_17_adds_fewer_frames_per_segment():
    assert lc.plan_segments(lc.TASK_EXTEND, 10.0, 17) == (2, 150)  # 76 per segment
    assert lc.plan_segments(lc.TASK_EXTEND, 10.2, 17) == (3, 153)


def test_max_segments_respected():
    assert lc.plan_segments(lc.TASK_I2V, 30.0, 13, max_segments=2) == (2, 173)


def test_settings_outputs_shape():
    node = lc.LongCatSettings()
    out = node.settings(FAST, lc.TASK_EXTEND, 5.0, "17", 6)
    assert len(out) == len(lc.LongCatSettings.RETURN_TYPES)
    assert out[3] == 17 and out[6] == 15.0


# ---------------- lazy index switch ----------------

def test_lazy_switch_requests_only_the_index():
    node = lc.LazyImageIndexSwitch()
    assert node.check_lazy_status(3) == ["image_3"]
    assert node.check_lazy_status(3, image_3="x") == []
    assert node.check_lazy_status(99) == ["image_6"]  # clamped
    assert node.pick(2, image_1="a", image_2="b") == ("b",)
    with pytest.raises(ValueError):
        node.pick(4, image_1="a")


def test_lazy_switch_inputs_are_lazy():
    opt = lc.LazyImageIndexSwitch.INPUT_TYPES()["optional"]
    assert len(opt) == lc.MAX_SEGMENTS
    assert all(v[1]["lazy"] for v in opt.values())


# ---------------- frame rate resample ----------------

def test_resample_24_to_15_keeps_duration():
    idx = lc.resample_indices(81, 24.0, 15.0)
    assert len(idx) == 51          # 80/24 s = 3.33 s -> 50 intervals at 15 fps
    assert idx[0] == 0 and idx[-1] == 80
    assert idx == sorted(idx)


def test_resample_30_to_24_and_skip_first():
    idx = lc.resample_indices(31, 30.0, 24.0, skip_first=True)
    assert idx[0] == 1 and idx[-1] == 30 and len(idx) == 24


def test_resample_identity():
    assert lc.resample_indices(10, 24.0, 24.0) == list(range(10))


def test_resample_node_returns_count():
    batch = FakeBatch(list(range(81)))
    images, count = lc.FrameRateResample().resample(batch, 24.0, 15.0)
    assert count == 51 and len(images) == 51


# ---------------- buckets and prompt lines ----------------

@pytest.mark.parametrize("w, h, expected", [
    (1920, 1080, (832, 480)),
    (1080, 1920, (480, 832)),
    (1024, 1024, (608, 640)),  # 0.95 and 1.05 tie; the official table lists 0.95 first
    (832, 480, (832, 480)),
])
def test_bucket(w, h, expected):
    assert lc.nearest_bucket(w, h) == expected


def test_bucket_node_reads_image_shape():
    assert lc.LongCatBucket().bucket(FakeBatch([0], h=1080, w=1920)) == (832, 480)


def test_buckets_are_multiples_of_16():
    assert all(w % 16 == 0 and h % 16 == 0 for w, h in lc.BUCKETS_480P)


def test_prompt_lines():
    text = "a woman slices bread\n\nshe pours milk\n  she drinks  \n"
    assert lc.pick_line(text, 1) == "a woman slices bread"
    assert lc.pick_line(text, 2) == "she pours milk"
    assert lc.pick_line(text, 3) == "she drinks"
    assert lc.pick_line(text, 6) == "she drinks"  # last line repeats
    assert lc.pick_line("one prompt", 4) == "one prompt"
    assert lc.pick_line("", 1) == ""


# ---------------- switch options ----------------

def test_i2v_longcat_option():
    node = I2VModelSwitch()
    assert "LongCat Video" in I2VModelSwitch.INPUT_TYPES()["required"]["selected_model"][0]
    image, audio, fps = node.i2v_model_switch("LongCat Video", frame_rate=24.0, longcat_image="frames")
    assert image == "frames" and audio is None and fps == 30.0


def test_v2v_longcat_option():
    assert "LongCat Extend" in ImageEditingModelSwitch.INPUT_TYPES()["required"]["selected_model"][0]
    node = ImageEditingModelSwitch()
    assert node.image_editing_model_switch("LongCat Extend", longcat_extend="frames") == ("frames",)
