"""Tests for the creaturenodes model switches.

These import switches.py directly, so they don't need google-genai, torch or ComfyUI.
"""

import pytest

from src.creaturenodes.switches import (
    I2IModelSwitch,
    I2VModelSwitch,
    ImageEditingModelSwitch,
    T2IModelSwitch,
)

ALL_SWITCHES = [T2IModelSwitch, I2VModelSwitch, I2IModelSwitch, ImageEditingModelSwitch]

# Labels and input names saved in existing workflows (master_vfx_workflow_260825).
# These must never be renamed or removed.
LEGACY = {
    T2IModelSwitch: (["SD3.5", "SDXL", "Z IMAGE", "Flux2 Klein"], ["sd35", "sdxl", "z_image", "flux2_klein"]),
    I2VModelSwitch: (
        ["LTX 2.0", "MiniMax H3", "MiniMax H3 Ref"],
        ["ltx_20_image", "ltx_20_audio", "mm_h3_image", "mm_h3_audio", "mm_h3_ref_image", "mm_h3_ref_audio"],
    ),
    I2IModelSwitch: (["Flux1 Dev Uso", "Flux1 Kontext"], ["flux1_dev_uso", "flux1_kontext"]),
    ImageEditingModelSwitch: (["Lucy Edit V1.1 Dev", "Seedance 2.5"], ["lucy_edit_v11_dev", "seedance_25"]),
}


@pytest.mark.parametrize("cls", ALL_SWITCHES)
def test_legacy_labels_and_inputs_kept(cls):
    labels, inputs = LEGACY[cls]
    types = cls.INPUT_TYPES()
    combo = types["required"]["selected_model"][0]
    for label in labels:
        assert label in combo
    for name in inputs:
        assert name in types["optional"]


@pytest.mark.parametrize("cls", ALL_SWITCHES)
def test_every_option_input_is_declared_and_lazy(cls):
    optional = cls.INPUT_TYPES()["optional"]
    for label, names in cls.OPTIONS.items():
        assert names, label
        for name in names:
            assert name in optional, (label, name)
            assert optional[name][1].get("lazy") is True


@pytest.mark.parametrize("cls", ALL_SWITCHES)
def test_lazy_requests_only_selected_branch(cls):
    node = cls()
    for label, names in cls.OPTIONS.items():
        assert node.check_lazy_status(label) == names
        # once the selected branch is evaluated, nothing more is requested
        filled = {n: object() for n in names}
        assert node.check_lazy_status(label, **filled) == []


@pytest.mark.parametrize("cls", ALL_SWITCHES)
def test_returns_selected_branch(cls):
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    all_inputs = {name: f"value:{name}" for name in cls.INPUTS}
    for label, names in cls.OPTIONS.items():
        out = fn(label, **all_inputs)
        assert len(out) == len(cls.RETURN_TYPES)
        assert out[0] == f"value:{names[0]}"


def test_t2i_new_klein_options():
    combo = T2IModelSwitch.INPUT_TYPES()["required"]["selected_model"][0]
    assert "Flux2 Klein 4B" in combo and "Flux2 Klein 9B" in combo


def test_i2i_new_options():
    combo = I2IModelSwitch.INPUT_TYPES()["required"]["selected_model"][0]
    for label in ["Qwen Image Edit 2511", "Flux2 Klein 4B", "Flux2 Klein 9B"]:
        assert label in combo


def test_i2v_audio_and_fps():
    node = I2VModelSwitch()
    # Wan has no audio branch; output audio must be None (VHS_VideoCombine accepts None)
    image, audio, fps = node.i2v_model_switch("Wan 2.2", wan_22_image="img")
    assert image == "img" and audio is None and fps == 16.0
    # LTX follows the frame_rate passed in, defaulting to 24
    assert node.i2v_model_switch("LTX 2.5", frame_rate=25, ltx_25_image="i", ltx_25_audio="a") == ("i", "a", 25.0)
    assert node.i2v_model_switch("LTX 2.0", ltx_20_image="i", ltx_20_audio="a")[2] == 24.0
    # MiniMax and Seedance have fixed native rates regardless of frame_rate
    assert node.i2v_model_switch("MiniMax H3", frame_rate=30, mm_h3_image="i", mm_h3_audio="a")[2] == 24.0
    assert node.i2v_model_switch("Seedance 2.5", seedance_25_image="i", seedance_25_audio="a")[2] == 24.0
    assert node.check_lazy_status("Wan 2.2") == ["wan_22_image"]


def test_i2v_frame_rate_is_not_lazy():
    spec = I2VModelSwitch.INPUT_TYPES()["optional"]["frame_rate"]
    assert spec[0] == "FLOAT" and not spec[1].get("lazy")


def test_v2v_new_options():
    combo = ImageEditingModelSwitch.INPUT_TYPES()["required"]["selected_model"][0]
    for label in ["Wan 2.2 Fun Control", "LTX 2.3 IC-LoRA", "MiniMax H3 Ref"]:
        assert label in combo
