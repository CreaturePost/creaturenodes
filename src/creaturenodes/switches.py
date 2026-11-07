"""Model switch nodes.

Each switch has a ``selected_model`` combo and one lazy input per branch output.
Only the inputs of the selected branch are requested, so only that branch runs.

Rules for adding an option:
- Add it to the class's ``OPTIONS`` dict: label -> the input names that branch feeds.
- Add the input names to ``INPUTS`` with their type.
- Never rename or remove an existing label or input name. Saved workflows store
  the label string and connect links by input name.

Selecting an option whose branch is not wired makes ComfyUI raise
"Node ... says it needs input ..., but there is no input to that node at all".
That is intended: it tells the user the branch is not connected.
"""

LAZY = {"lazy": True}


class _LazySwitch:
    """Shared logic. Subclasses define OPTIONS and INPUTS."""

    CATEGORY = "CreatureNodes/Switches"
    OPTIONS: dict = {}
    INPUTS: dict = {}
    EXTRA_OPTIONAL: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        optional = {name: (typ, LAZY) for name, typ in cls.INPUTS.items()}
        optional.update(cls.EXTRA_OPTIONAL)
        return {
            "required": {"selected_model": (list(cls.OPTIONS),)},
            "optional": optional,
        }

    def check_lazy_status(self, selected_model, **kwargs):
        return [name for name in self.OPTIONS[selected_model] if kwargs.get(name) is None]

    def _selected(self, selected_model, kwargs):
        return tuple(kwargs.get(name) for name in self.OPTIONS[selected_model])


class T2IModelSwitch(_LazySwitch):
    # "Flux2 Klein" is the legacy label (loads Klein 9B base in the 260825 master).
    # It keeps its own input so older saved workflows still work.
    OPTIONS = {
        "SD3.5": ["sd35"],
        "SDXL": ["sdxl"],
        "Z IMAGE": ["z_image"],
        "Flux2 Klein": ["flux2_klein"],
        "Flux2 Klein 4B": ["flux2_klein_4b"],
        "Flux2 Klein 9B": ["flux2_klein_9b"],
    }
    INPUTS = {
        "sd35": "IMAGE",
        "sdxl": "IMAGE",
        "z_image": "IMAGE",
        "flux2_klein": "IMAGE",
        "flux2_klein_4b": "IMAGE",
        "flux2_klein_9b": "IMAGE",
    }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "t2i_model_switch"

    def t2i_model_switch(self, selected_model, **kwargs):
        return self._selected(selected_model, kwargs)


class I2VModelSwitch(_LazySwitch):
    OPTIONS = {
        "LTX 2.0": ["ltx_20_image", "ltx_20_audio"],
        "MiniMax H3": ["mm_h3_image", "mm_h3_audio"],
        "MiniMax H3 Ref": ["mm_h3_ref_image", "mm_h3_ref_audio"],
        "LTX 2.5": ["ltx_25_image", "ltx_25_audio"],
        "Wan 2.2": ["wan_22_image"],  # Wan has no audio
        "Seedance 2.5": ["seedance_25_image", "seedance_25_audio"],
        "LongCat Video": ["longcat_image"],  # no audio; RIFE-interpolated to 30 fps
    }
    INPUTS = {
        "ltx_20_image": "IMAGE",
        "ltx_20_audio": "AUDIO",
        "mm_h3_image": "IMAGE",
        "mm_h3_audio": "AUDIO",
        "mm_h3_ref_image": "IMAGE",
        "mm_h3_ref_audio": "AUDIO",
        "ltx_25_image": "IMAGE",
        "ltx_25_audio": "AUDIO",
        "wan_22_image": "IMAGE",
        "seedance_25_image": "IMAGE",
        "seedance_25_audio": "AUDIO",
        "longcat_image": "IMAGE",
    }
    # Frame rate the LTX branches are sampled at. Plain value, not lazy, so it is
    # cheap to always read. Other models render at a fixed native rate.
    EXTRA_OPTIONAL = {
        "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01, "forceInput": True}),
    }
    NATIVE_FPS = {
        "MiniMax H3": 24.0,
        "MiniMax H3 Ref": 24.0,
        "Wan 2.2": 16.0,
        "Seedance 2.5": 24.0,
        "LongCat Video": 30.0,
    }
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("image", "audio", "fps")
    FUNCTION = "i2v_model_switch"

    def i2v_model_switch(self, selected_model, frame_rate=None, **kwargs):
        names = self.OPTIONS[selected_model]
        image = kwargs.get(names[0])
        audio = kwargs.get(names[1]) if len(names) > 1 else None
        fps = self.NATIVE_FPS.get(selected_model)
        if fps is None:
            fps = float(frame_rate) if frame_rate is not None else 24.0
        return (image, audio, fps)


class I2IModelSwitch(_LazySwitch):
    OPTIONS = {
        "Flux1 Dev Uso": ["flux1_dev_uso"],
        "Flux1 Kontext": ["flux1_kontext"],
        "Qwen Image Edit 2511": ["qwen_image_edit_2511"],
        "Flux2 Klein 4B": ["flux2_klein_4b"],
        "Flux2 Klein 9B": ["flux2_klein_9b"],
    }
    INPUTS = {
        "flux1_dev_uso": "IMAGE",
        "flux1_kontext": "IMAGE",
        "qwen_image_edit_2511": "IMAGE",
        "flux2_klein_4b": "IMAGE",
        "flux2_klein_9b": "IMAGE",
    }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "i2i_model_switch"

    def i2i_model_switch(self, selected_model, **kwargs):
        return self._selected(selected_model, kwargs)


class ImageEditingModelSwitch(_LazySwitch):
    """Video editing (V2V) switch."""

    OPTIONS = {
        "Lucy Edit V1.1 Dev": ["lucy_edit_v11_dev"],
        "Seedance 2.5": ["seedance_25"],
        "Wan 2.2 Fun Control": ["wan_22_fun_control"],
        "LTX 2.3 IC-LoRA": ["ltx_23_ic_lora"],
        "MiniMax H3 Ref": ["mm_h3_ref"],
        "LongCat Extend": ["longcat_extend"],
    }
    INPUTS = {
        "lucy_edit_v11_dev": "IMAGE",
        "seedance_25": "IMAGE",
        "wan_22_fun_control": "IMAGE",
        "ltx_23_ic_lora": "IMAGE",
        "mm_h3_ref": "IMAGE",
        "longcat_extend": "IMAGE",
    }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_editing_model_switch"

    def image_editing_model_switch(self, selected_model, **kwargs):
        return self._selected(selected_model, kwargs)
