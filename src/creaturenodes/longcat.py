"""Helper nodes for the LongCat-Video branches of the master VFX workflow.

LongCat-Video (Meituan, MIT) runs through Kijai's ComfyUI-WanVideoWrapper. These nodes hold the
official settings and the small bits of logic a static ComfyUI graph can't express easily:
how many continuation segments to run, picking the last needed segment lazily, resampling
frame rates by timestamp, the official 480p resolution buckets and one prompt per segment.

Settings source: github.com/meituan-longcat/LongCat-Video run_demo_*.py and
longcat_video/pipeline_longcat_video.py (checked 2026-11-07).

Pure Python apart from tensor indexing, so the tests don't need torch or ComfyUI.
"""

import math

NATIVE_FPS = 15.0
SEGMENT_FRAMES = 93  # frames per generated segment (official default, 4n+1)
MAX_SEGMENTS = 6

MODES = {
    # label: (steps, cfg, distill LoRA strength)
    "Fast (distill LoRA, 16 steps, CFG 1)": (16, 1.0, 1.0),
    "Full quality (50 steps, CFG 4)": (50, 4.0, 0.0),
}
TASK_I2V = "I2V: total duration"
TASK_EXTEND = "Extend: add seconds"

# Official 480p buckets (bukcet_config.py ASPECT_RATIO_627), as (width, height).
BUCKETS_480P = [
    (320, 1216), (352, 1120), (384, 1024), (416, 960), (448, 864), (480, 832), (512, 768),
    (544, 736), (576, 672), (608, 640), (640, 608), (672, 576), (704, 544), (736, 544),
    (768, 512), (800, 480), (832, 480), (896, 448), (960, 416), (992, 384), (1056, 384),
    (1088, 352), (1184, 320), (1216, 320), (1248, 320), (1280, 320),
]


def plan_segments(task, seconds, context_frames, max_segments=MAX_SEGMENTS):
    """Return (segments, frames) at 15 fps.

    I2V: frames = total output length; segment 1 makes 93 frames from the start image and
    every further segment adds 93 - context frames.
    Extend: frames = number of new frames to add after the source clip.
    """
    new_per_segment = SEGMENT_FRAMES - context_frames
    if task == TASK_I2V:
        total = max(1, round(seconds * NATIVE_FPS)) + 1
        segments = 1 + math.ceil(max(0, total - SEGMENT_FRAMES) / new_per_segment)
        segments = max(1, min(max_segments, segments))
        frames = min(total, SEGMENT_FRAMES + (segments - 1) * new_per_segment)
    elif task == TASK_EXTEND:
        new = max(1, round(seconds * NATIVE_FPS))
        segments = max(1, min(max_segments, math.ceil(new / new_per_segment)))
        frames = min(new, segments * new_per_segment)
    else:
        raise ValueError(f"unknown task {task!r}")
    return segments, frames


def resample_indices(count, source_fps, target_fps, skip_first=False):
    """Indices of the nearest source frame for each output frame, covering the same time span."""
    if count <= 0:
        return []
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("fps must be positive")
    duration = (count - 1) / source_fps
    n_out = int(math.floor(duration * target_fps + 1e-6)) + 1
    idx = [min(count - 1, int(round(i * source_fps / target_fps))) for i in range(n_out)]
    return idx[1:] if skip_first else idx


def nearest_bucket(width, height, buckets=BUCKETS_480P):
    ratio = width / height
    return min(buckets, key=lambda wh: abs(wh[0] / wh[1] - ratio))


def pick_line(text, index):
    """1-based line of a multi-line prompt; empty lines are skipped and the last line repeats."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[max(0, min(index, len(lines)) - 1)]


class LongCatSettings:
    """All user-facing LongCat controls in one node, with the official values."""

    CATEGORY = "CreatureNodes/LongCat"
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("steps", "cfg", "distill_lora_strength", "context_frames", "segments", "frames", "fps")
    FUNCTION = "settings"
    DESCRIPTION = (
        "LongCat-Video settings. Fast = official distill LoRA path (16 steps, CFG 1). Full quality = 50 steps, CFG 4 "
        "with the negative prompt. Context frames: 13 is official; 17 is experimental. Segments are 93 frames at 15 fps; "
        "each continuation segment adds 93 - context frames. 'frames' is the output length at 15 fps."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(MODES),),
                "task": ([TASK_I2V, TASK_EXTEND],),
                "seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "context_frames": (["13", "17"],),
                "max_segments": ("INT", {"default": MAX_SEGMENTS, "min": 1, "max": MAX_SEGMENTS}),
            }
        }

    def settings(self, mode, task, seconds, context_frames, max_segments=MAX_SEGMENTS):
        steps, cfg, lora = MODES[mode]
        ctx = int(context_frames)
        segments, frames = plan_segments(task, float(seconds), ctx, int(max_segments))
        return (steps, cfg, lora, ctx, segments, frames, NATIVE_FPS)


class LazyImageIndexSwitch:
    """Returns image_<index>. Only that input is evaluated, so later segments in a chain never run."""

    CATEGORY = "CreatureNodes/LongCat"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pick"
    COUNT = MAX_SEGMENTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"index": ("INT", {"default": 1, "min": 1, "max": cls.COUNT})},
            "optional": {f"image_{i}": ("IMAGE", {"lazy": True}) for i in range(1, cls.COUNT + 1)},
        }

    def _name(self, index):
        return f"image_{max(1, min(self.COUNT, int(index)))}"

    def check_lazy_status(self, index, **kwargs):
        name = self._name(index)
        return [name] if kwargs.get(name) is None else []

    def pick(self, index, **kwargs):
        image = kwargs.get(self._name(index))
        if image is None:
            raise ValueError(f"LazyImageIndexSwitch: {self._name(index)} is not connected")
        return (image,)


class FrameRateResample:
    """Change an image batch's frame rate by picking the nearest frame in time (no blending)."""

    CATEGORY = "CreatureNodes/LongCat"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "resample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.001}),
                "target_fps": ("FLOAT", {"default": 15.0, "min": 0.1, "max": 240.0, "step": 0.001}),
                "skip_first": ("BOOLEAN", {"default": False, "tooltip": "Drop output frame 0 (e.g. a context frame already in the clip)."}),
            }
        }

    def resample(self, images, source_fps, target_fps, skip_first=False):
        idx = resample_indices(len(images), float(source_fps), float(target_fps), bool(skip_first))
        if not idx:
            raise ValueError("FrameRateResample: no frames left after resampling")
        return (images[idx], len(idx))


class LongCatBucket:
    """Official LongCat 480p resolution bucket closest to the image's aspect ratio."""

    CATEGORY = "CreatureNodes/LongCat"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "bucket"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def bucket(self, image):
        h, w = int(image.shape[1]), int(image.shape[2])
        return nearest_bucket(w, h)


class PromptLinePicker:
    """Line <index> of a multi-line prompt (1-based). Blank lines are skipped; the last line repeats."""

    CATEGORY = "CreatureNodes/LongCat"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "pick"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "index": ("INT", {"default": 1, "min": 1, "max": 99}),
            }
        }

    def pick(self, text, index):
        return (pick_line(text, int(index)),)


class SeedOffset:
    """seed + offset (wraps at 2**64), e.g. a different seed for each LongCat segment."""

    CATEGORY = "CreatureNodes/LongCat"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "offset"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "offset": ("INT", {"default": 1, "min": 0, "max": 1000}),
            }
        }

    def offset(self, seed, offset):
        return ((int(seed) + int(offset)) % (1 << 64),)


LONGCAT_NODES = {
    "LongCat Settings": LongCatSettings,
    "Lazy Image Index Switch": LazyImageIndexSwitch,
    "Frame Rate Resample": FrameRateResample,
    "LongCat Bucket": LongCatBucket,
    "Prompt Line Picker": PromptLinePicker,
    "Seed Offset": SeedOffset,
}
