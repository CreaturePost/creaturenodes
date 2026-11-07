"""Helper nodes for the LTX Alpha Gen branch of the master VFX workflow.

LTX Alpha Gen (Lightricks, LTX-2.5 22B IC-LoRA, beta 0.9) turns an RGB clip into a frame-aligned grayscale
alpha matte. Official limits (docs.ltx.io/open-source-model/vfx-post-production/alpha-gen, model card, checked
2026-11-07):

- output no larger than 1920x1088; width and height divisible by 32; pad, don't stretch
- frame count 8n+1; 121 frames or fewer for the most predictable results; above 145 frames RGB leaks into the matte
- longer shots: split into segments of 121 frames or fewer and combine the mattes afterwards

These nodes do the parts a static graph can't: plan the windows for a clip of any length, cut and fit one window
(scale down if needed, pad to multiples of 32, pad the frame count to 8n+1), and stitch the window mattes back
into one matte at the source size, with a linear crossfade where windows overlap. The crossfade is ours, not an
official Lightricks recipe.

The planning maths is pure Python so it can be tested without torch or ComfyUI.
"""

import math

LONG_LIMIT = 1920
SHORT_LIMIT = 1088
MULTIPLE = 32
WINDOW_FRAMES = 121
MAX_FRAMES = 145
MAX_WINDOWS = 4
OVERLAP = 8

SIZE_FULL = "Full (fit inside 1920x1088)"
SIZE_HALF = "Half (faster, less VRAM)"
SIZES = [SIZE_FULL, SIZE_HALF]


# ------------------------------------------------------------------ pure planning maths

def frames_8n1(length):
    """Smallest 8n+1 frame count >= length."""
    length = max(1, int(length))
    return ((length - 1 + 7) // 8) * 8 + 1


def ceil_to(value, multiple=MULTIPLE):
    return int(math.ceil(value / multiple) * multiple)


def fit_size(width, height, working_size=SIZE_FULL):
    """Working size (scaled, aspect kept) and padded size (multiples of 32) for a source of width x height.

    The long side is limited to 1920 and the short side to 1088 (so portrait clips get 1088x1920 at most).
    Clips that already fit are not scaled up. Half mode scales the result by 0.5.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    long_side, short_side = max(width, height), min(width, height)
    scale = min(1.0, LONG_LIMIT / long_side, SHORT_LIMIT / short_side)
    if working_size == SIZE_HALF:
        scale *= 0.5
    elif working_size != SIZE_FULL:
        raise ValueError(f"unknown working size {working_size!r}")
    work_w = max(MULTIPLE, int(round(width * scale)))
    work_h = max(MULTIPLE, int(round(height * scale)))
    pad_w, pad_h = ceil_to(work_w), ceil_to(work_h)
    # rounding can push a side just past the limit; pull it back
    lim_w, lim_h = (LONG_LIMIT, SHORT_LIMIT) if width >= height else (SHORT_LIMIT, LONG_LIMIT)
    if pad_w > lim_w:
        work_w, pad_w = lim_w, lim_w
    if pad_h > lim_h:
        work_h, pad_h = lim_h, lim_h
    return work_w, work_h, pad_w, pad_h


def plan_windows(frames, window=WINDOW_FRAMES, overlap=OVERLAP, max_windows=MAX_WINDOWS):
    """Split a clip of `frames` frames into windows of at most `window` frames.

    Returns (windows, frames_used) where windows is a list of (start, length).
    Windows overlap by at least `overlap` frames and are spread evenly so every window is full length.
    If the clip needs more than `max_windows`, only the first frames_used frames are covered.
    """
    frames = int(frames)
    window = int(window)
    overlap = int(overlap)
    if frames <= 0:
        raise ValueError("clip has no frames")
    if not 9 <= window <= MAX_FRAMES:
        raise ValueError(f"window must be 9 to {MAX_FRAMES} frames")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be at least 0 and less than the window")
    max_windows = max(1, int(max_windows))
    if frames <= window:
        return [(0, frames)], frames
    step = window - overlap
    needed = 1 + math.ceil((frames - window) / step)
    if needed > max_windows:
        used = window + (max_windows - 1) * step
        return [(i * step, window) for i in range(max_windows)], used
    span = frames - window
    starts = [round(i * span / (needed - 1)) for i in range(needed)]
    return [(s, window) for s in starts], frames


def blend_weights(windows, frames_used, ramp=OVERLAP):
    """Per output frame: list of (window index, frame within window, weight). Weights sum to 1.

    Each window's weight ramps up linearly over its first `ramp` frames and down over its last `ramp` frames
    (not at the clip's own start or end), then the weights of all windows covering a frame are normalised.
    With two windows this is a linear crossfade across their overlap; it also handles three-way overlaps,
    which evenly spread windows can produce on short clips.
    """
    ramp = max(1, int(ramp))
    first_start = min(s for s, _ in windows)
    last_end = max(s + n for s, n in windows)
    out = []
    for f in range(frames_used):
        parts = []
        for i, (s, n) in enumerate(windows):
            if not s <= f < s + n:
                continue
            up = 1.0 if s == first_start else min(1.0, (f - s + 1) / (ramp + 1))
            down = 1.0 if s + n == last_end else min(1.0, (s + n - f) / (ramp + 1))
            parts.append((i, f - s, min(up, down)))
        total = sum(p[2] for p in parts)
        out.append([(i, local, w / total) for i, local, w in parts])
    return out


def describe_plan(src_w, src_h, frames, fps, windows, frames_used, work, pad):
    secs = f" ({frames / fps:.1f} s)" if fps else ""
    lines = [
        f"Clip: {src_w}x{src_h}, {frames} frames{secs}.",
        f"Working size {work[0]}x{work[1]}, padded to {pad[0]}x{pad[1]}"
        + (" (scaled down to fit 1920x1088)." if (work[0], work[1]) != (src_w, src_h) else "."),
        f"{len(windows)} window(s): " + ", ".join(f"{s}-{s + n - 1}" for s, n in windows) + ".",
    ]
    if frames_used < frames:
        lines.append(f"WARNING: only the first {frames_used} frames are matted (max windows reached). "
                     f"Raise max windows or run the rest separately with skip_first_frames = {frames_used}.")
    return "\n".join(lines)


# ------------------------------------------------------------------ tensor helpers (torch imported lazily)

def _resize(images, width, height):
    """IMAGE [N, H, W, C] -> [N, height, width, C], area/bicubic with antialias."""
    import torch.nn.functional as F

    if images.shape[1] == height and images.shape[2] == width:
        return images
    x = images.movedim(-1, 1)
    x = F.interpolate(x, size=(height, width), mode="bicubic", antialias=True, align_corners=False)
    return x.movedim(1, -1).clamp(0.0, 1.0)


# ------------------------------------------------------------------ nodes

class AlphaGenSettings:
    """Plans an Alpha Gen run: working size, padding and the 121-frame windows for the whole clip."""

    CATEGORY = "CreatureNodes/Alpha Gen"
    RETURN_TYPES = ("ALPHAGEN_PLAN", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("plan", "windows", "frames_used", "seed", "info")
    FUNCTION = "plan"
    DESCRIPTION = (
        "LTX Alpha Gen settings. The clip is scaled down if it is bigger than 1920x1088 (official limit), padded with "
        "black to multiples of 32 and split into windows of 121 frames (official recommendation; up to 145 may work but "
        "is less reliable). Windows overlap and are crossfaded (our addition). Only the windows the clip needs run."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The source clip (RGB)."}),
                "working_size": (SIZES, {"default": SIZE_FULL}),
                "window_frames": ("INT", {"default": WINDOW_FRAMES, "min": 9, "max": MAX_FRAMES, "step": 8,
                                          "tooltip": "121 = official recommendation. 129-145 may work but is less reliable."}),
                "overlap": ("INT", {"default": OVERLAP, "min": 0, "max": 32,
                                    "tooltip": "Minimum frames shared by neighbouring windows; crossfaded."}),
                "max_windows": ("INT", {"default": MAX_WINDOWS, "min": 1, "max": MAX_WINDOWS}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "1234 = model card samples. Try another seed if some frames look wrong."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Swap foreground and background in the output."}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 240.0, "step": 0.001, "forceInput": True}),
            },
        }

    def plan(self, images, working_size, window_frames, overlap, max_windows, seed, invert, fps=None):
        n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        work_w, work_h, pad_w, pad_h = fit_size(w, h, working_size)
        windows, used = plan_windows(n, window_frames, overlap, max_windows)
        plan = {
            "source_size": (w, h), "frames": n, "work": (work_w, work_h), "pad": (pad_w, pad_h),
            "windows": windows, "frames_used": used, "overlap": int(overlap), "invert": bool(invert),
        }
        info = describe_plan(w, h, n, fps, windows, used, (work_w, work_h), (pad_w, pad_h))
        return (plan, len(windows), used, int(seed), info)


class AlphaGenWindow:
    """Cuts window <index> from the clip and makes it Alpha Gen ready: working size, black padding to multiples of 32,
    frame count padded to 8n+1 by repeating the last frame."""

    CATEGORY = "CreatureNodes/Alpha Gen"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("frames", "width", "height", "length")
    FUNCTION = "window"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "plan": ("ALPHAGEN_PLAN",),
                "index": ("INT", {"default": 1, "min": 1, "max": MAX_WINDOWS}),
            }
        }

    def window(self, images, plan, index):
        import torch

        windows = plan["windows"]
        if not 1 <= index <= len(windows):
            raise ValueError(f"AlphaGenWindow: window {index} is not in the plan ({len(windows)} windows)")
        start, length = windows[index - 1]
        clip = images[start:start + length]
        work_w, work_h = plan["work"]
        pad_w, pad_h = plan["pad"]
        clip = _resize(clip, work_w, work_h)
        if (pad_w, pad_h) != (work_w, work_h):
            padded = torch.zeros((clip.shape[0], pad_h, pad_w, clip.shape[3]), dtype=clip.dtype, device=clip.device)
            padded[:, :work_h, :work_w, :] = clip
            clip = padded
        target = frames_8n1(clip.shape[0])
        if target > clip.shape[0]:
            clip = torch.cat([clip, clip[-1:].repeat(target - clip.shape[0], 1, 1, 1)], dim=0)
        return (clip, pad_w, pad_h, target)


class AlphaGenMatteStitch:
    """Joins the window mattes into one matte for the clip: trims the padded frames, crops the padding, crossfades the
    overlaps and scales back to the source size. Only the windows in the plan are evaluated (lazy inputs)."""

    CATEGORY = "CreatureNodes/Alpha Gen"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("matte", "mask")
    FUNCTION = "stitch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"plan": ("ALPHAGEN_PLAN",)},
            "optional": {f"matte_{i}": ("IMAGE", {"lazy": True}) for i in range(1, MAX_WINDOWS + 1)},
        }

    def check_lazy_status(self, plan, **kwargs):
        names = [f"matte_{i}" for i in range(1, len(plan["windows"]) + 1)]
        return [n for n in names if kwargs.get(n) is None]

    def stitch(self, plan, **kwargs):
        import torch

        windows = plan["windows"]
        work_w, work_h = plan["work"]
        mattes = []
        for i, (_, length) in enumerate(windows, start=1):
            m = kwargs.get(f"matte_{i}")
            if m is None:
                raise ValueError(f"AlphaGenMatteStitch: matte_{i} is not connected")
            if m.shape[0] < length:
                raise ValueError(f"AlphaGenMatteStitch: window {i} returned {m.shape[0]} frames, expected {length}")
            mattes.append(m[:length, :work_h, :work_w, :].mean(dim=-1))
        used = plan["frames_used"]
        out = torch.zeros((used, work_h, work_w), dtype=mattes[0].dtype, device=mattes[0].device)
        for f, parts in enumerate(blend_weights(windows, used, plan.get("overlap", OVERLAP))):
            for wi, local, weight in parts:
                out[f] += mattes[wi][local] * weight
        src_w, src_h = plan["source_size"]
        if (src_w, src_h) != (work_w, work_h):
            out = _resize(out.unsqueeze(-1), src_w, src_h).squeeze(-1)
        out = out.clamp(0.0, 1.0)
        if plan.get("invert"):
            out = 1.0 - out
        return (out.unsqueeze(-1).expand(-1, -1, -1, 3), out)


ALPHAGEN_NODES = {
    "Alpha Gen Settings": AlphaGenSettings,
    "Alpha Gen Window": AlphaGenWindow,
    "Alpha Gen Matte Stitch": AlphaGenMatteStitch,
}
