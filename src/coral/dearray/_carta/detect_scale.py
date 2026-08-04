"""Locked core-detection scale: source pixels to the YOLO detector's input.

Vendored from CARTA ``segmenter/core_detection_scale.py``. That file was
deliberately **not** vendored when the tissue segmenter came across:
``coral.tissue._carta.seg_scale`` records that it absorbed ``resample_to_target``
and dropped "its YOLO letterbox / box-remap / dearray-target helpers". Those are
what this module brings back, and nothing else.

The detector is scale-locked. It was trained on nuclear planes resampled to
:data:`TARGET_UM_PER_PX` and letterboxed into a
:data:`INPUT_SIZE` square, so inference has to reproduce that chain exactly or
the boxes come back in the wrong places. The chain is:

    source plane -> resample to 26.8 µm/px -> percentile stretch to uint8
                 -> letterbox into 1280 -> replicate to RGB

and box coordinates travel the inverse.

Edits versus upstream:

* ``prepare_slide_rgb`` is renamed :func:`prepare_detector_rgb`, because what it
  prepares is the detector's input and the caller need not be a whole slide.
* ``scale_boxes``, ``normalize_dapi_with_params``'s manifest tuple and
  ``prepare_slide_rgb``'s metadata extras are kept, since the run record wants
  the normalisation bounds that were actually used.
* Deprecated fixed-map constants (``FIXED_NORM_LO`` / ``FIXED_NORM_HI``) and the
  per-dataset µm/px constants for CARTA's own cohorts are dropped: CORAL reads
  mpp from the store.

The ``remap_boxes_*`` pair keeps its upstream names, and the ``l0`` in them is a
misnomer. Both are scale-agnostic: they undo the resample and letterbox relative
to whatever ``native_um_per_px`` the caller declared. CORAL feeds them a
**reduced pyramid level**, so they return coordinates in that level's space, not
level 0's. Reading the names as a promise of level-0 coordinates is the cheapest
possible route to a whole-slide coordinate offset that still looks plausible on
screen.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import zoom

# Settled upstream: tight fit of the largest slide (~34301 µm) into 1280.
TARGET_UM_PER_PX: float = 26.797918841894234
INPUT_SIZE: int = 1280

# Per-slide percentile stretch, applied AFTER resample and letterbox.
NORM_LO_PCT: float = 1.0
NORM_HI_PCT: float = 99.0

__all__ = [
    "INPUT_SIZE",
    "NORM_HI_PCT",
    "NORM_LO_PCT",
    "TARGET_UM_PER_PX",
    "dapi_to_rgb",
    "letterbox_to_input",
    "normalize_dapi",
    "normalize_dapi_with_params",
    "prepare_slide_rgb",
    "remap_boxes_input_to_l0",
    "remap_boxes_l0_to_input",
    "resample_to_target",
]


def resample_to_target(
    image: np.ndarray,
    native_um_per_px: float,
    *,
    target_um_per_px: float = TARGET_UM_PER_PX,
    order: int = 1,
) -> np.ndarray:
    """Resample so the result's µm/px equals ``target_um_per_px``.

    Large downsamples go through PIL bilinear, which is the same geometric
    factor as ``zoom`` and roughly 50-100x faster; factors near 1.0 keep
    ``scipy.ndimage.zoom``.
    """
    if native_um_per_px <= 0:
        raise ValueError(f"native_um_per_px must be > 0, got {native_um_per_px}")
    factor = float(native_um_per_px) / float(target_um_per_px)
    if abs(factor - 1.0) < 1e-9:
        return np.asarray(image)
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32)
    if factor < 0.85:
        from PIL import Image as _PILImage

        if arr.ndim == 2:
            h, w = arr.shape
            nh = max(1, int(round(h * factor)))
            nw = max(1, int(round(w * factor)))
            im = _PILImage.fromarray(arr.astype(np.float32), mode="F")
            return np.asarray(
                im.resize((nw, nh), _PILImage.BILINEAR), dtype=np.float32
            )
        if arr.ndim == 3:
            h, w, _ = arr.shape
            nh = max(1, int(round(h * factor)))
            nw = max(1, int(round(w * factor)))
            im = _PILImage.fromarray(arr.astype(np.uint8))
            return np.asarray(im.resize((nw, nh), _PILImage.BILINEAR))
    if arr.ndim == 2:
        return zoom(arr.astype(np.float32), factor, order=order)
    if arr.ndim == 3:
        return zoom(arr.astype(np.float32), (factor, factor, 1.0), order=order)
    raise ValueError(f"unsupported image ndim={arr.ndim}")


def normalize_dapi(
    image: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
    lo_pct: float = NORM_LO_PCT,
    hi_pct: float = NORM_HI_PCT,
) -> np.ndarray:
    """Per-slide percentile stretch to uint8.

    ``lo`` / ``hi`` default to the p1-p99 of on-slide pixels; zero-valued
    letterbox padding is excluded so padding cannot drag the window.
    """
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    sample = a[a > 0] if np.any(a > 0) else a.ravel()
    if lo is None:
        lo = float(np.percentile(sample, lo_pct))
    if hi is None:
        hi = float(np.percentile(sample, hi_pct))
    span = max(float(hi) - float(lo), 1e-6)
    return np.clip((a - float(lo)) / span * 255.0, 0, 255).astype(np.uint8)


def normalize_dapi_with_params(
    image: np.ndarray,
    *,
    lo_pct: float = NORM_LO_PCT,
    hi_pct: float = NORM_HI_PCT,
    hi_cap_median_mult: float = 8.0,
) -> tuple[np.ndarray, float, float]:
    """:func:`normalize_dapi` plus the bounds it used, for the run record."""
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    sample = a[a > 0] if np.any(a > 0) else a.ravel()
    lo = float(np.percentile(sample, lo_pct))
    hi = float(np.percentile(sample, hi_pct))
    if hi_cap_median_mult is not None and sample.size:
        med = float(np.median(sample))
        hi = min(hi, max(lo + 1.0, med * float(hi_cap_median_mult)))
    return normalize_dapi(a, lo=lo, hi=hi), lo, hi


def dapi_to_rgb(u8: np.ndarray) -> np.ndarray:
    """Replicate grayscale to three channels for YOLO. Not false colour."""
    if u8.ndim == 3 and u8.shape[-1] == 3:
        return u8
    return np.stack([u8, u8, u8], axis=-1)


def _resize_image(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Bilinear resize, matching :func:`resample_to_target`'s PIL path."""
    from PIL import Image as _PILImage

    arr = np.asarray(image)
    nw, nh = max(1, int(new_w)), max(1, int(new_h))
    if arr.ndim == 2:
        im = _PILImage.fromarray(arr.astype(np.float32), mode="F")
        out = np.asarray(im.resize((nw, nh), _PILImage.BILINEAR), dtype=np.float32)
        if np.issubdtype(image.dtype, np.integer):
            return out
        return out.astype(image.dtype, copy=False)
    if arr.ndim == 3:
        im = _PILImage.fromarray(arr.astype(np.uint8))
        return np.asarray(im.resize((nw, nh), _PILImage.BILINEAR))
    raise ValueError(f"unsupported image ndim={arr.ndim}")


def letterbox_to_input(
    image: np.ndarray,
    *,
    input_size: int = INPUT_SIZE,
    fill: int = 0,
    upscale_small: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pad or crop to exactly ``input_size`` square, preserving aspect.

    With ``upscale_small`` and a long edge below ``input_size``, the image is
    scaled up so the long edge fills it, which is the standard YOLO letterbox: a
    small field of view should not become a tiny island on black padding.
    Oversized images are centre-cropped.

    Returns ``(canvas, meta)``, where ``meta`` carries the offsets and scale that
    :func:`remap_boxes_input_to_l0` needs to invert.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        H, W = image.shape[:2]
        canvas = np.full(
            (input_size, input_size, image.shape[2]), fill, dtype=image.dtype
        )
    else:
        H, W = image.shape
        canvas = np.full((input_size, input_size), fill, dtype=image.dtype)

    src_h, src_w = int(H), int(W)
    lb_scale = 1.0
    if upscale_small and max(H, W) < input_size:
        lb_scale = float(input_size) / float(max(H, W))
        W = max(1, int(round(W * lb_scale)))
        H = max(1, int(round(H * lb_scale)))
        image = _resize_image(image, W, H)

    y0_src = max(0, (H - input_size) // 2)
    x0_src = max(0, (W - input_size) // 2)
    y1_src = min(H, y0_src + input_size)
    x1_src = min(W, x0_src + input_size)
    patch = image[y0_src:y1_src, x0_src:x1_src]
    ph, pw = patch.shape[:2]

    y0 = (input_size - ph) // 2
    x0 = (input_size - pw) // 2
    canvas[y0 : y0 + ph, x0 : x0 + pw] = patch
    meta: dict[str, Any] = {
        "offset_x": float(x0 - x0_src),
        "offset_y": float(y0 - y0_src),
        "letterbox_scale": float(lb_scale),
        "content_w": int(pw),
        "content_h": int(ph),
        "src_w": int(src_w),
        "src_h": int(src_h),
        "scaled_w": int(W),
        "scaled_h": int(H),
        "input_size": int(input_size),
    }
    return canvas, meta


def prepare_slide_rgb(
    nuclear: np.ndarray,
    native_um_per_px: float,
    *,
    target_um_per_px: float = TARGET_UM_PER_PX,
    input_size: int = INPUT_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The locked chain: resample, stretch, letterbox, replicate to RGB.

    ``native_um_per_px`` is the level-0 µm/px of the slide, as
    ``e2e_pipeline`` passes. Everything the inverse remap needs is in the
    returned metadata.
    """
    scaled = resample_to_target(
        nuclear, native_um_per_px, target_um_per_px=target_um_per_px
    )
    u8, lo, hi = normalize_dapi_with_params(scaled)
    canvas, meta = letterbox_to_input(u8, input_size=input_size, fill=0)
    rgb = dapi_to_rgb(canvas)
    meta["native_um_per_px"] = float(native_um_per_px)
    meta["target_um_per_px"] = float(target_um_per_px)
    meta["norm_lo_pct"] = NORM_LO_PCT
    meta["norm_hi_pct"] = NORM_HI_PCT
    meta["norm_lo"] = lo
    meta["norm_hi"] = hi
    return rgb, meta


def remap_boxes_l0_to_input(
    boxes_l0: list[tuple[float, float, float, float]],
    native_um_per_px: float,
    letterbox_meta: dict[str, Any],
    *,
    target_um_per_px: float = TARGET_UM_PER_PX,
) -> list[tuple[float, float, float, float]]:
    """Map L0 xyxy boxes through resample and letterbox into input coordinates."""
    factor = float(native_um_per_px) / float(target_um_per_px)
    lb_scale = float(letterbox_meta.get("letterbox_scale", 1.0))
    ox = float(letterbox_meta["offset_x"])
    oy = float(letterbox_meta["offset_y"])
    return [
        (
            b[0] * factor * lb_scale + ox,
            b[1] * factor * lb_scale + oy,
            b[2] * factor * lb_scale + ox,
            b[3] * factor * lb_scale + oy,
        )
        for b in boxes_l0
    ]


def remap_boxes_input_to_l0(
    boxes_input: list[tuple[float, float, float, float]],
    native_um_per_px: float,
    letterbox_meta: dict[str, Any],
    *,
    target_um_per_px: float = TARGET_UM_PER_PX,
) -> list[tuple[float, float, float, float]]:
    """Inverse of :func:`remap_boxes_l0_to_input`.

    Returns level-0 slide coordinates, given the level-0 ``native_um_per_px``
    that :func:`prepare_slide_rgb` was handed.
    """
    factor = float(native_um_per_px) / float(target_um_per_px)
    lb_scale = float(letterbox_meta.get("letterbox_scale", 1.0))
    denom = factor * lb_scale
    if abs(denom) < 1e-12:
        raise ValueError(f"degenerate box remap denominator: {denom}")
    ox = float(letterbox_meta["offset_x"])
    oy = float(letterbox_meta["offset_y"])
    return [
        (
            (float(b[0]) - ox) / denom,
            (float(b[1]) - oy) / denom,
            (float(b[2]) - ox) / denom,
            (float(b[3]) - oy) / denom,
        )
        for b in boxes_input
    ]
