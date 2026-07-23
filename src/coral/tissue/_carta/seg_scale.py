"""Shared DeepLab inference scale: native µm/px → 1.0 µm/px.

Vendored from CARTA ``segmenter/seg_scale.py`` + the resample helper from
``segmenter/core_detection_scale.py`` (merged here). Edits vs upstream:
``resample_to_target`` is absorbed from ``core_detection_scale.py`` (its YOLO
letterbox / box-remap / dearray-target helpers are not vendored), and
``mask_seg_to_native`` (from CARTA ``e2e_pipeline.py``) is added so a seg-scale
mask can be upsampled back to native.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

# Inference target = the model's ~L1 training scale (1.0 µm/px). The earlier
# 2.027 µm/px (L2) value was a speed-only downsample; inferring at the training
# scale is the more faithful default. Cleanup µm²/µm thresholds convert to px
# against this constant, so it is the single knob for the CARTA tissue scale.
SEG_INFERENCE_UM_PER_PX: float = 1.0


def validate_native_mpp(native_mpp: float | None) -> float:
    """Reject missing/invalid native mpp."""
    if native_mpp is None or float(native_mpp) <= 0.0:
        raise ValueError(
            "native_mpp is missing or invalid (0.0/None). "
            "Could not infer µm/px from file metadata or known scanner family."
        )
    return float(native_mpp)


def resample_to_target(
    image: np.ndarray,
    native_um_per_px: float,
    *,
    target_um_per_px: float,
    order: int = 1,
) -> np.ndarray:
    """Resample image so the resulting µm/px == ``target_um_per_px``.

    Large downsamples use PIL bilinear (same geometric factor as zoom) for
    speed; near-1.0 factors keep scipy.ndimage.zoom.
    """
    if native_um_per_px <= 0:
        raise ValueError(f"native_um_per_px must be > 0, got {native_um_per_px}")
    factor = float(native_um_per_px) / float(target_um_per_px)
    if abs(factor - 1.0) < 1e-9:
        return np.asarray(image)
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32)
    # Strong downsample (e.g. 0.5→2.027 µm/px): PIL is ~50–100× faster than zoom.
    if factor < 0.85:
        from PIL import Image as _PILImage
        if arr.ndim == 2:
            h, w = arr.shape
            nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
            im = _PILImage.fromarray(arr.astype(np.float32), mode="F")
            return np.asarray(im.resize((nw, nh), _PILImage.BILINEAR), dtype=np.float32)
        if arr.ndim == 3:
            h, w, c = arr.shape
            nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
            im = _PILImage.fromarray(arr.astype(np.uint8))
            return np.asarray(im.resize((nw, nh), _PILImage.BILINEAR))
    if arr.ndim == 2:
        return zoom(arr.astype(np.float32), factor, order=order)
    if arr.ndim == 3:
        return zoom(arr.astype(np.float32), (factor, factor, 1.0), order=order)
    raise ValueError(f"unsupported image ndim={arr.ndim}")


def resample_to_seg_scale(image: np.ndarray, native_mpp: float) -> np.ndarray:
    """Resample native full-resolution crop to locked DeepLab inference scale.

    Never upscales: coarse / preview inputs are segmented at their pixel grid.
    """
    mpp = validate_native_mpp(native_mpp)
    if mpp >= SEG_INFERENCE_UM_PER_PX * (1.0 - 1e-6):
        return np.asarray(image)
    out = resample_to_target(
        image,
        mpp,
        target_um_per_px=SEG_INFERENCE_UM_PER_PX,
        order=1,
    )
    return np.asarray(out)


def mask_seg_to_native(mask_seg: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    """Upsample a seg-scale boolean mask back to native (H, W) with nearest-neighbour."""
    h0, w0 = native_shape
    hs, ws = mask_seg.shape
    if (hs, ws) == (h0, w0):
        return mask_seg.astype(bool)
    zy = h0 / hs
    zx = w0 / ws
    up = zoom(mask_seg.astype(np.float32), (zy, zx), order=0)
    return up[:h0, :w0] > 0.5
