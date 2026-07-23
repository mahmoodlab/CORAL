"""
DAPI single-channel → 3-channel RGB for foundation / finetune models.

Strategies (cfg.dapi_rgb_mode)
------------------------------
  replicate    : stack normalized DAPI × 3
  gamma_invert : stack (1 - norm) × 3 — matches H&E nuclear polarity

Vendored from CARTA ``segmenter/dapi_rgb.py``. Edit vs upstream: ``normalize_percentile``
is inlined here instead of imported from ``segmenter/data.py``, so the tissue embed
does not pull in tifffile / imagecodecs / PIL.
"""
import numpy as np


def normalize_percentile(image: np.ndarray,
                         lo_pct: float = 1.0,
                         hi_pct: float = 99.0) -> np.ndarray:
    arr = image.astype(np.float32)
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def dapi_to_rgb(image: np.ndarray, mode: str = "gamma_invert",
                lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """
    Convert a 2-D DAPI image to uint8 HWC RGB.

    Returns
    -------
    rgb : (H, W, 3) uint8
    """
    if image.ndim != 2:
        raise ValueError(f"expected 2-D DAPI, got shape {image.shape}")
    norm = normalize_percentile(image, lo_pct, hi_pct)
    if mode == "replicate":
        ch = (norm * 255.0).clip(0, 255).astype(np.uint8)
        return np.stack([ch, ch, ch], axis=-1)
    if mode == "gamma_invert":
        inv = (1.0 - norm).clip(0.0, 1.0)
        ch = (inv * 255.0).astype(np.uint8)
        return np.stack([ch, ch, ch], axis=-1)
    raise ValueError(f"unknown dapi_rgb_mode {mode!r}; use replicate | gamma_invert")
