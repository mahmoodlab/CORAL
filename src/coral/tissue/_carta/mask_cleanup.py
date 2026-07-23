"""Config-driven cleanup for TMA whole-slide masks.

Vendored from CARTA ``segmenter/mask_cleanup.py``. Edit vs upstream: ``close_mask``
keeps only the scikit-image ``morphology.closing`` path; the optional
``cv2.morphologyEx`` fast path is removed so the CORAL tissue embed stays OpenCV-free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import label as sk_label
from skimage.morphology import closing, disk


@dataclass
class CleanupStats:
    raw_components: int
    min_area_components: int
    hole_fill_components: int
    closing_components: int


def component_count(mask: np.ndarray) -> int:
    """Count connected foreground components in a binary mask."""
    return int(sk_label(mask.astype(bool)).max())


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return mask.astype(bool)
    labeled = sk_label(mask.astype(bool))
    counts = np.bincount(labeled.ravel())
    keep = counts >= int(min_area_px)
    keep[0] = False
    return keep[labeled]


def fill_small_holes(mask: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    if max_hole_area_px <= 0:
        return mask.astype(bool)
    mask = mask.astype(bool)
    holes = ~mask
    labeled = sk_label(holes)
    counts = np.bincount(labeled.ravel())
    border_labels = np.unique(
        np.concatenate([labeled[0], labeled[-1], labeled[:, 0], labeled[:, -1]])
    )
    fill = counts <= int(max_hole_area_px)
    fill[0] = False
    fill[border_labels] = False
    out = mask.copy()
    out[fill[labeled]] = True
    return out


def close_mask(mask: np.ndarray, closing_radius_px: int) -> np.ndarray:
    if closing_radius_px <= 0:
        return mask.astype(bool)
    mask = mask.astype(bool)
    return closing(mask, footprint=disk(int(closing_radius_px)))


def cleanup_tma_mask(
    mask: np.ndarray,
    min_component_area_px: int,
    fill_holes_max_area_px: int,
    closing_radius_px: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], CleanupStats]:
    """
    Clean a TMA mask with the requested ordering:
      1. drop small components
      2. fill only bounded holes
      3. apply a light closing
    """
    raw = mask.astype(bool)
    raw_labeled = sk_label(raw)
    raw_components = int(raw_labeled.max())

    min_area = remove_small_components(raw, min_component_area_px)
    min_area_components = component_count(min_area)

    hole_fill = fill_small_holes(min_area, fill_holes_max_area_px)
    hole_fill_components = component_count(hole_fill)

    closing_mask = close_mask(hole_fill, closing_radius_px)
    closing_components = component_count(closing_mask)

    stages = {
        "raw": raw,
        "min_area": min_area,
        "hole_fill": hole_fill,
        "closing": closing_mask,
    }
    stats = CleanupStats(
        raw_components=raw_components,
        min_area_components=min_area_components,
        hole_fill_components=hole_fill_components,
        closing_components=closing_components,
    )
    return closing_mask, stages, stats
