"""Per-core mask cleanup at DeepLab inference scale (µm² / µm → px)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mask_cleanup import CleanupStats, cleanup_tma_mask
from .seg_scale import SEG_INFERENCE_UM_PER_PX


@dataclass(frozen=True)
class CoreMaskCleanup:
    """Morphological cleanup for per-core masks at ``SEG_INFERENCE_UM_PER_PX``."""

    min_component_area_um2: float = 17_000.0
    fill_holes_max_area_um2: float = 2_000.0
    closing_radius_um: float = 2.0

    @property
    def mask_mpp(self) -> float:
        return SEG_INFERENCE_UM_PER_PX

    def to_px(self) -> tuple[int, int, int]:
        mpp = self.mask_mpp
        mpp2 = mpp * mpp
        min_area_px = max(0, int(round(self.min_component_area_um2 / mpp2)))
        fill_holes_px = max(0, int(round(self.fill_holes_max_area_um2 / mpp2)))
        closing_px = max(0, int(round(self.closing_radius_um / mpp)))
        return min_area_px, fill_holes_px, closing_px


def apply_core_mask_cleanup(
    mask_seg: np.ndarray,
    params: CoreMaskCleanup | None,
) -> tuple[np.ndarray, CleanupStats | None]:
    """Apply area-bounded cleanup on a seg-scale binary mask."""
    if params is None:
        return mask_seg.astype(bool), None
    min_px, fill_px, close_px = params.to_px()
    cleaned, _stages, stats = cleanup_tma_mask(
        mask_seg,
        min_component_area_px=min_px,
        fill_holes_max_area_px=fill_px,
        closing_radius_px=close_px,
    )
    return cleaned, stats


DEFAULT_CORE_MASK_CLEANUP = CoreMaskCleanup()
