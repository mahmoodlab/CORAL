"""Cell-segmenter registry — ``name -> class`` + ``@register``.

Mirrors :mod:`coral.tissue.registry`. Lives in its own module so concrete
segmenters can ``from coral.cells.registry import register`` without
importing the package ``__init__`` mid-initialization (no import cycle).
"""

from __future__ import annotations

from coral.cells.base import BaseCellSegmenter
from coral.utils.registry import make_register_decorator

CELL_SEGMENTER_REGISTRY: dict[str, type[BaseCellSegmenter]] = {}

register = make_register_decorator(
    CELL_SEGMENTER_REGISTRY,
    kind="cell segmenter",
    base_class=BaseCellSegmenter,
)

__all__ = ["CELL_SEGMENTER_REGISTRY", "register"]
