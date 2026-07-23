"""Cell segmentation — Cellpose instance masks + centroid tables.

Public surface:

- :class:`BaseCellSegmenter` — the segmenter ABC.
- :class:`CellposeSegmenter` — Cellpose 4.x (cpsam); imports cellpose
  lazily, so this package imports without the ``cells`` extra.
- :data:`CELL_SEGMENTER_REGISTRY` / :func:`register` — the registry.
"""

from coral.cells.base import BaseCellSegmenter
from coral.cells.cellpose import CellposeSegmenter
from coral.cells.registry import CELL_SEGMENTER_REGISTRY, register

__all__ = [
    "CELL_SEGMENTER_REGISTRY",
    "BaseCellSegmenter",
    "CellposeSegmenter",
    "register",
]
