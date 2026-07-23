"""Tissue segmenters — abstract base, registry, and inference.

Concrete segmenters register themselves via :func:`register` (from
:mod:`coral.tissue.registry`); ``OtsuTissueSegmenter`` is the built-in.
Importing this package registers the built-in segmenters.

Example:
    >>> from coral.tissue import SEGMENTER_REGISTRY, OtsuTissueSegmenter
    >>> SEGMENTER_REGISTRY["otsu"] is OtsuTissueSegmenter
    True
"""

from __future__ import annotations

from coral.tissue.base import BaseTissueSegmenter
from coral.tissue.carta import CartaTissueSegmenter
from coral.tissue.infer import (
    infer_dapi_channel,
    infer_structural_channels,
)
from coral.tissue.otsu import OtsuTissueSegmenter
from coral.tissue.paths import (
    DEFAULT_TISSUE_METHOD,
    list_tissue_methods,
    resolve_tissue_method,
    tissue_dir,
    tissue_rel,
)
from coral.tissue.registry import SEGMENTER_REGISTRY, register

__all__ = [
    "DEFAULT_TISSUE_METHOD",
    "SEGMENTER_REGISTRY",
    "BaseTissueSegmenter",
    "CartaTissueSegmenter",
    "OtsuTissueSegmenter",
    "infer_dapi_channel",
    "infer_structural_channels",
    "list_tissue_methods",
    "register",
    "resolve_tissue_method",
    "tissue_dir",
    "tissue_rel",
]
