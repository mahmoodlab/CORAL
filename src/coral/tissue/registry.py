"""Tissue-segmenter registry — ``name -> class`` + ``@register``.

Lives in its own module (not ``__init__``) so concrete segmenters can
``from coral.tissue.registry import register`` without importing the
package ``__init__`` mid-initialization — ``registry`` depends only on
:mod:`coral.tissue.base` and :mod:`coral.utils.registry`, so there is no
import cycle.
"""

from __future__ import annotations

from coral.tissue.base import BaseTissueSegmenter
from coral.utils.registry import make_register_decorator

SEGMENTER_REGISTRY: dict[str, type[BaseTissueSegmenter]] = {}

register = make_register_decorator(
    SEGMENTER_REGISTRY,
    kind="segmenter",
    base_class=BaseTissueSegmenter,
)

__all__ = ["SEGMENTER_REGISTRY", "register"]
