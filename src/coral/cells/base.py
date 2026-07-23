"""``BaseCellSegmenter`` — abstract base for cell-segmentation models.

Concrete segmenters: ``CellposeSegmenter``; Mesmer is a future
sibling. Mirrors :class:`coral.tissue.base.BaseTissueSegmenter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class BaseCellSegmenter(ABC):
    """Abstract base for CORAL cell-instance segmenters.

    Subclasses declare:

    - ``name`` — registry key (e.g. ``"cellpose"``).

    and implement :meth:`segment`, which takes the **nuclear** + the
    **composite membrane** channels and returns a 2-D ``int32``
    instance-label mask (``0`` = background, each cell a unique id).

    Example:
        After tissue detection, segment cells on a slide::

            from coral import CoralSlide
            from coral.cells import CellposeSegmenter

            slide = CoralSlide.open("slide.zarr")
            mask = slide.segment_cells(CellposeSegmenter())
    """

    name: ClassVar[str] = "base"

    @abstractmethod
    def segment(
        self,
        nuclear: np.ndarray,
        membrane: np.ndarray,
        *,
        mpp: float,
    ) -> np.ndarray:
        """Return an ``int32`` ``(y, x)`` instance mask for the inputs.

        Args:
            nuclear: 2-D nuclear channel.
            membrane: 2-D composite membrane channel (same shape as
                ``nuclear``).
            mpp: Microns-per-pixel of the input (the cell-size prior).

        Returns:
            ``int32`` instance-label mask, ``0`` = background.

        Example:
            Concrete segmenters (e.g. :class:`CellposeSegmenter`) implement
            this; call via :meth:`coral.slide.CoralSlide.segment_cells`.
        """
