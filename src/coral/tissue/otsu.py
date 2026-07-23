"""``OtsuTissueSegmenter`` — hybrid Otsu detection + alpha-shape region.

Ports the validated reference ``detect_tissue`` thresholding — per-channel
percentile normalization, then a **hybrid** threshold: the nuclear channel
Otsu-thresholded at a lower factor (punctate nuclei are under-called by a
global Otsu) **unioned** with the Otsu threshold of the structural
max-projection. Those thresholds yield many small tissue *islands*; the
segmenter then encloses them in a single **tissue region** — an alpha
shape (see :mod:`coral.tissue.hull`) whose one physical knob,
``max_bridge_distance``, sets how encompassing the boundary is.

The segmenter is resolution-agnostic: it takes a ``(c, y, x)`` array whose
**channel 0 is the nuclear stain and channels 1.. are structural** (the
stack the slide driver assembles after inference), plus the working
``mpp`` so the micron parameters convert to pixels. Deterministic — no
randomness. No ML dependencies; no GPU.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from coral.tissue.base import BaseTissueSegmenter
from coral.tissue.hull import alpha_shape_region
from coral.tissue.registry import register

# Stride for the percentile subsample in normalization — percentiles
# are robust to uniform subsampling, so estimating them on 1/stride^2 of
# a large channel is much cheaper than sorting the whole thing.
_PCTL_STRIDE = 4


def _remove_small_objects(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Drop connected components smaller than ``min_px`` pixels.

    scipy-based (stable across the supported scikit-image range, which
    deprecates ``remove_small_objects``' ``min_size``).
    """
    labels, n = cast("tuple[np.ndarray, int]", ndi.label(mask))
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background label
    return np.asarray(counts[labels] >= min_px)


@register("otsu")
class OtsuTissueSegmenter(BaseTissueSegmenter):
    """Hybrid Otsu tissue segmenter (nuclear ∪ structural) + region hull.

    Args:
        nuclear_threshold_factor: Multiplier on the nuclear channel's Otsu
            threshold (``< 1`` captures more punctate nuclei).
        min_object_area_um2: Noise specks smaller than this (microns²) are
            dropped before the tissue region is built.
        max_bridge_distance: The user-facing knob — the widest
            background gap (microns) the tissue boundary bridges before it
            concaves inward. Larger → more encompassing (→ the convex
            hull); smaller → the boundary hugs each tissue piece.

    Example:
        >>> import numpy as np
        >>> seg = OtsuTissueSegmenter()
        >>> seg.name
        'otsu'
        >>> blank = np.zeros((4, 64, 64), dtype="uint8")
        >>> bool(seg.segment(blank).any())
        False
    """

    name = "otsu"
    uses_structural = True  # thresholds on the structural max-projection

    def __init__(
        self,
        *,
        nuclear_threshold_factor: float = 0.7,
        min_object_area_um2: float = 68.0,
        max_bridge_distance: float = 200.0,
    ) -> None:
        """Store the (documented, tunable) algorithm parameters.

        ``nuclear_threshold_factor`` (``< 1`` captures more punctate
        nuclei) and ``min_object_area_um2`` (drop noise specks below this
        physical area) reproduce the reference thresholding at any
        resolution. ``max_bridge_distance`` is the one user knob: the
        widest background gap the tissue boundary bridges before concaving
        inward — larger encompasses more, smaller hugs each piece.
        """
        self.nuclear_threshold_factor = nuclear_threshold_factor
        self.min_object_area_um2 = min_object_area_um2
        self.max_bridge_distance = max_bridge_distance

    @property
    def params(self) -> dict[str, Any]:
        """The three Otsu tuning knobs, recorded in ``tissue/config.json``.

        Example:
            >>> OtsuTissueSegmenter().params["max_bridge_distance"]
            200.0
        """
        return {
            "nuclear_threshold_factor": self.nuclear_threshold_factor,
            "min_object_area_um2": self.min_object_area_um2,
            "max_bridge_distance": self.max_bridge_distance,
        }

    def required_channels(self) -> list[str] | None:
        """``None`` — channels are inferred by the slide driver.

        Returns:
            Always ``None`` (nuclear + structural inferred at call time).

        Example:
            >>> from coral.tissue import OtsuTissueSegmenter
            >>> OtsuTissueSegmenter().required_channels() is None
            True
        """
        return None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Percentile-normalize each channel to ``[0, 1]`` float32.

        Args:
            image: ``(c, y, x)`` array; channel 0 nuclear, 1.. structural.

        Returns:
            Normalized ``(c, y, x)`` float32 array.

        Raises:
            ValueError: If ``image`` is not 3-D.

        Example:
            >>> import numpy as np
            >>> from coral.tissue import OtsuTissueSegmenter
            >>> img = np.zeros((2, 8, 8), dtype="uint8")
            >>> out = OtsuTissueSegmenter().preprocess(img)
            >>> out.dtype == np.float32 and out.shape == (2, 8, 8)
            True
        """
        image = np.asarray(image)
        if image.ndim != 3:
            msg = f"expected (c, y, x) image, got shape {image.shape}"
            raise ValueError(msg)
        return np.stack(
            [self._normalize(image[c]) for c in range(image.shape[0])]
        )

    def forward(self, image: np.ndarray, mpp: float = 1.0) -> np.ndarray:
        """Hybrid threshold → island cleanup → tissue-region alpha shape.

        Args:
            image: Normalized ``(c, y, x)`` stack (channel 0 nuclear).
            mpp: Working microns-per-pixel, for the micron parameters.

        Returns:
            Boolean ``(y, x)`` tissue-region mask.

        Example:
            See :meth:`segment` for an end-to-end nuclear-block example.
        """
        nuclear = image[0]
        structural = image[1:]
        masks = []
        nuc = self._otsu_mask(nuclear, self.nuclear_threshold_factor)
        if nuc is not None:
            masks.append(nuc)
        if structural.shape[0] > 0:
            struct = self._otsu_mask(structural.max(axis=0), 1.0)
            if struct is not None:
                masks.append(struct)
        if not masks:
            return np.zeros(nuclear.shape, dtype=bool)
        islands = np.logical_or.reduce(masks)
        min_object = max(1, round(self.min_object_area_um2 / mpp**2))
        islands = _remove_small_objects(islands, min_object)
        return alpha_shape_region(
            islands, mpp, max_bridge_um=self.max_bridge_distance
        )

    def segment(self, image: np.ndarray, mpp: float = 1.0) -> np.ndarray:
        """Run :meth:`preprocess` then :meth:`forward` at ``mpp``.

        Args:
            image: ``(c, y, x)`` array; channel 0 nuclear, 1.. structural.
            mpp: Working microns-per-pixel of ``image``.

        Returns:
            Boolean ``(y, x)`` tissue mask.

        Example:
            >>> import numpy as np
            >>> seg = OtsuTissueSegmenter()
            >>> img = np.zeros((1, 32, 32), dtype="uint8")
            >>> img[0, 8:24, 8:24] = 200  # one bright nuclear block
            >>> mask = seg.segment(img, mpp=1.0)
            >>> mask.dtype == bool and mask.shape == (32, 32)
            True
        """
        return self.forward(self.preprocess(image), mpp)

    @staticmethod
    def _normalize(
        channel: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
    ) -> np.ndarray:
        """Percentile-clip a single channel to ``[0, 1]`` float32.

        The clip bounds are estimated on a strided subsample (a single
        sort of ~1/16 the pixels), then the rescale runs on the full
        channel — much cheaper than sorting a whole WSI channel twice.
        """
        sample = channel[::_PCTL_STRIDE, ::_PCTL_STRIDE]
        lo, hi = (float(v) for v in np.percentile(sample, [p_low, p_high]))
        if hi <= lo:
            return np.zeros(channel.shape, dtype=np.float32)
        out = (channel.astype(np.float32) - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _otsu_mask(channel: np.ndarray, factor: float) -> np.ndarray | None:
        """Otsu threshold ``channel`` (× ``factor``); ``None`` if flat."""
        try:
            threshold = threshold_otsu(channel) * factor
        except ValueError:
            return None  # single-valued channel — no threshold
        return channel > threshold
