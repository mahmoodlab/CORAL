"""``MeanMarkerExtractor`` — per-marker mean intensity (no ML deps).

The authoritative baseline (Andrew, original author; reproduces
``kronos/image/spatial_image.py::run_mean_intensity_marker_extraction``)
is, per marker::

    mean = sum_nonzero(patch / scaling_factor) / (eps + count_nonzero)
         = raw_mean_nonzero / scaling_factor      # eps = 1e-6

The patch is first normalised by the image's dtype ``scaling_factor``
(``uint8 -> 255``, ``uint16 -> 65535``, ``float -> 400`` — the KRONOS
``ImagePatcher._get_scaling_factor`` logic), then averaged over its
**non-zero** pixels (NOT the box area). One formula serves grid + cell:
the grid-vs-cell difference is purely the patch handed in (raw box vs
neighbour/background-masked), per ``CoralSlide.encode_features``. The
``scaling_factor`` is taken from the patch dtype, so a stored uint16
slide yields ``/65535`` — tying out to the gold-standard features
exactly (verified on CRC to float precision).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from coral.features import register
from coral.features.base import CoralEncoder

_EPS = 1e-6
# Domain-specific factor for float images (KRONOS _get_scaling_factor).
_FLOAT_SCALE = 400.0


def _scaling_factor(dtype: np.dtype) -> float:
    """Image-normalisation divisor by dtype (KRONOS ``_get_scaling_factor``).

    ``uint8 -> 255``; any wider unsigned int (uint16, big-endian ``>u2``)
    ``-> 65535``; float ``-> 400``.

    Example:
        >>> import numpy as np
        >>> (
        ...     _scaling_factor(np.dtype("uint16")),
        ...     _scaling_factor(np.dtype("uint8")),
        ... )
        (65535.0, 255.0)
    """
    if dtype == np.uint8:
        return 255.0
    if np.issubdtype(dtype, np.unsignedinteger):  # uint16 / >u2
        return 65535.0
    if np.issubdtype(dtype, np.floating):
        return _FLOAT_SCALE
    msg = f"no scaling_factor for dtype {dtype!r}"
    raise ValueError(msg)


@register("mean_marker")
class MeanMarkerExtractor(CoralEncoder):
    """Per-marker mean over non-zero pixels of the dtype-normalised patch.

    Marker-agnostic, no model. Output is ``(n_patches, n_markers)`` — each
    patch's per-marker ``raw_mean_nonzero / scaling_factor``, where
    ``scaling_factor`` is read from the patch dtype (uint16 -> 65535).

    ``scale = False``: the patcher hands it the **raw** box and mean_marker
    does the divide itself in ``float64`` (its reduction's numerical
    precision, which the bit-exact tie-out depends on). :meth:`transform`
    is that float64 divide; :meth:`forward` is the nonzero mean.

    Example:
        >>> import numpy as np
        >>> ext = MeanMarkerExtractor()
        >>> # 1 marker, 2x2 uint16 box, two zero pixels: sum 6 / count 2.
        >>> patch = np.array([[[2, 0], [4, 0]]], dtype="uint16")
        >>> out = np.asarray(ext.encode(patch[None], ["m"]))
        >>> float(round(out[0, 0] * 65535, 3))  # nonzero mean 3.0, /65535
        3.0
    """

    name: ClassVar[str] = "mean_marker"
    # One value per marker -> stored as the array
    # features/<slug>/mean_marker/<variant> with dims (patch, marker).
    # Every encoder declares one logical output named "features"; the
    # trailing dim names the semantics ("marker" here, "feature" for CLS
    # encoders). The encoder + marker variant in the path disambiguate
    # which run produced it.
    output_schema: ClassVar[dict[str, tuple[str, ...]]] = {
        "features": ("marker",),
    }
    # The patcher gives raw pixels; mean_marker scales in its own float64.
    scale: ClassVar[bool] = False

    def required_markers(self) -> list[str] | None:
        """Marker-agnostic — uses whatever markers the patches carry."""
        return None

    def transform(
        self,
        patches: Any,  # noqa: ANN401 — raw patch batch
        markers: list[str],
        *,
        nuclear_marker: str | None = None,
    ) -> np.ndarray:
        """Normalise the raw patch by the image dtype ``scaling_factor``.

        Matches the KRONOS ``ImagePatcher`` ``patch /= scaling_factor``
        step in ``float64`` — the per-encoder image normalisation for
        mean-marker.

        Example:
            >>> import numpy as np
            >>> p = np.full((1, 1, 2, 2), 65535, dtype="uint16")
            >>> float(MeanMarkerExtractor().transform(p, ["m"]).max())
            1.0
        """
        arr = np.asarray(patches)
        return arr.astype(np.float64) / _scaling_factor(arr.dtype)

    def forward(
        self,
        x: Any,  # noqa: ANN401 — normalised patch batch
        markers: list[str],
        marker_emb: Any = None,  # noqa: ANN401 — unused (marker-agnostic)
    ) -> np.ndarray:
        """Per-marker ``sum / (eps + count_nonzero)`` over each patch.

        Args:
            x: Normalised patches ``(n_patches, n_markers, h, w)``.
            markers: Marker names (unused; mean is per-channel).
            marker_emb: Unused (marker-agnostic).

        Returns:
            ``(n_patches, n_markers)`` per-marker means (float64).

        Example:
            >>> import numpy as np
            >>> ext = MeanMarkerExtractor()
            >>> ext.forward(np.ones((2, 3, 4, 4)), ["a", "b", "c"]).shape
            (2, 3)
        """
        arr = np.asarray(x, dtype=np.float64)
        sums = arr.sum(axis=(-1, -2))
        counts = np.count_nonzero(arr, axis=(-1, -2))
        return sums / (_EPS + counts)
