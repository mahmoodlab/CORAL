"""Supported pixel dtypes — CORAL's one image-normalisation policy.

Ingest stores source pixels verbatim (it never rescales), so the divisor
that maps a patch into ``[0, 1]`` is set by the **image** dtype, never by
the encoder. This module is that policy's single home: ingest calls
:func:`validate_image_dtype` to reject an image it cannot scale before
anything is written, and the feature path divides by
:func:`scaling_factor`.

Supported: ``uint8`` (``/255``), ``uint16`` (``/65535``), and any float
already normalised to ``[0, 1]`` (``/1`` — passed through untouched).
Everything else (signed ints, ``uint32``/``uint64``) is rejected loudly
at ingest rather than silently mis-scaled several stages downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["scaling_factor", "validate_image_dtype"]

# Slack on the float [0, 1] bound. Normalising by a channel maximum
# routinely lands a hair either side of the bounds in float32 (eps
# ~1.2e-7), and background subtraction can leave a faintly negative
# floor; 1e-6 absorbs that rounding without admitting data that is
# genuinely on another scale.
_FLOAT_TOL = 1e-6

# Offending channels named in the out-of-range error before it truncates.
_MAX_REPORTED = 5

_SUPPORTED = "uint8, uint16, or float normalised to [0, 1]"

_FLOAT_RANGE_MSG = (
    "The values for float input should be between [0,1]. Please make "
    "sure the data is properly normalized."
)


def _divisor(dtype: np.dtype) -> float | None:
    """The dtype's ``[0, 1]`` divisor, or ``None`` when unsupported.

    Width is tested via ``itemsize`` rather than ``dtype == np.uint16``
    because that equality is **False** for a big-endian ``>u2`` — which
    is uint16 data and scales by 65535 like any other.
    """
    if np.issubdtype(dtype, np.unsignedinteger):
        if dtype.itemsize == 1:
            return 255.0
        if dtype.itemsize == 2:
            return 65535.0
        return None  # uint32 / uint64 — not a fluorescence range CORAL knows
    if np.issubdtype(dtype, np.floating):
        return 1.0
    return None


def _unsupported_msg(dtype: np.dtype) -> str:
    """The rejection message for a dtype CORAL cannot scale."""
    return (
        f"{dtype} is not currently supported by CORAL ingest "
        f"(supported: {_SUPPORTED})"
    )


def scaling_factor(dtype: Any) -> float:  # noqa: ANN401 — any dtype-like
    """Divisor mapping pixels of ``dtype`` into ``[0, 1]``.

    ``uint8 -> 255``; ``uint16 -> 65535`` (either byte order); float
    ``-> 1``, i.e. passed through, since ingest has already enforced
    that float pixels are normalised. Any other dtype raises.

    Args:
        dtype: The image or patch dtype (anything ``np.dtype`` accepts).

    Returns:
        The divisor for that dtype.

    Raises:
        ValueError: If CORAL does not support the dtype.

    Example:
        >>> import numpy as np
        >>> (
        ...     scaling_factor(np.dtype("uint16")),
        ...     scaling_factor(np.dtype("uint8")),
        ... )
        (65535.0, 255.0)
        >>> scaling_factor(np.dtype(">u2"))  # big-endian uint16
        65535.0
        >>> scaling_factor(np.dtype("float32"))
        1.0
        >>> try:
        ...     scaling_factor(np.dtype("uint32"))
        ... except ValueError as exc:
        ...     print(str(exc).split(" (")[0])
        uint32 is not currently supported by CORAL ingest
    """
    resolved = np.dtype(dtype)
    divisor = _divisor(resolved)
    if divisor is None:
        raise ValueError(_unsupported_msg(resolved))
    return divisor


def _channel_label(index: int, markers: list[str] | None) -> str:
    """``"channel 3 (CD8)"`` when markers are known, else ``"channel 3"``."""
    if markers is not None and index < len(markers):
        return f"channel {index} ({markers[index]})"
    return f"channel {index}"


def validate_image_dtype(
    image: np.ndarray,
    name: str,
    markers: list[str] | None = None,
) -> None:
    """Reject an image CORAL cannot ingest — before anything is written.

    Two gates. The dtype must be one CORAL scales (see
    :func:`scaling_factor`), and a float image must already be normalised
    to ``[0, 1]``, checked against the pixels because a float dtype
    implies no range of its own. ``NaN`` and ``inf`` fail that check too:
    every comparison against them is false, so they cannot pass a bound.

    Args:
        image: Canonical ``(c, y, x)`` pixels.
        name: Image entry name, for the error message.
        markers: Resolved marker names in channel order, used to name the
            offending channels; ``None`` reports bare indices.

    Raises:
        ValueError: If the dtype is unsupported, or a float image carries
            values outside ``[0, 1]``.

    Example:
        >>> import numpy as np
        >>> validate_image_dtype(np.zeros((2, 4, 4), "uint16"), "case1.tif")
        >>> img = np.full((1, 2, 2), 3.5, dtype="float32")
        >>> try:
        ...     validate_image_dtype(img, "case1.tif", ["CD8"])
        ... except ValueError as exc:
        ...     print("[0,1]" in str(exc), "channel 0 (CD8)" in str(exc))
        True True
    """
    pixels = np.asarray(image)
    dtype = pixels.dtype
    if _divisor(dtype) is None:
        raise ValueError(f"{name}: {_unsupported_msg(dtype)}")
    if not np.issubdtype(dtype, np.floating):
        return

    # Reduce over the spatial axes without reshaping — the canonical
    # image is a transposed view of the read, so a reshape would copy
    # the whole (potentially many-GB) array just to take a min.
    if pixels.ndim == 3:
        low = pixels.min(axis=(1, 2))
        high = pixels.max(axis=(1, 2))
    else:
        low = np.atleast_1d(pixels.min())
        high = np.atleast_1d(pixels.max())

    in_range = (low >= -_FLOAT_TOL) & (high <= 1.0 + _FLOAT_TOL)
    bad = np.flatnonzero(~in_range)
    if not bad.size:
        return
    detail = ", ".join(
        f"{_channel_label(int(i), markers)} [{low[i]:.4g}, {high[i]:.4g}]"
        for i in bad[:_MAX_REPORTED]
    )
    more = (
        ""
        if bad.size <= _MAX_REPORTED
        else f", ... (+{bad.size - _MAX_REPORTED} more)"
    )
    raise ValueError(
        f"{name}: {_FLOAT_RANGE_MSG} Out of range: {detail}{more}"
    )
