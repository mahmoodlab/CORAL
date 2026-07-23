"""Harmonize raw reader output into canonical ``(c, y, x)``.

Each reader in ``coral.io.readers`` returns
``(image, channels, mpp, source_meta)`` in whatever raw shape the
source provides. This module's job is to map any of those raw
shapes to CORAL's canonical ``(c, y, x)`` layout — flat channel
axis, standard image axes per OME-NGFF v0.4 — so every downstream
sprint (tissue, patch, feature extraction) sees a single shape.

Dispatch is keyed on the ``raw_axes`` string from the reader's
``source_meta["axes"]`` (e.g., ``"CYX"``, ``"TCYX"``, ``"YXC"``,
or the special ``"YX*"`` from ``read_channel_tiff_dir``). Helpers
land per axes pattern, one per supported axes string.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel

from coral.markers import MatchStatus
from coral.utils.errors import HarmonizationError

logger = logging.getLogger(__name__)

__all__ = [
    "CanonicalChannel",
    "HarmonizationError",
    "harmonize_to_canonical",
]


class CanonicalChannel(BaseModel):
    """One channel slot in canonical ``(c, y, x)`` layout.

    The slide's self-describing channel schema, with the list position
    serving as the channel index:

    - ``raw`` is the original observed channel name (or ``None`` if the
      source carried none — ingest substitutes a ``channel_{i}``
      placeholder marker).
    - ``marker`` is the resolved canonical/novel name, **lower-case**,
      filled by the ingest marker-resolution step.
    - ``match`` records how ``raw`` resolved to ``marker``
      (``RESOLVED`` / ``REVIEW`` / ``NOVEL``).
    - ``keep`` is the curation flag: ``False`` excludes the channel from
      tissue, cell, and feature extraction. The analysis panel is decided
      at ingest (from ``--subset`` + the QC auto-drop) and frozen here; it
      is not re-derived from the marker map afterward. Defaults ``True``.
    """

    marker: str | None = None
    raw: str | None
    match: MatchStatus | None = None
    keep: bool = True


_DispatcherFn = Callable[
    [np.ndarray, list[dict[str, Any]]],
    tuple[np.ndarray, list[CanonicalChannel]],
]


def _require_ndim(
    image: np.ndarray, expected_ndim: int, axes_label: str
) -> None:
    """Raise ``HarmonizationError`` if ``image`` isn't N-dimensional."""
    if image.ndim != expected_ndim:
        raise HarmonizationError(
            f"{axes_label} harmonize expected {expected_ndim}-D image, "
            f"got ndim={image.ndim} with shape {image.shape}."
        )


def _build_canonical_channels(
    channels: list[dict[str, Any]],
    *,
    expected_len: int,
) -> list[CanonicalChannel]:
    """Convert raw channel dicts to ``CanonicalChannel`` models.

    Shared helper used by every dispatch helper. Validates the incoming
    channel count and copies the raw observed name; ``marker`` + ``match``
    are filled later by the ingest marker-resolution step.
    """
    if len(channels) != expected_len:
        raise HarmonizationError(
            f"channel count mismatch: image has {expected_len} "
            f"channel slots but {len(channels)} channel dicts were "
            f"provided."
        )
    return [
        CanonicalChannel(raw=ch.get("marker_raw", ch.get("name")))
        for ch in channels
    ]


def _normalize_cyx(
    image: np.ndarray, channels: list[dict[str, Any]]
) -> tuple[np.ndarray, list[CanonicalChannel]]:
    """Pass-through for inputs already in canonical ``(c, y, x)``."""
    _require_ndim(image, 3, "CYX")
    canonical_channels = _build_canonical_channels(
        channels, expected_len=int(image.shape[0])
    )
    return image, canonical_channels


def _assert_yx_stack(
    image: np.ndarray, channels: list[dict[str, Any]]
) -> tuple[np.ndarray, list[CanonicalChannel]]:
    """Pass-through for the per-channel-file dir stack.

    ``read_channel_tiff_dir`` stacks per-channel files into a
    ``(C, Y, X)`` array before returning. This helper just
    validates the shape + channel count.
    """
    _require_ndim(image, 3, "YX*")
    canonical_channels = _build_canonical_channels(
        channels, expected_len=int(image.shape[0])
    )
    return image, canonical_channels


def _transpose_yxc_to_cyx(
    image: np.ndarray, channels: list[dict[str, Any]]
) -> tuple[np.ndarray, list[CanonicalChannel]]:
    """Transpose 3-D ``(Y, X, C)`` to canonical ``(C, Y, X)``."""
    _require_ndim(image, 3, "YXC")
    transposed = np.ascontiguousarray(image.transpose(2, 0, 1))
    canonical_channels = _build_canonical_channels(
        channels, expected_len=int(transposed.shape[0])
    )
    return transposed, canonical_channels


def _flatten_tcyx_to_cyx(
    image: np.ndarray, channels: list[dict[str, Any]]
) -> tuple[np.ndarray, list[CanonicalChannel]]:
    """Flatten 4-D channel-second ``(T, C, Y, X)`` to ``(T*C, Y, X)``.

    Cycle-major ordering: ``output[t*C + c, :, :] == input[t, c, :, :]``.
    This matches numpy's default C-order reshape and is consistent
    with the cycle-assignment in 01a's reader (``cycle = i // C``).
    Handles both ``TCYX`` and ``ZCYX`` — both have channel-second
    layout and flatten the same way.
    """
    _require_ndim(image, 4, "TCYX")
    n_cycles, n_per_cycle, height, width = image.shape
    flat = image.reshape(n_cycles * n_per_cycle, height, width)
    canonical_channels = _build_canonical_channels(
        channels, expected_len=int(flat.shape[0])
    )
    return flat, canonical_channels


def _flatten_tyxc_to_cyx(
    image: np.ndarray, channels: list[dict[str, Any]]
) -> tuple[np.ndarray, list[CanonicalChannel]]:
    """Flatten 4-D channel-last ``(T, Y, X, C)`` to ``(T*C, Y, X)``.

    Transpose to ``(T, C, Y, X)`` first, then reshape with cycle-major
    ordering. Handles ``TYXC`` / ``ZYXC`` (theoretical) and ``QYXS``
    (what tifffile reports for 4-D RGB-style files) — same flatten
    logic regardless of what the leading dim is called.
    """
    _require_ndim(image, 4, "TYXC")
    n_cycles, height, width, n_per_cycle = image.shape
    transposed = np.ascontiguousarray(image.transpose(0, 3, 1, 2))
    flat = transposed.reshape(n_cycles * n_per_cycle, height, width)
    canonical_channels = _build_canonical_channels(
        channels, expected_len=int(flat.shape[0])
    )
    return flat, canonical_channels


def _assume_qyx_is_cyx(
    image: np.ndarray, channels: list[dict[str, Any]]
) -> tuple[np.ndarray, list[CanonicalChannel]]:
    """Treat ``Q`` (tifffile-unknown leading dim) as channels.

    Pass-through with shape validation + WARNING log. Channel names
    are preserved as the reader provided them (likely ``None``); the
    ingest step in 01d substitutes ``f"channel_{i}"`` placeholders.
    """
    _require_ndim(image, 3, "QYX")
    logger.warning(
        "QYX axes detected — assuming Q axis is channels. "
        "Source had no explicit channel-axis metadata; verify the "
        "leading dim (size=%d) really is channels and not Z / T / "
        "tile-index before trusting downstream outputs.",
        image.shape[0],
    )
    canonical_channels = _build_canonical_channels(
        channels, expected_len=int(image.shape[0])
    )
    return image, canonical_channels


_DISPATCHER: dict[str, _DispatcherFn] = {
    "CYX": _normalize_cyx,
    "TCYX": _flatten_tcyx_to_cyx,
    "ZCYX": _flatten_tcyx_to_cyx,
    "TYXC": _flatten_tyxc_to_cyx,
    "ZYXC": _flatten_tyxc_to_cyx,
    "QYXS": _flatten_tyxc_to_cyx,  # tifffile reports QYXS for 4-D RGB
    "QYX": _assume_qyx_is_cyx,
    "YXC": _transpose_yxc_to_cyx,
    "YXS": _transpose_yxc_to_cyx,  # tifffile reports YXS for photometric=rgb
    "YX*": _assert_yx_stack,
}


def harmonize_to_canonical(
    raw_image: np.ndarray,
    raw_channels: list[dict[str, Any]],
    raw_axes: str,
) -> tuple[np.ndarray, list[CanonicalChannel]]:
    """Reshape raw reader output into canonical ``(c, y, x)``.

    Dispatches on ``raw_axes`` (the tifffile axes string from
    ``source_meta["axes"]``, or the literal ``"YX*"`` for the
    per-channel-file stack from ``read_channel_tiff_dir``).

    Args:
        raw_image: As returned by a reader.
        raw_channels: As returned by a reader.
        raw_axes: tifffile axes string, or ``"YX*"`` for dir-stack.

    Returns:
        Tuple ``(canonical_image, canonical_channels)``:

        - ``canonical_image``: ``(c, y, x)`` ndarray, dtype
          preserved from input.
        - ``canonical_channels``: list of ``CanonicalChannel``
          with length matching ``canonical_image.shape[0]``.

    Raises:
        HarmonizationError: If ``raw_axes`` is not in the
            dispatcher table. The error message names the input
            pattern and lists the supported patterns.

    Example:
        >>> import numpy as np
        >>> arr = np.zeros((3, 8, 8), dtype=np.uint16)
        >>> ch = [{"name": None, "source_index": i} for i in range(3)]
        >>> try:
        ...     harmonize_to_canonical(arr, ch, "made_up_axes")
        ... except HarmonizationError as exc:
        ...     "Unsupported axes pattern" in str(exc)
        True
    """
    if raw_axes not in _DISPATCHER:
        raise HarmonizationError(
            f"Unsupported axes pattern: {raw_axes!r}. "
            f"Supported: {sorted(_DISPATCHER) or '(none yet)'}. "
            f"If you have a real source that produces this pattern, "
            f"open an issue with a sample file."
        )
    return _DISPATCHER[raw_axes](raw_image, raw_channels)
