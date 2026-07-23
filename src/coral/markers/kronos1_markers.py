"""KRONOS1 marker resolution — vocab ids + per-marker z-score stats.

Reproduces the kronos ``KRONOSImage`` marker-record build
(``kronos/image/spatial_image.py``) so CORAL's KRONOS1 extractor feeds the
model the **same** integer ``marker_ids`` + per-marker z-score stats as
the published pipeline. Pure + model-free — the CI-testable correctness
core of the extractor.

The scoped use of kronos's ``MARKER_RENAMING_DICT`` (``hoechst*`` ->
``dapi`` etc.) lives here for **vocab/id resolution only** — CORAL never
renames a slide's stored channels (the scoped exception to
``feedback_no_auto_channel_rename``, Anurag 2026-06-24). It is *not* the
general :func:`coral.markers.normalize.normalize_marker_name`, which
deliberately drops the alias dict + the split/isotope heuristics.
"""

from __future__ import annotations

import difflib
import re
import warnings
from dataclasses import dataclass

from coral.markers._kronos_marker_meta import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    KRONOS1_MARKER_META,
    MARKER_RENAMING_DICT,
)
from coral.markers.normalize import clean_marker_name

__all__ = ["MarkerRecord", "normalize_marker_name", "resolve_markers"]

# Out-of-vocab markers get the next free odd id in this range, in panel
# order (kronos ``spatial_image.py``: ``odd_id_generator(103, 255)``). The
# vocab ids are all even, so the odd range can never collide with them.
_ODD_ID_START = 103
_ODD_ID_STOP = 255

# The 177 vocabulary names (cleaned) = the canonical match list.
_CANONICAL: tuple[str, ...] = tuple(KRONOS1_MARKER_META)
# Vocab ids already taken (the odd-id fallback skips these).
_USED_IDS: frozenset[int] = frozenset(
    marker_id for marker_id, _, _ in KRONOS1_MARKER_META.values()
)
# kronos step 4: strip a trailing ``_<number><letters>`` (isotope tag).
_ISOTOPE_RE = re.compile(r"_\d+[a-z]*")


@dataclass(frozen=True)
class MarkerRecord:
    """Resolved KRONOS1 inputs for one panel marker.

    Attributes:
        raw: The marker name as it appears in the panel.
        norm: The resolved vocabulary name (or the cleaned fallback).
        marker_id: The integer id fed to the model (vocab id, or an odd
            fallback id for out-of-vocab markers).
        mean: Per-marker z-score mean (default for out-of-vocab).
        std: Per-marker z-score std (default for out-of-vocab).
        in_vocab: Whether ``norm`` matched the 177-marker vocabulary.
    """

    raw: str
    norm: str
    marker_id: int
    mean: float
    std: float
    in_vocab: bool


def normalize_marker_name(name: str) -> str:
    """Resolve a raw marker name to a KRONOS1 vocabulary name.

    Lifts kronos ``utils_marker.normalize_marker_name`` verbatim against
    the 177-marker vocabulary + ``MARKER_RENAMING_DICT``, in order:

    1. **exact** — the cleaned name is a vocabulary key.
    2. **alias** — the cleaned name is a ``MARKER_RENAMING_DICT`` key.
    3. **parts** — for each ``_``-split part, a vocab then an alias hit.
    4. **isotope-strip** — drop a trailing ``_<isotope>`` and retry 1+2.
    5. **fuzzy** — closest ``difflib`` vocab match at cutoff ``0.8``
       (names >= 3 chars).
    6. **fallback** — the cleaned name, unchanged.

    Args:
        name: A raw marker / channel name.

    Returns:
        The resolved vocabulary name, or the cleaned fallback.

    Example:
        >>> normalize_marker_name("DAPI")
        'dapi'
        >>> normalize_marker_name("Hoechst1")
        'dapi'
        >>> normalize_marker_name("PanCK")
        'cytokeratin'
    """
    cleaned = clean_marker_name(name)
    if cleaned in KRONOS1_MARKER_META:
        return cleaned
    if cleaned in MARKER_RENAMING_DICT:
        return MARKER_RENAMING_DICT[cleaned]
    for part in cleaned.split("_"):
        if part in KRONOS1_MARKER_META:
            return part
        if part in MARKER_RENAMING_DICT:
            return MARKER_RENAMING_DICT[part]
    stripped = _ISOTOPE_RE.sub("", cleaned)
    if stripped in KRONOS1_MARKER_META:
        return stripped
    if stripped in MARKER_RENAMING_DICT:
        return MARKER_RENAMING_DICT[stripped]
    if len(cleaned) >= 3:
        close = difflib.get_close_matches(cleaned, _CANONICAL, n=1, cutoff=0.8)
        if close:
            return close[0]
    return cleaned


def resolve_markers(
    markers: list[str], *, nuclear_marker: str | None = None
) -> list[MarkerRecord]:
    """Resolve a panel to KRONOS1 ``(marker_id, mean, std)`` per marker.

    Reproduces kronos ``KRONOSImage`` (``spatial_image.py``): each marker
    is normalized to the vocabulary; vocab hits take their
    ``(id, mean, std)``; misses get the next free **odd** id in
    ``[103, 255]`` (panel order, deterministic) + the default z-score
    stats, and are named in a single warning. ``nuclear_marker`` (the
    slide's DAPI hint) forces that channel to ``"dapi"``; standard
    ``DAPI``/``Hoechst`` names already resolve via the alias dict.

    Args:
        markers: The panel's marker names, in channel order.
        nuclear_marker: The channel to treat as DAPI (optional hint).

    Returns:
        One :class:`MarkerRecord` per input marker, in order.

    Raises:
        ValueError: If more than 77 markers fall out of vocabulary
            (the odd-id range ``[103, 255]`` is exhausted).

    Example:
        >>> recs = resolve_markers(["DAPI", "CD8a"])
        >>> [(r.norm, r.marker_id, r.in_vocab) for r in recs]
        [('dapi', 4, True), ('cd8', 294, True)]
    """
    odd_ids = (
        i
        for i in range(_ODD_ID_START, _ODD_ID_STOP + 1, 2)
        if i not in _USED_IDS
    )
    nuclear_clean = (
        clean_marker_name(nuclear_marker) if nuclear_marker else None
    )

    records: list[MarkerRecord] = []
    out_of_vocab: list[str] = []
    for raw in markers:
        if (
            nuclear_clean is not None
            and clean_marker_name(raw) == nuclear_clean
        ):
            norm = "dapi"
        else:
            norm = normalize_marker_name(raw)
        meta = KRONOS1_MARKER_META.get(norm)
        if meta is not None:
            marker_id, mean, std = meta
            in_vocab = True
        else:
            try:
                marker_id = next(odd_ids)
            except StopIteration:
                raise ValueError(
                    "More than 77 out-of-vocab markers — ran out of "
                    "unique odd KRONOS1 marker ids in [103, 255]."
                ) from None
            mean, std, in_vocab = DEFAULT_MEAN, DEFAULT_STD, False
            out_of_vocab.append(raw)
        records.append(MarkerRecord(raw, norm, marker_id, mean, std, in_vocab))

    if out_of_vocab:
        warnings.warn(
            f"KRONOS1: {len(out_of_vocab)} of {len(markers)} markers are "
            f"not in the vocabulary; assigned fallback odd ids + default "
            f"z-score stats (mean={DEFAULT_MEAN:.4g}, std={DEFAULT_STD:.4g}"
            f"): {out_of_vocab}",
            stacklevel=2,
        )
    return records
