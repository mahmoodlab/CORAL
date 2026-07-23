"""Smart channel inference for tissue detection.

Tissue detection needs to know which channels are the **nuclear stain**
and which are **structural** markers — without the user naming a panel,
and without renaming anything (per
``feedback_no_auto_channel_rename``). Two complementary mechanisms:

- **Nuclear** is matched by a curated *lexicon* (``DAPI``, ``Hoechst``,
  ``DRAQ5``, ``DNA``, ``SYTO``). Nuclear stains are often not canonical
  panel markers and carry per-cycle suffixes (``Hoechst1``), so the
  registry alone would miss them — the no-rename rule means ``Hoechst``
  is never aliased to ``dapi``.
- **Structural** is matched by the marker *registry*'s ``family``
  column (epithelium / ECM / cytoskeleton families). These are real
  panel markers, so the registry classifies them precisely.

User-supplied marker names override either inference (resolved against
the slide's own marker list).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from coral.markers import clean_marker_name, normalize_marker_name
from coral.markers.registry import load_canonical_markers


class _MarkersLike(Protocol):
    """Anything exposing a ``markers`` list (e.g. ``CoralSlide``)."""

    @property
    def markers(self) -> list[str]: ...


__all__ = [
    "default_structural_channels",
    "infer_dapi_channel",
    "infer_dapi_index",
    "infer_structural_channels",
    "non_qc_channels",
    "resolve_marker_indices",
    "resolve_nuclear_index",
]

# Nuclear-stain name fragments (matched as substrings of the cleaned
# marker name). Mirrors the hints the fixture builder labels nuclei
# with; ``Hoechst`` stays ``Hoechst`` — never collapsed to ``dapi``.
_NUCLEAR_LEXICON = ("dapi", "hoechst", "draq5", "dna", "syto")

# Registry ``family`` values that mark a channel as structural/stromal
# scaffold (epithelium, basement membrane / ECM, fibroblast cytoskeleton).
_STRUCTURAL_FAMILIES = frozenset(
    {
        "epithelial_marker",
        "epithelial_keratin",
        "epithelial_junction",
        "ecm_matrix",
        "cytoskeleton",
    }
)

# Preferred structural markers for tissue detection, by cleaned name (with
# common aliases). Otsu finds tissue more reliably from a few strong
# scaffold stains — epithelium (pan-cytokeratin), mesenchyme (vimentin),
# and basement membrane (collagen IV) — than from every structural channel
# on the panel. detect_tissue uses whichever of these are present.
_DEFAULT_STRUCTURAL_NAMES = frozenset(
    {
        "pancytokeratin",
        "panck",
        "pan_ck",
        "cytokeratin",
        "vimentin",
        "collagen_iv",
        "collageniv",
        "col4",
        "col_iv",
        "collagen4",
    }
)


def _is_nuclear(cleaned: str) -> bool:
    """True if a cleaned marker name reads as a nuclear stain."""
    return any(term in cleaned for term in _NUCLEAR_LEXICON)


def default_structural_channels(slide: _MarkersLike) -> list[int]:
    """Indices of the preferred tissue-detection structural markers.

    The default for ``detect_tissue`` when the user names no structural
    markers: pan-cytokeratin, vimentin, and collagen IV — whichever are on
    the panel. A few strong epithelium / mesenchyme / ECM stains give a
    cleaner Otsu threshold than every structural channel. Empty if none of
    the three are present (the caller then segments on the nuclear stain).

    Args:
        slide: The slide whose ``markers`` to inspect.

    Returns:
        Sorted channel indices of the present default structural markers.

    Example:
        >>> class _S:
        ...     markers = ["Hoechst1", "PanCytokeratin", "Vimentin", "CD3"]
        >>> default_structural_channels(_S())
        [1, 2]
    """
    out = [
        i
        for i, marker in enumerate(slide.markers)
        if clean_marker_name(str(marker)) in _DEFAULT_STRUCTURAL_NAMES
    ]
    return sorted(out)


@lru_cache(maxsize=1)
def _family_by_name() -> dict[str, str]:
    """Map cleaned canonical marker name -> registry ``family``."""
    registry = load_canonical_markers()
    return {
        clean_marker_name(str(name)): str(family)
        for name, family in zip(
            registry["marker_name"], registry["family"], strict=True
        )
    }


def infer_dapi_channel(slide: _MarkersLike) -> int | None:
    """Index of the slide's nuclear-stain channel, or ``None``.

    Scans ``slide.markers`` for the first whose cleaned name contains a
    nuclear-lexicon fragment (``dapi`` / ``hoechst`` / ``draq5`` /
    ``dna`` / ``syto``). Returns ``None`` when no nuclear channel is
    found — callers fall back to a max projection rather than guessing.

    Args:
        slide: The slide whose ``markers`` to inspect.

    Returns:
        The channel index, or ``None`` if no nuclear stain is present.

    Example:
        >>> class _S:  # minimal slide-like stub
        ...     markers = ["Hoechst1", "PanCytokeratin", "Vimentin"]
        >>> infer_dapi_channel(_S())
        0
        >>> class _NoNuc:
        ...     markers = ["CD3", "CD20"]
        >>> infer_dapi_channel(_NoNuc()) is None
        True
    """
    return infer_dapi_index(list(slide.markers))


def infer_dapi_index(markers: list[str]) -> int | None:
    """Index of the nuclear-stain channel in a marker-name list, or None.

    The list-based core of :func:`infer_dapi_channel` — usable at ingest
    time (to store the result on the slide) before an ``CoralSlide``
    exists.

    Args:
        markers: Marker names in channel order.

    Returns:
        The channel index, or ``None`` if no nuclear stain is present.

    Example:
        >>> infer_dapi_index(["PanCytokeratin", "Hoechst1"])
        1
        >>> infer_dapi_index(["CD3", "CD20"]) is None
        True
    """
    for i, marker in enumerate(markers):
        if _is_nuclear(clean_marker_name(str(marker))):
            return i
    return None


def resolve_nuclear_index(markers: list[str], nuclear_marker: str) -> int:
    """Resolve an explicit nuclear-marker name to a channel index.

    Case-insensitive exact match against the panel's own marker names —
    both sides are passed through :func:`clean_marker_name`, so ``"DAPI"``,
    ``"dapi"`` and ``"Dapi"`` are equivalent (and ``"HLA-DR"`` matches
    ``"hla_dr"``). Used at ingest to honour a user ``--nuclear-marker``
    override instead of auto-inferring with :func:`infer_dapi_index`.

    Unlike the fuzzy :func:`resolve_marker_indices`, this requires an
    exact (cleaned) match: an explicit override that doesn't name a real
    channel is a user error, not something to guess at.

    Args:
        markers: Marker names in channel order.
        nuclear_marker: The marker name to force as the nuclear channel.

    Returns:
        The channel index of the named marker.

    Raises:
        ValueError: If ``nuclear_marker`` matches no marker on the panel.

    Example:
        >>> resolve_nuclear_index(["DAPI", "CD3"], "dapi")
        0
        >>> resolve_nuclear_index(["Hoechst1", "Hoechst2"], "Hoechst2")
        1
    """
    cleaned = clean_marker_name(str(nuclear_marker))
    cleaned_markers = [clean_marker_name(str(m)) for m in markers]
    if cleaned not in cleaned_markers:
        msg = (
            f"nuclear marker {nuclear_marker!r} not found among the "
            f"{len(cleaned_markers)} marker(s) — check the spelling, or "
            f"list them with 'coral status <slide>'."
        )
        raise ValueError(msg)
    return cleaned_markers.index(cleaned)


def infer_structural_channels(slide: _MarkersLike) -> list[int]:
    """Indices of structural markers, by registry ``family``.

    Each marker is resolved against the canonical registry; those whose
    family is a structural/stromal scaffold (epithelial, ECM, or
    cytoskeleton) are kept. Nuclear stains are excluded up front so a
    fuzzy registry match can never mislabel a stain as structural. An
    empty list is valid (no structural markers on the panel).

    Args:
        slide: The slide whose ``markers`` to inspect.

    Returns:
        Sorted channel indices of the structural markers.

    Example:
        >>> class _S:
        ...     markers = ["Hoechst1", "PanCytokeratin", "Vimentin", "CD3"]
        >>> infer_structural_channels(_S())
        [1, 2]
    """
    family_by_name = _family_by_name()
    canonical = list(family_by_name)
    out: list[int] = []
    for i, marker in enumerate(slide.markers):
        cleaned = clean_marker_name(str(marker))
        if _is_nuclear(cleaned):
            continue
        resolved, level = normalize_marker_name(
            cleaned, canonical_names=canonical
        )
        if level == "none":
            continue
        if family_by_name.get(resolved) in _STRUCTURAL_FAMILIES:
            out.append(i)
    return out


def non_qc_channels(slide: _MarkersLike) -> list[int]:
    """Indices of all channels except registry ``qc_mask`` markers.

    The all-marker max-projection fallback uses these. The registry's
    ``qc_mask`` family is the viral / pathogen markers (EBV / HBV / HIV,
    e.g. ``EBNA1``, ``LMP1``, ``P24``) — sparse, pathogen-specific
    signal, not a general tissue-extent indicator.

    Args:
        slide: The slide whose ``markers`` to inspect.

    Returns:
        Channel indices excluding markers in the ``qc_mask`` family.

    Example:
        >>> class _S:
        ...     markers = ["DAPI", "CD3", "EBNA1"]
        >>> non_qc_channels(_S())  # EBNA1 (viral) excluded
        [0, 1]
    """
    family_by_name = _family_by_name()
    canonical = list(family_by_name)
    out: list[int] = []
    for i, marker in enumerate(slide.markers):
        resolved, level = normalize_marker_name(
            clean_marker_name(str(marker)), canonical_names=canonical
        )
        if level != "none" and family_by_name.get(resolved) == "qc_mask":
            continue
        out.append(i)
    return out


def resolve_marker_indices(slide: _MarkersLike, names: list[str]) -> list[int]:
    """Resolve user-supplied marker names to channel indices.

    Each name is matched against the slide's own marker list (cleaned,
    then exact-or-fuzzy via :func:`normalize_marker_name`). Used to apply
    explicit ``nuclear_marker`` / ``structural_markers`` overrides.

    Args:
        slide: The slide whose ``markers`` to resolve against.
        names: Marker names supplied by the user.

    Returns:
        Channel indices, one per input name, in order.

    Raises:
        ValueError: If any name resolves to no channel on the slide.

    Example:
        >>> class _S:
        ...     markers = ["Hoechst1", "PanCytokeratin", "CD3"]
        >>> resolve_marker_indices(_S(), ["pancytokeratin"])
        [1]
    """
    cleaned_markers = [clean_marker_name(str(m)) for m in slide.markers]
    out: list[int] = []
    for name in names:
        resolved, level = normalize_marker_name(
            clean_marker_name(str(name)), canonical_names=cleaned_markers
        )
        if level == "none":
            msg = (
                f"marker {name!r} not found among the slide's "
                f"{len(cleaned_markers)} marker(s) — check the spelling, "
                f"or list them with 'coral status <slide>'."
            )
            raise ValueError(msg)
        out.append(cleaned_markers.index(resolved))
    return out
