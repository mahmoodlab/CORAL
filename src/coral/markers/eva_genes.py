"""Eva marker mapping — KRONOS marker names → Eva ``bms`` (marker names).

``EvaExtractor`` feeds Eva a list of **Eva marker names** (``bms``); Eva's
``MarkerEmbeddingGenePT`` does ``gene = marker_to_gene[name]`` then looks
up the GenePT embedding, so each ``bms`` entry must be a **key** of Eva's
``marker_to_gene`` (a gene symbol would ``KeyError``). This module lifts
the kronos ``extractors/eva.py`` mapping **verbatim** — its own
normalization (lower-case, ``-``/space → ``_``; a compact form dropping
those) plus ~50 manual aliases — *not* CORAL's ``clean_marker_name`` (which
drops ``/`` + folds ``α``), because the gold was produced by this exact
logic. The Eva vocabulary is vendored in
:mod:`coral.markers._eva_marker_to_gene`.

Pure + model-free: the correctness core of the Eva extractor, fully
testable without the model.
"""

from __future__ import annotations

import warnings

from coral.markers._eva_marker_to_gene import EVA_MARKER_TO_GENE

__all__ = ["get_mappable_markers", "map_to_eva_name"]

# Nuclear / blank channels the kronos eva.py skips before mapping (note:
# "dapi" is NOT here — DAPI is a real Eva vocab marker, so it is kept).
_SKIP_SUBSTRINGS = ("hoechst", "empty", "blank", "draq")

# Manual KRONOS-name → Eva-name aliases, lifted verbatim from kronos
# extractors/eva.py::_setup_marker_mapping (keyed by the eva.py primary
# normalized form: lower-case, ``-``/space → ``_``).
_MANUAL_ALIASES = {
    "ki_67": "Ki67",
    "ki67": "Ki67",
    "pd_l1": "PDL1",
    "pdl1": "PDL1",
    "pd_1": "PD1",
    "pd1": "PD1",
    "bcl_2": "BCL2",
    "bcl2": "BCL2",
    "t_bet": "Tbet",
    "tbet": "Tbet",
    "b_catenin": "bCatenin",
    "bcatenin": "bCatenin",
    "beta_catenin": "bCatenin",
    "ido_1": "IDO1",
    "ido1": "IDO1",
    "cd3": "CD3e",
    "ccr6": "CD196",
    "ccr4": "CD194",
    "cla_cd162": "CD162",
    "cd162": "CD162",
    "muc_1": "CD227",
    "muc1": "CD227",
    "cd1a": "CD1c",
    "mastcell_tryptase": "Tryptase",
    "mast_cell_tryptase": "Tryptase",
    "tryptase": "Tryptase",
    "ph2ax": "yH2AX",
    "p16ink4a": "p16",
    "p16_ink4a": "p16",
    "tcf17": "TCF1",
    "tcf1_7": "TCF1",
    "tcf1": "TCF1",
    "col1a": "CollagenI",
    "col1a1": "CollagenI",
    "collagen_i": "CollagenI",
    "collageni": "CollagenI",
    "mhci": "HLA-ABC",
    "mhc_i": "HLA-ABC",
    "hla_abc": "HLA-ABC",
    "hla1": "HLA-ABC",
    "ifn_y": "IFNg",
    "ifny": "IFNg",
    "ifng": "IFNg",
    "tox_tox2": "TOX",
    "tox": "TOX",
    "cytokeratin": "PanCK",
    "pan_ck": "PanCK",
    "panck": "PanCK",
    "collagen": "CollagenI",
    "gzmb": "GranzymeB",
    "granzymeb": "GranzymeB",
    "granzyme_b": "GranzymeB",
}


def _norm(name: str) -> str:
    """eva.py primary normalization: lower-case, ``-``/space → ``_``."""
    return name.lower().replace("-", "_").replace(" ", "_")


def _norm_compact(name: str) -> str:
    """eva.py compact normalization: lower-case, drop ``-``/space/``_``."""
    return name.lower().replace("-", "").replace(" ", "").replace("_", "")


def _build_lookup() -> dict[str, str]:
    """Build ``{normalized: eva_name}`` from the vocab + manual aliases."""
    lookup: dict[str, str] = {}
    for eva_name in EVA_MARKER_TO_GENE:
        lookup[_norm(eva_name)] = eva_name
        lookup[_norm_compact(eva_name)] = eva_name
    lookup.update(_MANUAL_ALIASES)
    return lookup


_LOOKUP = _build_lookup()


def map_to_eva_name(marker: str) -> str | None:
    """Map a KRONOS marker name to an Eva ``bms`` name, or ``None``.

    Reproduces kronos ``eva.py``: try the normalized then the compact
    form against the Eva-name lookup; return the Eva name iff it is a
    key of Eva's ``marker_to_gene`` (else ``None`` — unmappable).

    Args:
        marker: A raw KRONOS marker / channel name.

    Returns:
        The matching Eva marker name, or ``None`` if unmappable.

    Example:
        >>> map_to_eva_name("CD3")
        'CD3e'
        >>> map_to_eva_name("Ki-67")
        'Ki67'
        >>> map_to_eva_name("EBNA1") is None
        True
    """
    eva_name = _LOOKUP.get(_norm(marker)) or _LOOKUP.get(_norm_compact(marker))
    if eva_name is not None and eva_name in EVA_MARKER_TO_GENE:
        return eva_name
    return None


def get_mappable_markers(
    markers: list[str],
) -> tuple[list[int], list[str], list[str]]:
    """Resolve a panel to Eva's mappable channels (kronos eva.py logic).

    Nuclear/blank stains (``hoechst``/``empty``/``blank``/``draq``) are
    silently skipped (kept out of ``bms`` but *not* reported as dropped,
    matching the kronos pipeline); every other marker maps via
    :func:`map_to_eva_name` or is dropped (unmappable — e.g. the 9 viral
    markers). ``DAPI`` is a real Eva vocab marker, so it is kept. Pure +
    model-free.

    Args:
        markers: The panel's marker names in channel order.

    Returns:
        ``(kept_indices, eva_names, dropped)`` — the channel indices to
        feed Eva, their Eva ``bms`` names (aligned to ``kept_indices``),
        and the unmappable markers. Warns naming the dropped markers.

    Example:
        >>> get_mappable_markers(["Hoechst1", "CD3", "CD20"])
        ([1, 2], ['CD3e', 'CD20'], [])
    """
    kept_indices: list[int] = []
    eva_names: list[str] = []
    dropped: list[str] = []
    for i, marker in enumerate(markers):
        if any(s in _norm(marker) for s in _SKIP_SUBSTRINGS):
            continue
        eva_name = map_to_eva_name(marker)
        if eva_name is not None:
            kept_indices.append(i)
            eva_names.append(eva_name)
        else:
            dropped.append(marker)
    if dropped:
        warnings.warn(
            f"Eva: dropping {len(dropped)} unmappable markers: {dropped}",
            stacklevel=2,
        )
    return kept_indices, eva_names, dropped
