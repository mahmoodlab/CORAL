"""Vendored Eva model (github.com/YAndrewL/Eva, Apache-2.0).

A verbatim copy of the Eva inference code (``eva``, ``mae``, ``layers``,
``masking``, ``pos_embed``, ``utils``), so CORAL no longer depends on the
external ``Eva`` package. Only three edits to the upstream source: the
intra-package imports point at ``coral.features._eva``; ``layers`` imports
its marker→gene map from :mod:`coral.markers._eva_marker_to_gene` (CORAL's
verified-identical snapshot of Eva's ``utils.constant.marker_to_gene``);
and ``mae`` loads the GenePT pickle from a conf-injected path instead of a
hardcoded CWD-relative one. See ``LICENSE`` + ``NOTICE`` for attribution.

This subpackage imports ``torch``/``timm`` at module load, so it is the
opt-in ``eva`` extra; :class:`coral.features.eva.EvaExtractor` imports it
lazily and is excluded from CORAL's lint/type/doctest passes.
"""

from coral.features._eva.utils import extract_features, load_from_hf

__all__ = ["extract_features", "load_from_hf"]
