"""Marker handling — name cleaning, normalization, panel resolution.

``clean_marker_name`` + ``normalize_marker_name`` live in
``coral.markers.normalize``; the canonical registry loader lives in
``coral.markers.registry``. Encoder-specific marker resolution is kept
separate: ``eva_genes`` (GenePT names) and ``kronos1_markers``
(``resolve_markers`` → KRONOS1 vocab ids + z-score stats).
"""

from coral.markers.eva_genes import get_mappable_markers, map_to_eva_name
from coral.markers.kronos1_markers import MarkerRecord, resolve_markers
from coral.markers.normalize import (
    MatchLevel,
    MatchStatus,
    clean_marker_name,
    normalize_marker_name,
)

__all__ = [
    "MarkerRecord",
    "MatchLevel",
    "MatchStatus",
    "clean_marker_name",
    "get_mappable_markers",
    "map_to_eva_name",
    "normalize_marker_name",
    "resolve_markers",
]
