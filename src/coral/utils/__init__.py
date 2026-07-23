"""Small utility helpers — errors, hashing, time, registry factory."""

from coral.utils.device import resolve_device
from coral.utils.errors import (
    CohortMetadataError,
    CoralError,
    HarmonizationError,
    MarkerError,
    ReaderError,
)
from coral.utils.hashing import stable_slide_hash
from coral.utils.logging import setup_logging
from coral.utils.registry import make_register_decorator
from coral.utils.time import epoch_now, now_iso

__all__ = [
    "CohortMetadataError",
    "CoralError",
    "HarmonizationError",
    "MarkerError",
    "ReaderError",
    "epoch_now",
    "make_register_decorator",
    "now_iso",
    "resolve_device",
    "setup_logging",
    "stable_slide_hash",
]
