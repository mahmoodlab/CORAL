"""IO utilities for CORAL.

Atomic writes, source-format readers, the harmonize layer, and
the ingest pipeline.
"""

from coral.io.atomic import atomic_write_json, atomic_write_text
from coral.io.harmonize import (
    CanonicalChannel,
    HarmonizationError,
    harmonize_to_canonical,
)
from coral.io.readers import (
    extract_channel_names,
    read_channel_tiff_dir,
    read_ometiff,
)

__all__ = [
    "CanonicalChannel",
    "HarmonizationError",
    "atomic_write_json",
    "atomic_write_text",
    "extract_channel_names",
    "harmonize_to_canonical",
    "read_channel_tiff_dir",
    "read_ometiff",
]
