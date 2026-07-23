"""Source-format readers for CORAL ingest.

Each reader opens one kind of source and returns
``(image, channels, mpp, source_meta)`` in whatever raw shape the
source provides — harmonization to canonical ``(c, y, x)`` is the
caller's job.
"""

from coral.io.readers.channel_tiff_dir import read_channel_tiff_dir
from coral.io.readers.ome_tiff import extract_channel_names, read_ometiff

__all__ = [
    "extract_channel_names",
    "read_channel_tiff_dir",
    "read_ometiff",
]
