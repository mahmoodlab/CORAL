"""Shared helper: open a TIFF as a memory-mapped array with fallback.

Adapted from the reference KRONOS preprocessing pipeline. Both
readers use this to avoid loading the full file into RAM for large
inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def open_with_memmap_fallback(path: Path) -> np.ndarray:
    """Open a TIFF as a memory-mapped array; fall back to in-RAM load.

    ``tifffile.memmap`` only works for uncompressed, contiguous,
    single-sample-per-pixel layouts. When it can't (compressed
    pages, tiled layouts, multi-page non-contiguous), we fall back
    to ``tifffile.imread`` which loads everything into RAM.

    Args:
        path: Path to a TIFF file.

    Returns:
        The image array. May be a ``np.memmap`` (lazy, disk-backed)
        or a regular ``np.ndarray`` (RAM-resident) depending on the
        file's layout.

    Example:
        >>> import tempfile
        >>> import numpy as np
        >>> import tifffile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "tiny.tif"
        ...     tifffile.imwrite(p, np.zeros((4, 8), dtype=np.uint16))
        ...     arr = open_with_memmap_fallback(p)
        ...     arr.shape
        (4, 8)
    """
    try:
        return tifffile.memmap(str(path))
    except (ValueError, OSError):
        return tifffile.imread(str(path))
