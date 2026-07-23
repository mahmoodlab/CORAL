"""Cell centroid table — per-cell ``(cell_id, x, y)`` at level 0.

The canonical phenotyping join key: ``cell_id`` matches the instance
mask's labels, and ``(x, y)`` are level-0 pixel centroids. Written as a
single **CSV** (what CORAL code reads and users open).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["mask_to_centroids", "write_cell_table"]

_COLUMNS = ["cell_id", "x", "y", "cell_area", "slide"]


def mask_to_centroids(mask: np.ndarray, slide: str) -> pd.DataFrame:
    """Per-cell centroid table from an instance-label mask.

    Args:
        mask: 2-D int instance mask (0 = background, each cell a unique
            label).
        slide: Slide name, recorded in the ``slide`` column for
            cross-slide joins.

    Returns:
        A ``DataFrame`` with columns ``cell_id, x, y, cell_area, slide``
        — one row per cell, centroids in level-0 pixels, area in pixels.
        ``cell_id`` equals the mask label, so the table joins straight
        onto the stored mask.

    Example:
        >>> import numpy as np
        >>> m = np.zeros((10, 10), dtype="int32")
        >>> m[2:4, 2:4] = 1
        >>> df = mask_to_centroids(m, "demo")
        >>> (
        ...     int(df.loc[0, "cell_id"]),
        ...     float(df.loc[0, "x"]),
        ...     float(df.loc[0, "y"]),
        ...     int(df.loc[0, "cell_area"]),
        ... )
        (1, 2.5, 2.5, 4)
    """
    from skimage.measure import regionprops

    labels = np.asarray(mask, dtype=np.int32)
    rows = [
        (
            int(p.label),
            float(p.centroid[1]),
            float(p.centroid[0]),
            int(p.area),
            slide,
        )
        for p in regionprops(labels)
    ]
    return pd.DataFrame(rows, columns=_COLUMNS)


def write_cell_table(df: pd.DataFrame, csv_path: str | Path) -> None:
    """Write the cell table as a single CSV (the canonical per-cell table).

    Args:
        df: The cell table (from :func:`mask_to_centroids`).
        csv_path: Destination ``.csv`` (the path CORAL reads and users open).

    Example:
        >>> import tempfile
        >>> import numpy as np
        >>> from pathlib import Path
        >>> m = np.zeros((8, 8), dtype="int32")
        >>> m[1:3, 1:3] = 1
        >>> df = mask_to_centroids(m, "demo")
        >>> with tempfile.TemporaryDirectory() as d:
        ...     write_cell_table(df, Path(d) / "c.csv")
        ...     (Path(d) / "c.csv").exists()
        True
    """
    df.to_csv(Path(csv_path), index=False)
