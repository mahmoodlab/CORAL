"""Write a human-readable map of a slide store's zarr layout.

``<store>/structure.txt`` mirrors zarr's annotated group/array tree, so a
glance shows what a store contains (the ``0`` image, ``labels/``,
``patches/<slug>``, ``features/<slug>/<encoder>/<variant>``). Refreshed
automatically after each store-mutating stage — pure stdlib + zarr, no
torch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import zarr

logger = logging.getLogger(__name__)


def write_structure_map(store_path: str | Path) -> None:
    """Write ``<store_path>/structure.txt`` = the zarr group/array tree.

    Best-effort + tolerant: any failure is logged at DEBUG and swallowed,
    so refreshing the map can never fail the stage that triggered it.

    Args:
        store_path: The slide ``.zarr`` store directory.

    Example:
        >>> import tempfile
        >>> import numpy as np
        >>> import zarr
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "s.zarr"
        ...     _ = zarr.open_group(str(p), mode="w").create_dataset(
        ...         "0", data=np.zeros((1, 4, 4), dtype="uint8")
        ...     )
        ...     write_structure_map(p)
        ...     "0" in (p / "structure.txt").read_text()
        True
    """
    try:
        root = zarr.open_group(str(store_path), mode="r")
        (Path(store_path) / "structure.txt").write_text(str(root.tree()))
    except Exception as exc:  # noqa: BLE001 — never fail a stage on this
        logger.debug(
            "could not write structure map for %s: %s", store_path, exc
        )
