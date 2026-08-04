"""Per-method dearray output paths.

Layout (ground truth)::

    <store>/dearray/dearray_<method>/cores.geojson
                                    overlay.png
                                    dearray.json

Mirrors :mod:`coral.tissue.paths` exactly, including discovery by scanning
for ``dearray_*`` folders that contain a ``cores.geojson``, so a method is
found without editing an allowlist.

Two methods exist, and keeping them apart is the whole reason the layout is
per-method. ``carta`` is what the detector found. ``imported`` is what a
human handed back, whether that is a QuPath correction or boxes from
someone's own script. Because they are different directories, re-running
detection cannot overwrite a human's boxes, so there is no edit to detect
and nothing to force past. :mod:`coral.tissue` settles the same question the
same way.

Example:
    >>> from pathlib import Path
    >>> dearray_dir(Path("/tmp/s.zarr"), "carta").as_posix().endswith(
    ...     "dearray/dearray_carta"
    ... )
    True
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "DEFAULT_DEARRAY_METHOD",
    "IMPORTED_METHOD",
    "CORES_FILE",
    "dearray_dir",
    "dearray_rel",
    "list_dearray_methods",
]

DEFAULT_DEARRAY_METHOD = "carta"
#: Cores supplied by a human rather than detected. Its own directory, so a
#: later ``coral ingest-wsi --dearray`` cannot touch it.
IMPORTED_METHOD = "imported"
CORES_FILE = "cores.geojson"
_PREFIX = "dearray_"


def dearray_dir(store: Path, method: str) -> Path:
    """Absolute path to ``<store>/dearray/dearray_<method>/``.

    Args:
        store: Slide ``.zarr`` directory.
        method: Method key: ``carta`` or ``imported``.

    Returns:
        The method's output directory. May not exist yet.
    """
    return Path(store) / "dearray" / f"{_PREFIX}{method}"


def dearray_rel(method: str) -> str:
    """Store-relative prefix ``dearray/dearray_<method>`` for state outputs."""
    return f"dearray/{_PREFIX}{method}"


def list_dearray_methods(store: Path) -> list[str]:
    """Methods that have a ``cores.geojson`` under ``dearray/dearray_*/``.

    Discovered by scanning rather than from a fixed list.

    Args:
        store: Slide ``.zarr`` directory.

    Returns:
        Sorted method names.
    """
    root = Path(store) / "dearray"
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith(_PREFIX):
            continue
        if (child / CORES_FILE).exists():
            found.append(child.name[len(_PREFIX) :])
    return found
