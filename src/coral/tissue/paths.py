"""Per-method tissue output paths and mask-method resolution.

Layout (ground truth)::

    <store>/tissue/tissue_<method>/tissue.geojson

The method name is the single source of truth for the subdirectory.
Discovery scans for ``tissue_*`` folders that contain a geojson — a new
registered method is picked up without editing an allowlist. The only
hardcoded method name is the multi-method default ``"otsu"``.

Example:
    >>> from pathlib import Path
    >>> tissue_dir(Path("/tmp/s.zarr"), "otsu").as_posix().endswith(
    ...     "tissue/tissue_otsu"
    ... )
    True
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TISSUE_METHOD",
    "list_tissue_methods",
    "resolve_tissue_method",
    "tissue_dir",
    "tissue_rel",
]

DEFAULT_TISSUE_METHOD = "otsu"
_PREFIX = "tissue_"


def tissue_dir(store: Path, method: str) -> Path:
    """Absolute path to ``<store>/tissue/tissue_<method>/``.

    Args:
        store: Slide ``.zarr`` directory.
        method: Segmenter / import key (e.g. ``otsu``, ``carta``,
            ``imported``).

    Returns:
        The method's output directory (may not exist yet).
    """
    return Path(store) / "tissue" / f"{_PREFIX}{method}"


def tissue_rel(method: str) -> str:
    """Store-relative prefix ``tissue/tissue_<method>`` for state outputs."""
    return f"tissue/{_PREFIX}{method}"


def list_tissue_methods(store: Path) -> list[str]:
    """Methods that have a ``tissue.geojson`` under ``tissue/tissue_*/``.

    Discovers by scanning the filesystem — not a fixed enum — so a new
    method's folder is found automatically.

    Args:
        store: Slide ``.zarr`` directory.

    Returns:
        Sorted method names (prefix stripped), possibly empty.
    """
    root = Path(store) / "tissue"
    if not root.is_dir():
        return []
    found: list[str] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(_PREFIX):
            continue
        method = child.name[len(_PREFIX) :]
        if not method:
            continue
        if (child / "tissue.geojson").exists():
            found.append(method)
    return sorted(found)


def resolve_tissue_method(store: Path, requested: str | None = None) -> str:
    """Pick which tissue method's mask to use for this store.

    Resolution:

    1. Discover present ``tissue_<method>/`` trees with a geojson.
    2. Exactly one → use it; if ``requested`` disagrees, warn and still
       use the only one.
    3. Multiple → use ``requested`` if set; else default to
       :data:`DEFAULT_TISSUE_METHOD` (``otsu``).
    4. None → raise naming the store.
    5. Multiple + ``requested`` not on disk → raise listing available.

    Args:
        store: Slide ``.zarr`` directory.
        requested: Optional ``--tissue-method`` value.

    Returns:
        The method name to read.

    Raises:
        ValueError: When no masks exist, or a required method is missing
            among several.
    """
    store = Path(store)
    present = list_tissue_methods(store)
    name = store.name

    if not present:
        msg = (
            f"{name}: no tissue mask found under tissue/tissue_*/"
            f"tissue.geojson — run coral tissue first."
        )
        raise ValueError(msg)

    if len(present) == 1:
        only = present[0]
        if requested is not None and requested != only:
            logger.warning(
                "%s: only tissue method %r is present; ignoring "
                "--tissue-method %r",
                name,
                only,
                requested,
            )
        return only

    if requested is not None:
        if requested not in present:
            msg = (
                f"{name}: tissue method {requested!r} not found; "
                f"available: {present}"
            )
            raise ValueError(msg)
        return requested

    if DEFAULT_TISSUE_METHOD in present:
        return DEFAULT_TISSUE_METHOD

    msg = (
        f"{name}: multiple tissue methods {present} and no "
        f"--tissue-method; default {DEFAULT_TISSUE_METHOD!r} is not "
        f"among them — pass --tissue-method explicitly."
    )
    raise ValueError(msg)
