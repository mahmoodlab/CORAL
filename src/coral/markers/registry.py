"""Load the canonical CORAL marker registry + build its match index.

The registry is a committed seed (``data/registry_v1.csv``, generated
by ``data/_make_registry.py``) listing every canonical marker with six
descriptive columns. ``load_canonical_markers`` returns it as a pandas
DataFrame; ``canonical_key_index`` turns it into the exact-match index
used by :func:`coral.markers.normalize.match_marker`.
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path

import pandas as pd

from coral.markers.normalize import normalize_key

__all__ = ["canonical_key_index", "load_canonical_markers"]

logger = logging.getLogger(__name__)

_SEED = "registry_v1.csv"


def load_canonical_markers(
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the canonical marker registry as a DataFrame.

    Args:
        csv_path: Registry CSV to load. Defaults to the shipped seed
            (``coral.markers.data/registry_v1.csv``) resolved via
            ``importlib.resources``. The leading ``#`` provenance row
            is skipped on read.

    Returns:
        DataFrame with the six descriptive columns ``marker_name``,
        ``compartment``, ``family``, ``compartment_desc``,
        ``family_desc`` and ``marker_full_name``.

    Raises:
        FileNotFoundError: If an explicit ``csv_path`` does not exist.

    Example:
        >>> df = load_canonical_markers()
        >>> list(df.columns)[:2]
        ['marker_name', 'compartment']
        >>> bool((df["marker_name"] == "DAPI").any())
        True
    """
    if csv_path is None:
        seed = importlib.resources.files("coral.markers.data") / _SEED
        with importlib.resources.as_file(seed) as path:
            return pd.read_csv(path, comment="#")
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"registry CSV not found: {path}")
    return pd.read_csv(path, comment="#")


def canonical_key_index(
    registry: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Map each canonical marker's match key to its display name.

    Builds ``{normalize_key(marker_name): marker_name}`` — the exact-match
    target for :func:`coral.markers.normalize.match_marker`. Duplicate
    ``marker_name`` rows and key **collisions** (two distinct canonical
    names that collapse to the same key) are warned and the first
    occurrence kept, so a registry blemish surfaces in the logs without
    blocking ingest. This is the load-time sanity check.

    Args:
        registry: A loaded registry DataFrame; defaults to
            :func:`load_canonical_markers`.

    Returns:
        ``{match_key: canonical_display_name}``.

    Example:
        >>> idx = canonical_key_index()
        >>> idx["dapi"]
        'DAPI'
    """
    df = load_canonical_markers() if registry is None else registry
    index: dict[str, str] = {}
    seen: set[str] = set()
    for raw in df["marker_name"]:
        name = str(raw)
        if name in seen:
            logger.debug(
                "registry: duplicate marker_name %r — keeping the first.",
                name,
            )
            continue
        seen.add(name)
        key = normalize_key(name)
        existing = index.get(key)
        if existing is not None and existing != name:
            logger.debug(
                "registry: %r and %r collapse to the same match key %r"
                " — keeping %r.",
                existing,
                name,
                key,
                existing,
            )
            continue
        index[key] = name
    return index
