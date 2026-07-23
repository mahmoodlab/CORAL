"""``additional_markers.csv`` — the CORAL↔KRONOS2 novel-marker contract.

A **novel marker** (one absent from KRONOS2's pretraining vocabulary,
``marker_metadata.csv``) needs a row here so the model can z-score it
(Journey 1) and embed its name (Journey 2). The user supplies the
biological/text columns — KRONOS2's BioLinkBERT prompt and its
compartment/family ids are built from them and cannot be invented — while
``mean``/``std`` are optional: CORAL fills them from the data when
they are left blank, and leaves user-supplied values untouched.

Same column shape as the bundled ``marker_metadata.csv`` so the model loads
both the same way; CORAL matches a slide's marker to a row by the
separator-insensitive :func:`coral.markers.normalize.normalize_key` (so
``HLA-DR`` and ``HLA_DR`` resolve to the same row). The cohort-level pooling
that turns per-slide partials into a single ``(mean, std)`` lives in
:func:`pool_marker_stats`; the prepare pass that produces those partials is in
:mod:`coral.features.prepare`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from coral.markers.normalize import normalize_key
from coral.utils.errors import CoralError

__all__ = [
    "ADDITIONAL_COLUMNS",
    "REQUIRED_TEXT_COLUMNS",
    "STAT_COLUMNS",
    "SlideStat",
    "markers_needing_stats",
    "missing_marker_rows",
    "pool_marker_stats",
    "present_keys",
    "read_additional_markers",
    "region_key",
    "validate_additional_markers",
    "write_marker_stats",
]

# Column shape mirrors KRONOS2's marker_metadata.csv (sans its index col).
ADDITIONAL_COLUMNS = [
    "marker_name",
    "compartment",
    "family",
    "compartment_desc",
    "family_desc",
    "marker_full_name",
    "mean",
    "std",
    "pretraining",
]
# Text columns the user MUST supply for every row (BioLinkBERT prompt +
# the compartment/family categorical ids).
REQUIRED_TEXT_COLUMNS = [
    "marker_name",
    "compartment",
    "family",
    "family_desc",
    "marker_full_name",
]
STAT_COLUMNS = ["mean", "std"]

_CSV_NAME = "additional_markers.csv"


class SlideStat(NamedTuple):
    """One slide's partial stats for a marker, on the scaled ``[0, 1]`` domain.

    ``var`` is the **sample** variance (``ddof=1``); it is ``0.0`` when
    ``n_pixels < 2`` (variance undefined for a single observation).
    """

    n_pixels: int
    mean: float
    var: float


def region_key(mask: np.ndarray, scaling: float) -> str:
    """Cheap reuse signature of a tissue mask + dtype scaling.

    Two prepare passes over the same masked region (same shape, same
    tissue-pixel count) at the same dtype ``scaling`` produce identical
    per-slide partials, so stats persisted under this key can be reused; a
    re-segmentation changes the pixel count and invalidates them.

    Args:
        mask: Boolean tissue mask, ``(y, x)``.
        scaling: The dtype divisor (``mean_marker._scaling_factor``).

    Returns:
        A short ``"{shape}:{n_tissue}:{scaling}"`` signature.

    Example:
        >>> import numpy as np
        >>> m = np.zeros((4, 4), dtype=bool)
        >>> m[:2] = True
        >>> region_key(m, 65535.0)
        '(4, 4):8:65535.0'
    """
    return f"{tuple(mask.shape)}:{int(mask.sum())}:{scaling}"


def _is_blank(value: object) -> bool:
    """True for ``None``, NaN, or an empty / ``"nan"`` string cell."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def read_additional_markers(path: str | Path) -> pd.DataFrame:
    """Read ``additional_markers.csv`` as an all-string, blank-filled frame.

    Missing optional columns (e.g. ``mean``/``std`` when the user supplied
    none) are added as blank so write-back always produces a
    KRONOS2-readable file; user columns and their order are preserved.

    Args:
        path: Path to the CSV the user passed via ``--additional-markers``.

    Returns:
        The table with every cell a ``str`` and blanks as ``""``.

    Raises:
        CoralError: If the file is missing or lacks a required text column.

    Example:
        >>> import tempfile, pandas as pd
        >>> from pathlib import Path
        >>> row = {c: "x" for c in REQUIRED_TEXT_COLUMNS}
        >>> row["marker_name"] = "FoxA1"
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "additional_markers.csv"
        ...     pd.DataFrame([row]).to_csv(p, index=False)
        ...     df = read_additional_markers(p)
        ...     (df["marker_name"].item(), df["mean"].item())
        ('FoxA1', '')
    """
    path = Path(path)
    if not path.exists():
        msg = f"additional markers file not found: {path}"
        raise CoralError(msg)
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in REQUIRED_TEXT_COLUMNS if c not in df.columns]
    if missing:
        msg = (
            f"{path}: additional_markers.csv is missing required "
            f"column(s): {missing}"
        )
        raise CoralError(msg)
    for col in ADDITIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def validate_additional_markers(
    df: pd.DataFrame, path: str | Path = _CSV_NAME
) -> None:
    """Raise if any row leaves a required text field blank.

    ``mean``/``std`` are **not** checked here — blank stats are the signal
    for CORAL to compute them.

    Args:
        df: A table from :func:`read_additional_markers`.
        path: Source path, only for the error message.

    Raises:
        CoralError: Listing every ``marker_name -> [missing columns]``.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {c: ["x"] for c in REQUIRED_TEXT_COLUMNS}
        ...     | {"marker_name": ["FoxA1"]}
        ... )
        >>> validate_additional_markers(df) is None
        True
    """
    bad: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        missing = [c for c in REQUIRED_TEXT_COLUMNS if _is_blank(row.get(c))]
        if missing:
            bad[str(row.get("marker_name", "?"))] = missing
    if bad:
        msg = (
            f"{path}: additional_markers.csv rows missing required text "
            f"field(s): {bad}. The user must supply these for every novel "
            f"marker (mean/std may be left blank for CORAL to compute)."
        )
        raise CoralError(msg)


def present_keys(df: pd.DataFrame) -> set[str]:
    """The set of ``normalize_key(marker_name)`` for the rows present.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"marker_name": ["HLA-DR", "FoxA1"]})
        >>> sorted(present_keys(df))
        ['foxa1', 'hladr']
    """
    return {
        normalize_key(str(m)) for m in df["marker_name"] if not _is_blank(m)
    }


def missing_marker_rows(df: pd.DataFrame, markers: Iterable[str]) -> list[str]:
    """The ``markers`` that have **no** row in ``df`` (by match key).

    Order-preserving and de-duplicated by the original spelling.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"marker_name": ["HLA-DR"]})
        >>> missing_marker_rows(df, ["HLA_DR", "FoxA1", "FoxA1"])
        ['FoxA1']
    """
    keys = present_keys(df)
    out: list[str] = []
    for m in dict.fromkeys(markers):
        if normalize_key(m) not in keys:
            out.append(m)
    return out


def markers_needing_stats(df: pd.DataFrame) -> list[str]:
    """The ``marker_name``s whose ``mean`` or ``std`` cell is blank.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "marker_name": ["FoxA1", "CD99"],
        ...         "mean": ["", "0.01"],
        ...         "std": ["", "0.02"],
        ...     }
        ... )
        >>> markers_needing_stats(df)
        ['FoxA1']
    """
    out: list[str] = []
    for _, row in df.iterrows():
        if _is_blank(row.get("mean")) or _is_blank(row.get("std")):
            out.append(str(row["marker_name"]))
    return out


def write_marker_stats(
    df: pd.DataFrame,
    path: str | Path,
    stats: dict[str, tuple[float, float]],
) -> Path:
    """Fill blank ``mean``/``std`` from ``stats`` and atomically write the CSV.

    ``stats`` is keyed by :func:`normalize_key`. Only **blank** stat cells
    are filled — a user-supplied value is never overwritten — so the call is
    idempotent. The frame is written to a temp file then ``os.replace``-d.

    Args:
        df: A table from :func:`read_additional_markers` (mutated in place).
        path: Destination CSV path.
        stats: ``{normalize_key(marker): (mean, std)}`` from
            :func:`pool_marker_stats`.

    Returns:
        The written CSV path.

    Example:
        >>> import tempfile, pandas as pd
        >>> from pathlib import Path
        >>> df = pd.DataFrame(
        ...     {c: ["x"] for c in REQUIRED_TEXT_COLUMNS}
        ...     | {"marker_name": ["FoxA1"], "mean": [""], "std": [""]}
        ... )
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "additional_markers.csv"
        ...     _ = write_marker_stats(df, p, {"foxa1": (0.1, 0.2)})
        ...     back = read_additional_markers(p)
        ...     (back["mean"].item(), back["std"].item())
        ('0.1', '0.2')
    """
    path = Path(path)
    key_to_idx = {
        normalize_key(str(m)): i for i, m in df["marker_name"].items()
    }
    for key, (mean, std) in stats.items():
        idx = key_to_idx.get(key)
        if idx is None:
            continue
        # Cells are strings (read with dtype=str); keep the column dtype.
        if _is_blank(df.at[idx, "mean"]):
            df.at[idx, "mean"] = str(mean)
        if _is_blank(df.at[idx, "std"]):
            df.at[idx, "std"] = str(std)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def pool_marker_stats(partials: Sequence[SlideStat]) -> tuple[float, float]:
    """Pool per-slide ``(n, μ, s²)`` into one cohort ``(mean, std)``.

    The exact pooled-variance decomposition (within-slide spread +
    between-slide mean shifts), so the result equals computing over all
    pixels concatenated::

        μ  = ( Σ nᵢ·μᵢ ) / ( Σ nᵢ )
        s² = [ Σ (nᵢ−1)·sᵢ² + Σ nᵢ·(μᵢ−μ)² ] / ( Σ nᵢ − 1 )

    For a single slide this reduces to that slide's own ``(μ₁, s₁²)``; with
    a single total pixel the variance is undefined and ``std`` is ``0.0``.

    Args:
        partials: One :class:`SlideStat` per slide (sample variance).

    Returns:
        ``(mean, std)`` on the same domain as the inputs.

    Raises:
        CoralError: If the total pixel count is zero.

    Example:
        >>> import numpy as np
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([5.0, 9.0])
        >>> sa = SlideStat(a.size, a.mean(), a.var(ddof=1))
        >>> sb = SlideStat(b.size, b.mean(), b.var(ddof=1))
        >>> mean, std = pool_marker_stats([sa, sb])
        >>> both = np.concatenate([a, b])
        >>> bool(np.isclose(mean, both.mean()))
        True
        >>> bool(np.isclose(std, both.std(ddof=1)))
        True
    """
    total_n = sum(p.n_pixels for p in partials)
    if total_n <= 0:
        msg = "cannot pool marker stats: zero total pixels"
        raise CoralError(msg)
    mean = sum(p.n_pixels * p.mean for p in partials) / total_n
    if total_n == 1:
        return mean, 0.0
    within = sum((p.n_pixels - 1) * p.var for p in partials)
    between = sum(p.n_pixels * (p.mean - mean) ** 2 for p in partials)
    var = (within + between) / (total_n - 1)
    return mean, math.sqrt(max(var, 0.0))
