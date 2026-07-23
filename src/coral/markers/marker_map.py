"""The job-dir marker map — the user-reviewed raw→canonical mapping.

At ingest, a cohort's unique raw marker names are matched against the
canonical registry (:func:`coral.markers.normalize.match_marker`) and the
result is written to the job dir as ``marker_map.csv`` for review. The CSV
lists only the **kept** analysis panel (marker names) — which channels are
kept is decided at ingest and frozen in each store's ``.zattrs``, not
edited here. For each ``REVIEW`` row the user either sets
``mapped_canonical_name`` to the right registry name (a typo or variant),
or sets ``status`` to ``NOVEL`` for a genuinely new marker (their own name
is kept). Validation hard-errors on a blank review row or an invalid
name, so a store is never left half-resolved.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from coral.markers.normalize import (
    NOVEL_TOKEN,
    RESOLVED_TOKEN,
    REVIEW_TOKEN,
    MatchStatus,
    clean_marker_name,
    match_marker,
)

__all__ = [
    "MAP_COLUMNS",
    "NOVEL_TOKEN",
    "RESOLVED_TOKEN",
    "REVIEW_TOKEN",
    "MarkerMapError",
    "apply_blank_drop",
    "apply_hoechst_drop",
    "apply_panel_to_keep",
    "build_marker_map",
    "is_qc_channel",
    "promote_reviewed",
    "read_marker_map",
    "resolve_marker_map",
    "validate_and_flag",
    "validate_marker_map",
    "write_marker_map",
]

logger = logging.getLogger(__name__)


class MarkerMapError(ValueError):
    """A ``marker_map.csv`` that cannot be applied as-is.

    Carries the offending rows, categorised so a CLI can render each
    problem on its own line instead of one dense string:

    - ``invalid_names``: ``(original, value)`` where ``value`` is neither
      a registry marker name nor ``NOVEL`` (a typo, or a real marker the
      user forgot to mark novel).
    - ``unresolved``: ``(original, status)`` for kept rows left blank —
      the user must supply a name or mark them ``NOVEL``. The ``status``
      is carried so the renderer can flag the common trap of setting
      ``status`` without filling the name.

    Subclasses :class:`ValueError`, so existing ``except ValueError`` sites
    keep catching it.
    """

    def __init__(
        self,
        *,
        invalid_names: list[tuple[str, str]] | None = None,
        unresolved: list[tuple[str, str]] | None = None,
    ) -> None:
        """Store the categorised offending rows + a summary message."""
        self.invalid_names = invalid_names or []
        self.unresolved = unresolved or []
        super().__init__(self._summary())

    def _summary(self) -> str:
        parts: list[str] = []
        if self.invalid_names:
            parts.append(f"{len(self.invalid_names)} invalid mapped name(s)")
        if self.unresolved:
            parts.append(f"{len(self.unresolved)} unresolved marker(s)")
        return "marker_map.csv: " + ", ".join(parts)


MAP_COLUMNS = [
    "original_name",
    "mapped_canonical_name",
    "status",
    "suggestion_1",
    "suggestion_2",
]
# ``keep_for_analysis`` is an in-memory ingest-working column, NOT persisted:
# the analysis panel is frozen at ingest in each store's ``.zattrs``. It rides
# along during ingest so ``--subset`` / QC drops can set it, is used to filter
# the CSV down to the kept panel, then dropped — ``write_marker_map`` persists
# only :data:`MAP_COLUMNS`.
_INGEST_COLUMNS = [*MAP_COLUMNS, "keep_for_analysis"]
_CSV_NAME = "marker_map.csv"


def _is_blank_empty(name: str) -> bool:
    """True if a channel name reads as a blank or empty QC channel.

    Matches names starting with ``blank`` or ``empty`` (case-insensitive),
    so ``blank``, ``emptyA488-1``, and ``emptyCy3-2`` are all caught.
    """
    return name.strip().lower().startswith(("blank", "empty"))


def _is_hoechst(name: str) -> bool:
    """True if a channel name reads as a Hoechst nuclear stain."""
    return "hoechst" in name.strip().lower()


def is_qc_channel(name: str) -> bool:
    """True if a channel name reads as a QC channel (blank/empty/Hoechst).

    These are the channels ingest auto-excludes from analysis by default
    (unless kept with ``--keep-hoechst``, named in ``--subset``, or set as
    the nuclear stain). Useful for labelling why a channel was excluded.

    Example:
        >>> (
        ...     is_qc_channel("Hoechst1"),
        ...     is_qc_channel("blank"),
        ...     is_qc_channel("CD8"),
        ... )
        (True, True, False)
    """
    return _is_blank_empty(name) or _is_hoechst(name)


def build_marker_map(
    raw_names: list[str],
    key_index: dict[str, str],
    *,
    n_suggestions: int = 2,
) -> pd.DataFrame:
    """Build a marker map from a cohort's raw names via ``match_marker``.

    One row per **unique** raw name (order preserved). An exact key hit
    fills ``mapped_canonical_name`` with ``status=RESOLVED_TOKEN``; a miss
    leaves ``mapped_canonical_name`` blank, sets ``status=REVIEW_TOKEN``,
    and records up to two advisory suggestions for the user.

    Args:
        raw_names: Observed channel/marker names (repeats are collapsed).
        key_index: ``{normalize_key(canonical): display}`` from
            :func:`coral.markers.registry.canonical_key_index`.
        n_suggestions: Advisory candidates per review row.

    Returns:
        A DataFrame with columns :data:`MAP_COLUMNS`.

    Example:
        >>> idx = {"cd8": "CD8", "dapi": "DAPI"}
        >>> df = build_marker_map(["DAPI", "CD-8", "FancyX"], idx)
        >>> list(df["status"])
        ['RESOLVED', 'RESOLVED', 'REVIEW']
        >>> df.loc[
        ...     df["original_name"] == "CD-8", "mapped_canonical_name"
        ... ].item()
        'CD8'
    """
    rows: list[dict[str, str]] = []
    for name in dict.fromkeys(raw_names):
        canonical, status, suggestions = match_marker(
            name, key_index, n_suggestions=n_suggestions
        )
        if is_qc_channel(name):
            # A Hoechst stain, or a blank/empty QC channel, is recognised
            # as itself (its own cleaned name) and resolved — never
            # matched to a protein, and not flagged for review (we drop it
            # by default).
            canonical = clean_marker_name(name)
            status = RESOLVED_TOKEN
            suggestions = []
        padded = [*suggestions, "", ""][:2]
        rows.append(
            {
                "original_name": name,
                "mapped_canonical_name": canonical or "",
                "status": status,
                # Every marker is kept for analysis by default, review rows
                # included; the QC auto-drop and a --subset set "no" later.
                # This working column is not persisted (see _INGEST_COLUMNS).
                "keep_for_analysis": "yes",
                "suggestion_1": padded[0],
                "suggestion_2": padded[1],
            }
        )
    return pd.DataFrame(rows, columns=_INGEST_COLUMNS)


def write_marker_map(df: pd.DataFrame, job_dir: str | Path) -> Path:
    """Atomically write ``marker_map.csv`` into ``job_dir``.

    The CSV is the single, user-facing source of truth for marker names. It
    is written to a temp file then atomically replaced, so a crash never
    leaves a half-written map. Only :data:`MAP_COLUMNS` is persisted — an
    in-memory ``keep_for_analysis`` working column is dropped here (the
    panel is frozen in each store's ``.zattrs``, not the CSV).

    Args:
        df: The marker map to write (kept rows only; extra columns ignored).
        job_dir: Directory to write into (created if missing).

    Returns:
        The path to the written ``marker_map.csv``.
    """
    out = Path(job_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / _CSV_NAME

    tmp_csv = csv_path.with_name(_CSV_NAME + ".tmp")
    df.reindex(columns=MAP_COLUMNS).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, csv_path)

    # Remove a mirror left by an older version, so only the CSV remains.
    (out / "marker_map.parquet").unlink(missing_ok=True)

    return csv_path


def read_marker_map(job_dir: str | Path) -> pd.DataFrame | None:
    """Read the marker map from ``job_dir`` — the user-edited CSV.

    All cells are returned as strings with blanks as ``""``.

    Args:
        job_dir: Directory holding the map.

    Returns:
        The map DataFrame, or ``None`` when no map exists (first ingest).
    """
    csv_path = Path(job_dir) / _CSV_NAME
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path, dtype=str).fillna("").astype(str)


def validate_marker_map(df: pd.DataFrame, valid_names: set[str]) -> None:
    """Validate the whole edited map; raise on anything unapplyable.

    The map lists only the kept analysis panel (excluded / QC channels are
    not persisted), so every row must carry a resolved name. Walks every
    row and collects **all** problems in one pass, so a single run surfaces
    every mistake the user introduced — not just the first:

    - a row mapped to a name that is neither a registry marker nor
      ``NOVEL`` (a typo, or a real marker not marked novel);
    - a ``REVIEW`` row left blank — it must be resolved or marked
      ``NOVEL``, never applied half-resolved.

    A recognised QC channel the user deliberately kept (Hoechst / blank /
    empty) is a self-map, not a registry name, and is skipped.

    Args:
        df: A marker map (e.g. from :func:`read_marker_map`).
        valid_names: The canonical registry display names.

    Raises:
        MarkerMapError: Carrying every offending row, categorised.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "original_name": ["CD3"],
        ...         "mapped_canonical_name": ["CD3"],
        ...         "status": ["RESOLVED"],
        ...     }
        ... )
        >>> validate_marker_map(df, {"CD3"})  # valid -> no error
    """
    invalid_names: list[tuple[str, str]] = []
    unresolved: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        original = str(row["original_name"])
        status = str(row["status"]).strip().upper()
        value = str(row["mapped_canonical_name"]).strip()
        if is_qc_channel(original):
            continue  # deliberately-kept QC self-map; not a registry name
        is_novel = status == NOVEL_TOKEN or value.upper() == NOVEL_TOKEN
        if not value:
            # A kept marker must carry a name — a blank mapped column is
            # ALWAYS a hard error, even with status=NOVEL (the user must
            # supply their own marker name, never left for us to invent).
            # Carry the status so the renderer can flag a RESOLVED/NOVEL
            # row whose name the user forgot to fill.
            unresolved.append((original, status))
        elif not is_novel and value not in valid_names:
            invalid_names.append((original, value))
    if invalid_names or unresolved:
        raise MarkerMapError(
            invalid_names=invalid_names,
            unresolved=unresolved,
        )


def _reset_flagged_to_review(
    df: pd.DataFrame, exc: MarkerMapError
) -> tuple[pd.DataFrame, list[str]]:
    """Set ``status=REVIEW`` for the rows an error flagged as unusable.

    A row whose ``mapped_canonical_name`` is invalid, or blank while kept,
    is not resolved — whatever ``status`` the user typed. Reset those rows
    to ``REVIEW`` (keeping the value they entered, so they can see and fix
    it). Returns the updated frame and the original names changed.
    """
    flagged = {o for o, _ in exc.invalid_names} | {
        o for o, _ in exc.unresolved
    }
    out = df.copy()
    changed: list[str] = []
    for i, row in out.iterrows():
        original = str(row["original_name"])
        already = str(row["status"]).strip().upper() == REVIEW_TOKEN
        if original in flagged and not already:
            out.at[i, "status"] = REVIEW_TOKEN
            changed.append(original)
    return out, changed


def validate_and_flag(
    df: pd.DataFrame, valid_names: set[str], job_dir: str | Path
) -> None:
    """Validate the map; on failure, reset the bad rows to ``REVIEW``.

    Runs :func:`validate_marker_map`. If it raises, any invalid or
    unresolved row has its ``status`` reset to ``REVIEW`` and the CSV is
    rewritten before the error is re-raised — so a mapping the user forced
    to ``RESOLVED`` (or ``NOVEL``) but got wrong never stays stuck in that
    state; the next run sees it as a review row again.

    Args:
        df: The marker map to validate.
        valid_names: The canonical registry display names.
        job_dir: Directory holding ``marker_map.csv`` (rewritten on reset).

    Raises:
        MarkerMapError: The original validation error, after the reset.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "original_name": ["CD3"],
        ...         "mapped_canonical_name": ["CD3"],
        ...         "status": ["RESOLVED"],
        ...         "keep_for_analysis": ["yes"],
        ...     }
        ... )
        >>> validate_and_flag(df, {"CD3"}, ".")  # valid -> no error, no write
    """
    try:
        validate_marker_map(df, valid_names)
    except MarkerMapError as exc:
        fixed, changed = _reset_flagged_to_review(df, exc)
        if changed:
            write_marker_map(fixed, job_dir)
            logger.warning(
                "reset %d marker(s) to REVIEW in marker_map.csv "
                "(their mapping was invalid): %s",
                len(changed),
                ", ".join(changed),
            )
        raise


def promote_reviewed(
    df: pd.DataFrame, valid_names: set[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Promote ``REVIEW``/``NOVEL`` rows whose name is in the registry.

    Any ``REVIEW`` **or** ``NOVEL`` row whose ``mapped_canonical_name`` is
    a registry marker *is* a resolved marker — its status should say so
    (a user who marked a marker ``NOVEL`` then mapped it to a real registry
    name resolved it). Returns a copy of the map with those rows flipped to
    ``RESOLVED``, plus the promoted original names, so the caller can
    rewrite the CSV and keep it consistent with the store. Run **after**
    :func:`validate_marker_map` (every filled name is already known valid).

    Args:
        df: A validated marker map.
        valid_names: The canonical registry display names.

    Returns:
        ``(updated_df, promoted_original_names)``.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "original_name": ["CD31"],
        ...         "mapped_canonical_name": ["CD163"],
        ...         "status": ["REVIEW"],
        ...         "keep_for_analysis": ["yes"],
        ...     }
        ... )
        >>> out, promoted = promote_reviewed(df, {"CD163"})
        >>> (promoted, out["status"].iloc[0])
        (['CD31'], 'RESOLVED')
    """
    out = df.copy()
    promoted: list[str] = []
    for i, row in out.iterrows():
        status = str(row["status"]).strip().upper()
        value = str(row["mapped_canonical_name"]).strip()
        if status in (REVIEW_TOKEN, NOVEL_TOKEN) and value in valid_names:
            out.at[i, "status"] = RESOLVED_TOKEN
            promoted.append(str(row["original_name"]))
    return out, promoted


def resolve_marker_map(
    df: pd.DataFrame,
) -> dict[str, tuple[str, MatchStatus, bool]]:
    """Resolve a (validated) map to ``{original: (name, level, keep)}``.

    Per row:

    - ``status == NOVEL`` (or ``mapped_canonical_name == NOVEL``) → a
      **novel** marker:
      ``(clean(mapped_canonical_name or original), NOVEL_TOKEN)`` — the
      user's own name is kept (lower-cased), or the original if none.
    - a registry name in ``mapped_canonical_name`` →
      ``(name, RESOLVED_TOKEN)`` (auto-matched or a user-corrected typo).
    - blank → ``("", REVIEW_TOKEN)`` — an unresolved row keeps an empty
      marker (validate rejects a *kept* blank before resolve runs).

    The trailing keep flag is ``False`` only when ``keep_for_analysis``
    reads ``no`` (the user excluded the channel); blank/``yes`` → ``True``.

    Assumes the map already passed :func:`validate_marker_map`.

    Args:
        df: A validated marker map.

    Returns:
        ``{original_name: (resolved_name, match_level, keep)}``.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "original_name": ["DAPI", "MyTarget"],
        ...         "mapped_canonical_name": ["DAPI", "MyTarget"],
        ...         "status": ["RESOLVED", "NOVEL"],
        ...         "keep_for_analysis": ["yes", "no"],
        ...         "suggestion_1": ["", ""],
        ...         "suggestion_2": ["", ""],
        ...     }
        ... )
        >>> r = resolve_marker_map(df)
        >>> (r["DAPI"], r["MyTarget"])
        (('DAPI', 'RESOLVED', True), ('MyTarget', 'NOVEL', False))
    """
    out: dict[str, tuple[str, MatchStatus, bool]] = {}
    for _, row in df.iterrows():
        original = str(row["original_name"])
        status = str(row["status"]).strip().upper()
        value = str(row["mapped_canonical_name"]).strip()
        keep = str(row.get("keep_for_analysis", "")).strip().lower() != "no"
        is_novel = status == NOVEL_TOKEN or value.upper() == NOVEL_TOKEN
        if is_novel:
            # Keep the user's EXACT name in the resolution (for the
            # terminal); the store re-cleans it for downstream matching.
            keep_name = (
                value if value and value.upper() != NOVEL_TOKEN else original
            )
            out[original] = (keep_name, NOVEL_TOKEN, keep)
        elif value:
            out[original] = (value, RESOLVED_TOKEN, keep)
        else:
            # Unresolved: never invent a name. A kept blank is rejected by
            # validate_marker_map before resolve runs; an excluded blank
            # keeps an empty marker (it is not used in analysis).
            out[original] = ("", REVIEW_TOKEN, keep)
    return out


def apply_panel_to_keep(
    df: pd.DataFrame,
    channels: Any,  # noqa: ANN401 — a Selection
    *,
    label: str = "subset",
) -> None:
    """Seed the ``keep_for_analysis`` column from a selection (in place).

    Each marker's effective name — its ``mapped_canonical_name``, or the
    cleaned ``original_name`` when unresolved — is matched against the
    selection's ``include`` (default: all) and ``exclude`` globs,
    case-insensitively. A marker matching an include glob and no exclude
    glob gets ``keep="yes"``; every other marker gets ``keep="no"``.
    Every include and exclude pattern must match at least one marker, or
    the selection names a marker that isn't there and a ``ValueError`` is
    raised.

    Args:
        df: A marker map; its ``keep_for_analysis`` column is overwritten.
        channels: A channel selection — anything with ``include`` /
            ``exclude`` glob lists (a :class:`coral.config.subset.Selection`).
        label: Name of the selection, used only in error messages.

    Raises:
        ValueError: If an include or exclude pattern matches no marker.

    Example:
        For markers ``DAPI, CD3, CD8`` and ``include=["DAPI", "CD*"]`` all
        three are kept; ``include=["DAPI"]`` keeps only ``DAPI``.
    """
    import fnmatch

    effective = [
        (
            str(row["mapped_canonical_name"]).strip()
            or clean_marker_name(str(row["original_name"]))
        ).lower()
        for _, row in df.iterrows()
    ]
    include = [p.lower() for p in channels.include] or ["*"]
    exclude = [p.lower() for p in channels.exclude]
    for orig in channels.include:
        if not any(fnmatch.fnmatch(m, orig.lower()) for m in effective):
            msg = (
                f"subset {label!r} include(s) markers {orig!r}, which "
                f"matches no marker in the image."
            )
            raise ValueError(msg)
    for orig in channels.exclude:
        if not any(fnmatch.fnmatch(m, orig.lower()) for m in effective):
            msg = (
                f"subset {label!r} exclude(s) markers {orig!r}, which "
                f"matches no marker in the image."
            )
            raise ValueError(msg)
    df["keep_for_analysis"] = [
        "yes"
        if any(fnmatch.fnmatch(m, p) for p in include)
        and not any(fnmatch.fnmatch(m, p) for p in exclude)
        else "no"
        for m in effective
    ]


def _drop_category(
    df: pd.DataFrame,
    predicate: Any,  # noqa: ANN401 — a (name: str) -> bool callable
    *,
    keep_all: bool,
    channels: Any = None,  # noqa: ANN401 — a Selection or None
    extra_keep: tuple[str, ...] = (),
) -> list[str]:
    """Set ``keep_for_analysis="no"`` for matching rows; return them.

    A row whose ``original_name`` satisfies ``predicate`` is dropped
    unless ``keep_all`` is True, its cleaned name is in ``extra_keep``, or
    a channel ``include`` pattern *other than* ``"*"`` matches it (and no
    ``exclude`` does). A bare ``"*"`` never rescues. Updates the column in
    place; returns the dropped ``original_name`` values, for logging.
    """
    if keep_all:
        return []
    import fnmatch

    specific = [
        p.lower()
        for p in (channels.include if channels else [])
        if p.strip() != "*"
    ]
    exclude = [p.lower() for p in (channels.exclude if channels else [])]
    extra = {clean_marker_name(n).lower() for n in extra_keep}
    keeps = list(df["keep_for_analysis"])
    dropped: list[str] = []
    for i, (_, row) in enumerate(df.iterrows()):
        original = str(row["original_name"])
        if not predicate(original):
            continue
        eff = (
            str(row["mapped_canonical_name"]).strip()
            or clean_marker_name(original)
        ).lower()
        rescued = eff in extra or (
            any(fnmatch.fnmatch(eff, p) for p in specific)
            and not any(fnmatch.fnmatch(eff, p) for p in exclude)
        )
        if rescued:
            keeps[i] = "yes"
        else:
            keeps[i] = "no"
            dropped.append(original)
    df["keep_for_analysis"] = keeps
    return dropped


def apply_blank_drop(
    df: pd.DataFrame,
    *,
    channels: Any = None,  # noqa: ANN401 — a Selection or None
) -> list[str]:
    """Drop blank/empty QC channels; return the dropped names.

    Blank/empty channels (``blank``, ``emptyA488-1``, ...) get
    ``keep_for_analysis="no"`` unless the channel selection specifically
    (non-``*``) includes them.

    Example:
        For a map with ``blank`` and ``emptyA488-1`` rows and no subset,
        ``apply_blank_drop(df)`` drops both and returns
        ``["blank", "emptyA488-1"]``.
    """
    return _drop_category(
        df, _is_blank_empty, keep_all=False, channels=channels
    )


def apply_hoechst_drop(
    df: pd.DataFrame,
    *,
    keep_hoechst: bool,
    channels: Any = None,  # noqa: ANN401 — a Selection or None
    nuclear_marker: str | None = None,
) -> list[str]:
    """Drop Hoechst channels by default; return the dropped names.

    Hoechst stains are usually per-cycle QC; DRAQ5/DAPI are the working
    nuclear. So every Hoechst gets ``keep_for_analysis="no"`` unless
    ``keep_hoechst`` is True, the channel selection specifically (non-``*``)
    includes it, or it is the explicit ``nuclear_marker`` (which must stay).

    Example:
        For a map with ``HOECHST1``/``HOECHST2`` rows and no subset,
        ``apply_hoechst_drop(df, keep_hoechst=False)`` drops both.
    """
    extra = (nuclear_marker,) if nuclear_marker else ()
    return _drop_category(
        df,
        _is_hoechst,
        keep_all=keep_hoechst,
        channels=channels,
        extra_keep=extra,
    )
