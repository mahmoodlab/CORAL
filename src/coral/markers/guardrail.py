"""Enforce the marker map on a job dir before a stage runs.

The editable ``marker_map.csv`` is the single source of truth for how
each raw channel name maps to a canonical marker. Every stage command
runs :func:`enforce_marker_map` first, so an edit to the CSV takes effect
automatically — there is no separate "apply" step to remember. The
analysis panel (which channels are kept) is decided once at ingest and
frozen in each store's ``.zattrs``; the guardrail never moves it.

The guardrail is loud and strict. It validates the map (a blank or
invalid mapping is a hard error that names the offending rows), promotes
any reviewed row whose name is now in the registry, and re-syncs each
store's marker names to match the map — writing only the stores that
actually drifted (keep is left untouched), and warning when a change may
have invalidated a stage that already ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import zarr

from coral.markers.marker_map import (
    NOVEL_TOKEN,
    RESOLVED_TOKEN,
    REVIEW_TOKEN,
    MarkerMapError,
    promote_reviewed,
    read_marker_map,
    resolve_marker_map,
    validate_and_flag,
    validate_marker_map,
    write_marker_map,
)
from coral.markers.normalize import clean_marker_name
from coral.markers.registry import canonical_key_index
from coral.tissue.infer import infer_dapi_index
from coral.utils import CoralError

if TYPE_CHECKING:
    from coral.io.ingest import Resolution

logger = logging.getLogger(__name__)

# Stages whose outputs depend on the marker set / nuclear stain. If the
# map changes a store after one of these completed, its output may be
# stale and the user is warned (never silently invalidated).
_DOWNSTREAM_STAGES = ("tissue", "cells", "patch", "extract")


def enforce_marker_map(
    job_dir: str | Path,
    stores: list[Path],
) -> Resolution:
    """Validate the marker map and sync every store to match it.

    Reads ``marker_map.csv`` from ``job_dir``, validates it, promotes any
    reviewed row whose mapped name is a registry marker, then brings each
    store's recorded channels into agreement with the map. Only stores
    that differ are rewritten (no pixel data is read); a store that
    already matches is left untouched. Every decision is logged. The
    nuclear stain is fixed at ingest: each store keeps its current stain
    (re-inferred only if that channel is no longer kept), never overridden.

    Args:
        job_dir: Directory holding ``marker_map.csv`` and the stores.
        stores: The ``.zarr`` stores to bring into sync.

    Returns:
        The resolved map, ``{raw_name: (marker, status, keep)}``.

    Raises:
        MarkerMapError: If the map has a blank or invalid mapping — the
            offending rows are carried on the exception for display.
        CoralError: If the map is missing, every marker is excluded, or no
            kept channel is a nuclear stain.

    Example:
        Before segmenting, a command syncs its stores to the edited map::

            resolution = enforce_marker_map(job_dir, stores)
    """
    job_dir = Path(job_dir)
    logger.info("Checking marker map (job dir: %s) ...", job_dir)

    map_df = read_marker_map(job_dir)
    if map_df is None:
        # No map to apply — each slide's stored marker names stand. This
        # happens only outside the normal flow (ingest always writes one).
        logger.warning(
            "no marker_map.csv in %s — using each slide's stored marker "
            "names (run 'coral ingest' to regenerate the map).",
            job_dir,
        )
        return {}

    valid_names = set(canonical_key_index().values())
    # Validate; a mapping the user forced but got wrong is reset to REVIEW
    # in the CSV before the error is raised, so it never stays stuck.
    validate_and_flag(map_df, valid_names, job_dir)

    map_df, promoted = promote_reviewed(map_df, valid_names)
    if promoted:
        write_marker_map(map_df, job_dir)  # keep the CSV consistent
        logger.info(
            "  promoted %d marker(s) to RESOLVED "
            "(a registry name was supplied): %s",
            len(promoted),
            ", ".join(promoted),
        )

    resolution = resolve_marker_map(map_df)
    _log_resolution(resolution)
    _check_panel_covered(stores, resolution)
    _check_nuclear(stores)
    _sync_all(stores, resolution)
    return resolution


@dataclass
class MarkerMapReport:
    """A read-only summary of a job dir's marker map and store sync.

    Produced by :func:`inspect_marker_map` for display; nothing is
    written or changed.

    Attributes:
        present: Whether ``marker_map.csv`` exists in the job dir.
        problems: Validation errors (unresolved / invalid rows), or
            ``None`` when the map is applyable.
        n_markers: Total markers in the map.
        n_resolved: Markers mapped to a registry name.
        n_novel: Markers marked novel.
        n_review: Markers still needing review.
        review_names: Original names of the rows still under review.
        drifted: Store names whose recorded markers differ from the map.
        in_sync: Number of stores already matching the map.
        stale: Drifted store name → downstream stages already completed
            (whose outputs a sync may invalidate).
    """

    present: bool
    problems: MarkerMapError | None = None
    n_markers: int = 0
    n_resolved: int = 0
    n_novel: int = 0
    n_review: int = 0
    review_names: list[str] = field(default_factory=list)
    drifted: list[str] = field(default_factory=list)
    in_sync: int = 0
    stale: dict[str, list[str]] = field(default_factory=dict)


def inspect_marker_map(
    job_dir: str | Path, stores: list[Path]
) -> MarkerMapReport:
    """Summarise a job dir's marker map and store sync — read-only.

    Validates the map (reporting problems, never raising), counts
    resolved / novel / review markers, and checks each store for drift
    and for completed stages a sync would make stale. Writes nothing.

    Args:
        job_dir: Directory holding ``marker_map.csv`` and the stores.
        stores: The ``.zarr`` stores to check for drift.

    Returns:
        A :class:`MarkerMapReport`.

    Example:
        Back the marker-map section of a status view::

            report = inspect_marker_map(job_dir, stores)
    """
    map_df = read_marker_map(job_dir)
    if map_df is None:
        return MarkerMapReport(present=False)

    problems: MarkerMapError | None = None
    try:
        validate_marker_map(map_df, set(canonical_key_index().values()))
    except MarkerMapError as exc:
        problems = exc

    resolution = resolve_marker_map(map_df)
    drifted: list[str] = []
    stale: dict[str, list[str]] = {}
    for store in stores:
        if _plan_sync(store, resolution).changed:
            drifted.append(store.name)
            done = _completed_downstream(store)
            if done:
                stale[store.name] = done

    return MarkerMapReport(
        present=True,
        problems=problems,
        n_markers=len(resolution),
        n_resolved=sum(
            1 for _, lvl, _ in resolution.values() if lvl == RESOLVED_TOKEN
        ),
        n_novel=sum(
            1 for _, lvl, _ in resolution.values() if lvl == NOVEL_TOKEN
        ),
        n_review=sum(
            1 for _, lvl, _ in resolution.values() if lvl == REVIEW_TOKEN
        ),
        review_names=[
            orig
            for orig, (_, lvl, _) in resolution.items()
            if lvl == REVIEW_TOKEN
        ],
        drifted=drifted,
        in_sync=len(stores) - len(drifted),
        stale=stale,
    )


def _log_resolution(resolution: Resolution) -> None:
    """Log the kept-panel marker counts (resolved vs novel)."""
    n_resolved = sum(
        1 for _, lvl, _ in resolution.values() if lvl == RESOLVED_TOKEN
    )
    n_novel = sum(1 for _, lvl, _ in resolution.values() if lvl == NOVEL_TOKEN)
    logger.info(
        "  analysis panel: %d marker(s) (%d resolved, %d novel)",
        len(resolution),
        n_resolved,
        n_novel,
    )


def _check_panel_covered(stores: list[Path], resolution: Resolution) -> None:
    """Fail loudly if a kept channel has no row in ``marker_map.csv``.

    The kept-only CSV is the panel's name list; a kept channel whose raw is
    absent from it (e.g. its row was hand-deleted) would otherwise slip past
    the blank-``REVIEW`` gate and analyse an unnamed channel. Excluded / QC
    channels are ``keep=False`` and legitimately absent, so they are exempt.
    """
    for store in stores:
        channels = list(
            zarr.open_group(str(store), mode="r").attrs.get("channels", [])
        )
        orphans = [
            str(c.get("raw") or c.get("marker") or f"channel_{i}")
            for i, c in enumerate(channels)
            if c.get("keep", True) and c.get("raw") not in resolution
        ]
        if orphans:
            raise CoralError(
                f"{store.name}: {len(orphans)} kept channel(s) missing from "
                f"marker_map.csv: {', '.join(orphans)} — restore their "
                "row(s), or re-ingest into a fresh --job-dir."
            )


def _check_nuclear(stores: list[Path]) -> None:
    """Fail loudly if a store's frozen kept panel has no nuclear stain.

    The panel (``keep``) is frozen in each store's ``.zattrs`` at ingest;
    this is a cheap per-stage guard that a store still has a nuclear stain
    to anchor tissue / cell / feature extraction.
    """
    for store in stores:
        root = zarr.open_group(str(store), mode="r")
        channels = list(root.attrs.get("channels", []))
        kept = [
            str(c["marker"])
            for c in channels
            if c.get("keep", True) and c.get("marker")
        ]
        if not kept:
            raise CoralError(
                f"{store.name}: every marker is excluded — re-ingest with a "
                "panel that keeps at least one marker."
            )
        if infer_dapi_index(kept) is None:
            raise CoralError(
                f"{store.name}: no nuclear stain among the kept markers — "
                "re-ingest keeping a nuclear marker."
            )


def _sync_all(
    stores: list[Path],
    resolution: Resolution,
) -> None:
    """Sync every store to the map, logging each change and the summary."""
    n_synced = 0
    for store in stores:
        if _sync_store(store, resolution):
            n_synced += 1
    if n_synced == 0:
        logger.info("  all %d store(s) already in sync", len(stores))
    else:
        logger.info(
            "  synced %d store(s) from marker_map.csv, %d already in sync",
            n_synced,
            len(stores) - n_synced,
        )


@dataclass
class _SyncPlan:
    """How a store would change under the map — computed without writing."""

    changed: bool
    eff_nuclear: str | None
    raws: list[str | None]
    cur_markers: list[str]
    exp_markers: list[str]
    exp_nuclear: str | None


def _plan_sync(
    store: Path,
    resolution: Resolution,
) -> _SyncPlan:
    """Read a store and work out how the map would change it — no write.

    The single definition of "drift": the store's recorded markers and
    nuclear stain versus what the map resolves to. ``keep`` (the analysis
    panel) is frozen at ingest and never part of drift. Used both to decide
    whether to rewrite a store and to report status read-only.
    """
    root = zarr.open_group(str(store), mode="r")
    channels = list(root.attrs.get("channels", []))
    raws = [c.get("raw") for c in channels]
    cur_markers = [c.get("marker") for c in channels]
    cur_keep = [bool(c.get("keep", True)) for c in channels]
    cur_nuclear = root.attrs.get("nuclear_channel")

    exp_markers = _expected(channels, resolution)
    cur_nuc_idx = next(
        (i for i, m in enumerate(cur_markers) if m == cur_nuclear), None
    )
    keep_nuclear = cur_nuc_idx is not None and cur_keep[cur_nuc_idx]
    eff_nuclear = _effective_nuclear(exp_markers, cur_nuc_idx, keep_nuclear)
    exp_nuclear = _expected_nuclear(exp_markers, cur_keep, eff_nuclear)
    changed = not (exp_markers == cur_markers and exp_nuclear == cur_nuclear)
    return _SyncPlan(
        changed=changed,
        eff_nuclear=eff_nuclear,
        raws=raws,
        cur_markers=cur_markers,
        exp_markers=exp_markers,
        exp_nuclear=exp_nuclear,
    )


def _sync_store(
    store: Path,
    resolution: Resolution,
) -> bool:
    """Bring one store into agreement with the map; return True if changed.

    When the store already matches the map it is left untouched (no
    write). When it differs, the names are re-applied (no pixel re-read)
    and the change — plus any stale-stage warning — is logged.
    """
    plan = _plan_sync(store, resolution)
    if not plan.changed:
        return False

    stale = _completed_downstream(store)
    try:
        from coral.io.ingest import apply_marker_names

        apply_marker_names(store, resolution, nuclear_marker=plan.eff_nuclear)
    except Exception as exc:  # noqa: BLE001 — re-raised with store context
        raise CoralError(
            f"{store.name}: could not apply marker_map.csv — {exc}"
        ) from exc

    _log_change(
        store,
        raws=plan.raws,
        cur_markers=plan.cur_markers,
        new_markers=plan.exp_markers,
        new_nuclear=plan.exp_nuclear,
        resolution=resolution,
    )
    if stale:
        logger.warning(
            "  %s: marker_map changed but %s already completed — those "
            "output(s) may be stale; delete them or use a fresh --job-dir "
            "to refresh.",
            store.name,
            ", ".join(stale),
        )
    return True


def _expected(channels: list[dict], resolution: Resolution) -> list[str]:
    """Marker names a store would have under the map — no write.

    Mirrors how names are applied to a store: a raw name found in the
    (kept-only) map takes its resolved marker; a channel absent from the
    map is an excluded/QC channel whose stored name is frozen and preserved
    (never renamed). ``keep`` is never recomputed — the panel is frozen at
    ingest.
    """
    markers: list[str] = []
    for i, ch in enumerate(channels):
        raw = ch.get("raw")
        if raw is not None and raw in resolution:
            markers.append(clean_marker_name(resolution[raw][0]))
        else:
            markers.append(ch.get("marker") or f"channel_{i}")
    return markers


def _effective_nuclear(
    exp_markers: list[str],
    cur_nuc_idx: int | None,
    keep_nuclear: bool,
) -> str | None:
    """The nuclear name to apply: preserved or re-inferred.

    The store's current nuclear channel is preserved (by its new name at
    the same position), unless it is no longer kept — then ``None`` lets it
    be re-inferred. The nuclear stain is fixed at ingest, so there is no
    per-run override.
    """
    if cur_nuc_idx is not None and keep_nuclear:
        return exp_markers[cur_nuc_idx]
    return None


def _expected_nuclear(
    exp_markers: list[str],
    keep: list[bool],
    eff_nuclear: str | None,
) -> str | None:
    """The nuclear stain name a store would have — predicted, no write.

    Matches the applied behaviour: a preserved name resolves to that
    channel; otherwise the stain is inferred from the kept markers
    (``keep`` is the store's frozen panel), then from all markers.
    """
    if eff_nuclear is not None:
        return eff_nuclear
    kept_idxs = [i for i, k in enumerate(keep) if k]
    rel = infer_dapi_index([exp_markers[i] for i in kept_idxs])
    if rel is not None:
        return exp_markers[kept_idxs[rel]]
    idx = infer_dapi_index(exp_markers)
    return exp_markers[idx] if idx is not None else None


def _completed_downstream(store: Path) -> list[str]:
    """Downstream stages already completed for a store, in pipeline order."""
    from coral.slide.state import load_state

    tasks = load_state(store).tasks
    done: list[str] = []
    for stage in _DOWNSTREAM_STAGES:
        slot = getattr(tasks, stage)
        if isinstance(slot, dict):
            if any(t.status == "completed" for t in slot.values()):
                done.append(stage)
        elif slot.status == "completed":
            done.append(stage)
    return done


def _log_change(
    store: Path,
    *,
    raws: list[str | None],
    cur_markers: list[str],
    new_markers: list[str],
    new_nuclear: str | None,
    resolution: Resolution,
) -> None:
    """Log one store's applied change: renames, novels, nuclear.

    Names are shown exactly as the user wrote them in the map (not the
    store's cleaned form), and each rename is classed as a registry rename
    or a novel marker so the line reads at a glance. ``keep`` is frozen at
    ingest, so the panel never changes here.
    """

    def exact(raw: str | None, fallback: str) -> str:
        if raw and raw in resolution and resolution[raw][0]:
            return resolution[raw][0]
        return fallback

    renamed: list[str] = []
    novel: list[str] = []
    for raw, old, new in zip(raws, cur_markers, new_markers, strict=False):
        if old == new:
            continue
        level = resolution.get(raw or "", ("", "", True))[1]
        target = novel if level == NOVEL_TOKEN else renamed
        target.append(f"{raw} --> {exact(raw, new)}")

    parts: list[str] = []
    if renamed:
        parts.append(f"renamed {len(renamed)}: {', '.join(renamed)}")
    if novel:
        parts.append(f"marked {len(novel)} novel: {', '.join(novel)}")
    logger.info(
        "  %s: %d markers · nuclear=%s · %s",
        store.name,
        len(new_markers),
        new_nuclear,
        "; ".join(parts) if parts else "nuclear updated",
    )
