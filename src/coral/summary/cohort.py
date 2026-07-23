"""Cohort-wide status aggregation + zarr↔state reconciliation.

Aggregates every slide's ``state.json`` under a job dir into per-slide rows
(one collapsed cell per stage) and reconciles the recorded state against
what physically exists in each ``.zarr`` (drift). Backs both
``coral status <dir>`` and the job ``summary.md``. Reads state via the
lightweight :func:`coral.slide.state.load_state` — it never opens the heavy
``CoralSlide``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from coral.slide.state import SlideState, TaskState, load_state

# Order the cohort table + summary rollup present stages in.
STAGES = ("ingest", "tissue", "patch", "cells", "extract")
_SCALAR_STAGES = frozenset({"ingest", "cells"})


@dataclass
class StageCell:
    """One stage's collapsed status for a slide.

    ``duration_s`` is set for scalar stages (ingest/tissue/cells);
    ``count`` is set for the dict stages (patch/extract — number of
    config/encoder entries). ``error`` carries the failure reason when
    ``status == "error"`` (the first failing sub-task for a dict stage).
    """

    status: str
    duration_s: float | None = None
    count: int | None = None
    error: str | None = None


@dataclass
class SlideStatus:
    """One slide's per-stage status + any zarr↔state drift."""

    name: str
    stages: dict[str, StageCell]
    drift: list[str] = field(default_factory=list)


def cohort_status(directory: Path) -> list[SlideStatus]:
    """Aggregate per-slide status across a cohort/job directory.

    Args:
        directory: A job dir holding ``<name>.zarr`` stores.

    Returns:
        One :class:`SlideStatus` per store, sorted by store name
        (empty if the path is not a directory or holds no stores).

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     cohort_status(Path(d))
        []
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    stores = sorted(
        p for p in directory.iterdir() if p.is_dir() and p.suffix == ".zarr"
    )
    rows: list[SlideStatus] = []
    for store in stores:
        try:
            state = load_state(store)
        except Exception as exc:  # noqa: BLE001
            # A corrupt/unreadable state.json must never crash the whole
            # cohort view — flag this store and keep going.
            rows.append(
                SlideStatus(
                    name=store.name,
                    stages={s: StageCell("unreadable") for s in STAGES},
                    drift=[f"state.json unreadable: {exc}"],
                )
            )
            continue
        rows.append(
            SlideStatus(
                name=store.name,
                stages={s: collapse_stage(state, s) for s in STAGES},
                drift=reconcile(store, state),
            )
        )
    return rows


def collapse_stage(state: SlideState, stage: str) -> StageCell:
    """Collapse a stage's task(s) into one status cell.

    Scalar stages (ingest/cells) map straight through with their
    duration. Dict stages (tissue/patch/extract) roll many sub-tasks into
    one status — ``error`` if any failed, else ``running`` if any is live,
    else ``completed`` only when all are, else ``pending`` — and carry the
    sub-task count plus the first failure reason.

    Args:
        state: The slide's loaded :class:`SlideState`.
        stage: One of :data:`STAGES`.

    Returns:
        The collapsed :class:`StageCell` for that stage.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coral.slide.state import default_state
        >>> with tempfile.TemporaryDirectory() as d:
        ...     state = default_state(Path(d) / "s.zarr")
        ...     collapse_stage(state, "tissue").status
        'pending'
    """
    tasks = state.tasks
    if stage in _SCALAR_STAGES:
        ts: TaskState = getattr(tasks, stage)
        return StageCell(
            status=ts.status, duration_s=ts.duration_s, error=ts.error
        )
    entries: dict[str, TaskState] = getattr(tasks, stage)
    if not entries:
        return StageCell(status="pending", count=0)
    statuses = [t.status for t in entries.values()]
    if "error" in statuses:
        rolled = "error"
    elif "running" in statuses:
        rolled = "running"
    elif all(s == "completed" for s in statuses):
        rolled = "completed"
    else:
        rolled = "pending"
    error = next(
        (t.error for t in entries.values() if t.status == "error"), None
    )
    return StageCell(status=rolled, count=len(entries), error=error)


def reconcile(store: Path, state: SlideState) -> list[str]:
    """Drift between recorded state and the store's actual contents.

    Two directions: (a) a completed task whose recorded output path is
    missing on disk; (b) a physical patches/features/labels group with no
    completed task.

    Args:
        store: The slide's ``.zarr`` directory.
        state: Its loaded :class:`SlideState`.

    Returns:
        Human-readable drift messages (empty when consistent).

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coral.slide.state import default_state
        >>> with tempfile.TemporaryDirectory() as d:
        ...     reconcile(Path(d), default_state(Path(d) / "s.zarr"))
        []
    """
    store = Path(store)
    tasks = state.tasks
    drift: list[str] = []

    # (a) completed task → its recorded outputs must exist.
    if tasks.ingest.status == "completed" and not (store / "0").exists():
        drift.append("ingest: completed but level-0 image '0' missing")
    if tasks.cells.status == "completed":
        drift += _missing_outputs(store, "cells", tasks.cells)
    for method, ts in tasks.tissue.items():
        if ts.status == "completed":
            drift += _missing_outputs(store, f"tissue[{method}]", ts)
    for slug, ts in tasks.patch.items():
        if ts.status == "completed":
            drift += _missing_outputs(store, f"patch[{slug}]", ts)
    for key, ts in tasks.extract.items():
        if ts.status == "completed":
            drift += _missing_outputs(store, f"extract[{key}]", ts)

    # (b) physical group present → its task must be completed.
    drift += _orphan_groups(store, state)
    return drift


def _missing_outputs(store: Path, label: str, ts: TaskState) -> list[str]:
    """Recorded output paths of a completed task that don't exist."""
    return [
        f"{label}: completed but '{rel}' missing"
        for rel in ts.outputs.values()
        if not (store / rel).exists()
    ]


def _orphan_groups(store: Path, state: SlideState) -> list[str]:
    """Physical groups present with no matching completed task."""
    out: list[str] = []
    tasks = state.tasks
    if (store / "cells" / "cell_mask").exists() and (
        tasks.cells.status != "completed"
    ):
        out.append("cells/cell_mask present but cells not completed")

    tissue_root = store / "tissue"
    if tissue_root.is_dir():
        for child in sorted(p for p in tissue_root.iterdir() if p.is_dir()):
            if not child.name.startswith("tissue_"):
                continue
            method = child.name[len("tissue_") :]
            if not method or not (child / "tissue.geojson").exists():
                continue
            ts = tasks.tissue.get(method)
            if ts is None or ts.status != "completed":
                out.append(
                    f"tissue/{child.name} present but tissue[{method}] "
                    "not completed"
                )

    patches = store / "patches"
    if patches.is_dir():
        for slug in sorted(p for p in patches.iterdir() if p.is_dir()):
            ts = tasks.patch.get(slug.name)
            if ts is None or ts.status != "completed":
                out.append(
                    f"patches/{slug.name} present but patch not completed"
                )

    features = store / "features"
    if features.is_dir():
        for slug in sorted(p for p in features.iterdir() if p.is_dir()):
            for enc in sorted(e for e in slug.iterdir() if e.is_dir()):
                for var in sorted(v for v in enc.iterdir() if v.is_dir()):
                    key = f"{slug.name}/{enc.name}/{var.name}"
                    ts = tasks.extract.get(key)
                    if ts is None or ts.status != "completed":
                        out.append(
                            f"features/{key} present but extract not completed"
                        )
    return out
