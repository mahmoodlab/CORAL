"""Workflow state for one slide — lives inside ``slide.zarr/state.json``.

State here is the "what's been done to this slide?" view. Per-feature
provenance (markers used, encoder version, timestamps) lives in the
Zarr attrs of the corresponding feature array, not here. State is
the workflow ledger; the array attrs are the per-output receipts.

The state file is per-slide because one slide's outputs travel
together; per-run information lives at the job level in
``<job_dir>/runs/<run_id>.json``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from coral.io.atomic import atomic_write_json
from coral.utils.hashing import stable_slide_hash
from coral.utils.run_context import current_run_id
from coral.utils.time import now_iso

STATE_FILENAME = "state.json"
SCHEMA_VERSION = 2

TaskStatus = Literal[
    "pending",
    "running",
    "completed",
    "skipped",
    "error",
]


class SlideRef(BaseModel):
    """Stable identifying info for a slide.

    ``image_path`` is the source image this slide was ingested from (a
    file, or a per-channel directory); ``store_path`` is the ``.zarr``
    store. (Older states carried ``abs_path`` here, which duplicated
    ``store_path``; schema v2 replaced it. ``extra="allow"`` keeps such a
    state loadable.)
    """

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    ext: str
    image_path: str = ""
    store_path: str


class SlideMeta(BaseModel):
    """Slide metadata snapshot. Forward-compatible (extra fields ok)."""

    model_config = ConfigDict(extra="allow")

    dimensions: tuple[int, int] | None = None
    n_markers: int | None = None
    mpp: float | None = None


class TaskState(BaseModel):
    """State of one workflow task for one slide.

    Forward-compatible — subclasses or future schema versions can
    attach extra fields (``n_patches``, ``markers_used`` etc.) without
    breaking older readers.
    """

    model_config = ConfigDict(extra="allow")

    status: TaskStatus = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        """Seconds from ``started_at`` to ``completed_at`` (derived).

        ``None`` until both stamps exist (i.e. while pending/running, or
        for an old state written before this field). Second-resolution
        (``now_iso`` truncates to seconds), so a sub-second task reads 0.0.

        Example:
            >>> TaskState(
            ...     started_at="2026-06-23T10:00:00-04:00",
            ...     completed_at="2026-06-23T10:00:42-04:00",
            ... ).duration_s
            42.0
        """
        if not (self.started_at and self.completed_at):
            return None
        return (
            datetime.fromisoformat(self.completed_at)
            - datetime.fromisoformat(self.started_at)
        ).total_seconds()


class TasksBlock(BaseModel):
    """All known tasks for one slide, one field per pipeline stage.

    ``ingest`` and ``cells`` are **single** tasks (a slide has one of
    each), so each is a ``TaskState`` (``pending`` until run). ``tissue``,
    ``patch``, and ``extract`` are **dicts keyed by config**, because a
    slide can hold many of each — one tissue mask per method
    (``tissue["otsu"]``, ``tissue["carta"]``), one patch set per
    (mpp, size), one extract set per (config, encoder, panel). They
    start empty (``{}``) and gain a keyed entry per run.
    """

    ingest: TaskState = Field(default_factory=TaskState)
    tissue: dict[str, TaskState] = Field(default_factory=dict)
    patch: dict[str, TaskState] = Field(default_factory=dict)
    cells: TaskState = Field(default_factory=TaskState)
    extract: dict[str, TaskState] = Field(default_factory=dict)


class SlideState(BaseModel):
    """Top-level slide workflow state. Schema v1.

    Forward-compatible: unknown fields under the sub-models are
    preserved when round-tripping through ``load_state`` /
    ``save_state``.
    """

    schema_version: int = SCHEMA_VERSION
    coral_version: str
    slide: SlideRef
    meta: SlideMeta = Field(default_factory=SlideMeta)
    tasks: TasksBlock = Field(default_factory=TasksBlock)
    last_run_id: str | None = None
    updated_at: str = ""


def _coral_version() -> str:
    """Return the installed CORAL version (deferred import for cycles)."""
    from coral import __version__

    return __version__


def default_state(
    slide_zarr_path: str | Path, *, image_path: str = ""
) -> SlideState:
    """Build a fresh ``SlideState`` for a new slide store.

    The slide ``id`` is a stable hash of the store's absolute path.

    Args:
        slide_zarr_path: Path to the slide's ``.zarr`` store
            (directory).
        image_path: Absolute path of the source image this store was
            ingested from (set by ingest; ``""`` when unknown).

    Returns:
        New ``SlideState`` with empty task block and current
        timestamp.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     store = Path(d) / "slide_001.zarr"
        ...     state = default_state(store)
        ...     state.slide.name
        'slide_001'
    """
    path = Path(slide_zarr_path)
    store_path = str(path.resolve())
    return SlideState(
        coral_version=_coral_version(),
        slide=SlideRef(
            id=stable_slide_hash(store_path),
            name=path.stem,
            ext=path.suffix,
            image_path=image_path,
            store_path=str(path),
        ),
        updated_at=now_iso(),
    )


def load_state(slide_zarr_path: str | Path) -> SlideState:
    """Load ``state.json`` from a slide store; return default if missing.

    Args:
        slide_zarr_path: Path to the slide's ``.zarr`` directory.

    Returns:
        Parsed ``SlideState``, or a fresh default if no state file
        exists at the path.

    Raises:
        pydantic.ValidationError: If ``state.json`` is malformed JSON
            or doesn't match the schema.
    """
    path = Path(slide_zarr_path)
    state_fp = path / STATE_FILENAME
    if not state_fp.exists():
        return default_state(path)
    data: Any = json.loads(state_fp.read_text())
    return SlideState.model_validate(data)


def save_state(
    slide_zarr_path: str | Path,
    state: SlideState,
) -> None:
    """Atomically write ``state.json`` into the slide store.

    **Side effect:** mutates ``state.updated_at`` on the caller's
    object before writing. This is intentional so the in-memory
    view stays in sync with the file, but callers passing a
    ``SlideState`` they intend to read again should be aware.

    Args:
        slide_zarr_path: Path to the slide's ``.zarr`` directory.
            The directory must already exist.
        state: ``SlideState`` to write. ``state.updated_at`` is
            overwritten in place.
    """
    path = Path(slide_zarr_path)
    state_fp = path / STATE_FILENAME
    state.updated_at = now_iso()
    run_id = current_run_id()
    if run_id is not None:
        state.last_run_id = run_id
    atomic_write_json(state_fp, state.model_dump(mode="json"))
