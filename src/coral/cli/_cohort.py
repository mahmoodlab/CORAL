"""Shared helpers for the cohort-capable CLI commands.

The downstream stages (``coral tissue`` / ``cell`` / ``patch`` /
``extract``) all process the ``.zarr`` stores found in ``--job-dir`` (the
output of ``coral ingest``) and, for the import paths, resolve a per-store
mask/label from a combined file-or-directory flag. This module is the one
home for that shared logic — discovering stores, mapping per-store import
artifacts by name, and the skip-completed check.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from coral.io.ingest import Resolution

__all__ = [
    "expand_cohort_zarr_dir",
    "primary_output",
    "reject_param_conflict",
    "resolve_artifact",
    "run_marker_guardrail",
    "stage_completed",
]

logger = logging.getLogger(__name__)


def run_marker_guardrail(
    job_dir: Path,
    stores: list[Path],
) -> Resolution:
    """Enforce the job dir's marker map before a stage runs, or exit.

    Validates ``marker_map.csv`` and syncs every store to it (see
    :func:`coral.markers.guardrail.enforce_marker_map`). A broken map or a
    kept set with no nuclear stain is rendered as clear, multi-line
    output and the command exits non-zero — nothing downstream runs. The
    nuclear stain is fixed at ingest and is never changed here.

    Args:
        job_dir: Directory holding ``marker_map.csv`` and the stores.
        stores: The ``.zarr`` stores to bring into sync.

    Returns:
        The resolved map, ``{raw_name: (marker, status, keep)}``.

    Raises:
        typer.Exit: With code 1 if the map cannot be applied.

    Example:
        Called by stage CLIs after expanding the job dir::

            from coral.cli._cohort import (
                expand_cohort_zarr_dir,
                run_marker_guardrail,
            )

            stores = expand_cohort_zarr_dir(job_dir)
            resolution = run_marker_guardrail(job_dir, stores)
    """
    from coral.cli._render import log_marker_map_error
    from coral.markers.guardrail import enforce_marker_map
    from coral.markers.marker_map import MarkerMapError
    from coral.utils import CoralError

    try:
        return enforce_marker_map(job_dir, stores)
    except MarkerMapError as exc:
        log_marker_map_error(exc)
        raise typer.Exit(code=1) from exc
    except CoralError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc


def expand_cohort_zarr_dir(directory: Path) -> list[Path]:
    """Return the ``.zarr`` slide stores directly inside ``directory``.

    Subdirectories whose name ends in ``.zarr`` are returned sorted by
    name; everything else (the job dir's ``marker_map.csv``, ``runs/``,
    ``summary.md`` …) is skipped silently.

    Args:
        directory: A job directory of ``.zarr`` slide stores.

    Returns:
        Sorted list of ``.zarr`` store paths.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     (Path(d) / "b.zarr").mkdir()
        ...     (Path(d) / "a.zarr").mkdir()
        ...     (Path(d) / "marker_map.csv").touch()
        ...     [p.name for p in expand_cohort_zarr_dir(Path(d))]
        ['a.zarr', 'b.zarr']
    """
    stores = [
        child
        for child in directory.iterdir()
        if child.is_dir() and child.suffix == ".zarr"
    ]
    return sorted(stores, key=lambda p: p.name)


def resolve_artifact(
    directory: Path, stem: str, *, ext: str | None = None
) -> Path | None:
    """Map a slide ``stem`` to its artifact file in ``directory``.

    With ``ext`` (e.g. ``".csv"`` for labels) the lookup is the exact
    ``{stem}{ext}``. Otherwise it globs ``{stem}.*`` (any extension) and
    requires a single match. A missing file returns ``None`` so the
    caller can soft-skip that artifact for that slide; an ambiguous stem
    (more than one extension) is an error.

    Args:
        directory: Folder of per-slide artifact files.
        stem: The slide name without ``.zarr`` (e.g. ``"A-1"``).
        ext: Fixed extension to require (incl. dot), or ``None`` to glob.

    Returns:
        The matching file, or ``None`` if absent.

    Raises:
        typer.BadParameter: If ``ext`` is ``None`` and the stem matches
            more than one file (ambiguous extension).

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "A-1.tiff").write_bytes(b"x")
        ...     resolve_artifact(Path(d), "A-1").name
        'A-1.tiff'
    """
    if ext is not None:
        candidate = directory / f"{stem}{ext}"
        return candidate if candidate.is_file() else None
    matches = sorted(p for p in directory.glob(f"{stem}.*") if p.is_file())
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        msg = (
            f"ambiguous artifact for {stem!r} in {directory}: {names}. "
            f"Keep one file per slide stem."
        )
        raise typer.BadParameter(msg)
    return matches[0] if matches else None


def reject_param_conflict(
    *, mask_supplied: bool, params: Mapping[str, object], mode: str
) -> None:
    """Reject compute-mode params when a user mask is being imported.

    Args:
        mask_supplied: Whether a user mask (file or dir) was given — i.e.
            the command is in *import* mode.
        params: ``flag name -> value`` for the compute-only params (those
            with no default, so a non-``None`` value means the user set
            it).
        mode: The compute mode the params belong to, for the message
            (e.g. ``"Cellpose segmentation"``).

    Raises:
        typer.BadParameter: If ``mask_supplied`` and any param is set.

    Example:
        >>> reject_param_conflict(
        ...     mask_supplied=False,
        ...     params={"--diameter": None},
        ...     mode="Cellpose segmentation",
        ... )
    """
    on = [k for k, v in params.items() if v is not None]
    if mask_supplied and on:
        raise typer.BadParameter(
            f"{on[0]} applies to {mode}; not valid when importing a mask."
        )


def primary_output(stage: str, slug: str | None) -> str | None:
    """Store-relative path to the load-bearing artifact of ``stage``.

    This is the single output whose presence proves the stage actually
    produced results — the file/array a user deletes to force a rerun.
    :func:`stage_completed` requires it on disk (on top of a ``completed``
    status), so deleting the output folder re-triggers work.

    ``None`` means "gate on status only": ``extract`` does its own
    on-disk check in ``encode_features``, and a per-key stage called
    without a ``slug`` has no resolvable primary.

    Args:
        stage: ``ingest``/``cells``/``tissue``/``patch``/``extract``.
        slug: The tissue method / patch config slug (per-key stages).

    Returns:
        The store-relative path, or ``None`` for status-only stages.

    Example:
        >>> primary_output("patch", "0.5mpp_256px")
        'patches/0.5mpp_256px/coords'
        >>> primary_output("extract", "kronos2") is None
        True
    """
    if stage == "ingest":
        return "0"
    if stage == "cells":
        return "cells/cell_mask"
    if stage == "tissue":
        if slug is None:
            return None
        from coral.tissue.paths import tissue_rel

        return f"{tissue_rel(slug)}/tissue.geojson"
    if stage == "patch":
        return f"patches/{slug}/coords" if slug is not None else None
    return None


def stage_completed(
    item: Path, stage: str, *, slug: str | None = None
) -> bool:
    """Whether ``stage`` is already done for the slide at ``item``.

    The gate is *reconciled*: a stage counts as done only when its
    ``state.json`` status is ``completed`` **and** its primary output
    (see :func:`primary_output`) still exists on disk. Deleting the
    output folder therefore re-triggers a rerun even though the status
    still reads ``completed``; a folder present but not ``completed`` (a
    crashed/partial write) also reruns. ``extract`` gates on status only.

    Centralizes the skip-completed check shared by ``coral tissue`` /
    ``patch`` / ``cell``. Scalar stages (``cells``/``ingest``) read a
    single task; per-key stages (``tissue``/``patch``/``extract``) read
    the ``slug``'s slot (tissue key = method name). A missing store or
    missing slot is "not completed".

    Args:
        item: The slide ``.zarr`` store path.
        stage: ``tissue``/``cells``/``ingest``/``patch``/``extract``.
        slug: The tissue method / patch / extract key (for per-key
            stages).

    Returns:
        ``True`` iff the store exists, the stage's task is ``completed``,
        and its primary output is present on disk.

    Raises:
        ValueError: On an unknown ``stage``.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     stage_completed(Path(d) / "missing.zarr", "tissue")
        False
    """
    if not item.exists():
        return False
    from coral.slide.state import TaskState, load_state

    tasks = load_state(item).tasks
    if stage in {"cells", "ingest"}:
        slot = getattr(tasks, stage)
    elif stage in {"tissue", "patch", "extract"}:
        slot = getattr(tasks, stage).get(slug or "", TaskState())
    else:
        raise ValueError(f"unknown stage {stage!r}")
    if slot.status != "completed":
        return False
    primary = primary_output(stage, slug)
    return primary is None or (item / primary).exists()
