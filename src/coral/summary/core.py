"""Run-level provenance — per-run manifest + auto-`summary.md`.

Two artifacts live per ``job_dir``:

- ``<job_dir>/runs/<run_id>.json`` — one manifest per CLI invocation.
- ``<job_dir>/summary.md`` — auto-regenerated markdown, one section
  per run (newest first), listing tool / args / status / errors.

The ``summary.md`` opens with a **Cohort** block — a job-level rollup of
every ``<slide>.zarr/state.json`` (per-stage completion counts + any
zarr↔state drift), via :func:`coral.summary.cohort.cohort_status`.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from coral import __version__
from coral.io.atomic import atomic_write_json, atomic_write_text
from coral.utils.logging import add_run_logfile, remove_run_logfile
from coral.utils.run_context import reset_run_id, set_run_id
from coral.utils.time import now_iso

RUNS_DIRNAME = "runs"
SUMMARY_FILENAME = "summary.md"


def _runs_dir(job_dir: Path) -> Path:
    """Return + create ``<job_dir>/runs/``."""
    d = job_dir / RUNS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json_safe(value: Any) -> Any:  # noqa: ANN401
    """Coerce a value into something json.dumps can handle."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def start_run(
    job_dir: str | Path,
    *,
    tool: str,
    args: dict[str, Any] | None = None,
) -> str:
    """Create a per-run manifest under ``<job_dir>/runs/`` and return run_id.

    Called once per CLI invocation.

    Args:
        job_dir: Per-job output directory.
        tool: Tool name, e.g. ``"coral extract"``.
        args: CLI args (JSON-safe values are kept verbatim; others
            stringified).

    Returns:
        A new ``run_id`` (12 hex chars).

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     rid = start_run(Path(d), tool="demo", args={"x": 1})
        ...     len(rid)
        12
    """
    job_dir = Path(job_dir)
    run_id = uuid.uuid4().hex[:12]
    manifest = {
        "run_id": run_id,
        "tool": tool,
        "started_at": now_iso(),
        "finished_at": None,
        "status": "running",
        "error": None,
        "coral_version": __version__,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "job_dir": str(job_dir.resolve()),
        "args": _json_safe(args or {}),
    }
    fp = _runs_dir(job_dir) / f"{run_id}.json"
    atomic_write_json(fp, manifest)
    return run_id


def finalize_run(
    job_dir: str | Path,
    run_id: str,
    *,
    status: str,
    error: str | None = None,
) -> None:
    """Finalise a run manifest and regenerate ``summary.md``.

    Updates the run manifest with ``finished_at`` + final ``status``,
    then re-renders ``<job_dir>/summary.md`` from all manifests in
    ``<job_dir>/runs/``.

    Args:
        job_dir: Per-job output directory.
        run_id: Run ID returned by :func:`start_run`.
        status: Final status, typically ``"completed"`` or ``"error"``.
        error: Optional error message (when ``status == "error"``).
    """
    job_dir = Path(job_dir)
    manifest_fp = _runs_dir(job_dir) / f"{run_id}.json"
    try:
        manifest = json.loads(manifest_fp.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # Best-effort: create a minimal record so finalize doesn't
        # silently no-op on lost manifests.
        manifest = {"run_id": run_id, "tool": "unknown", "args": {}}

    manifest["finished_at"] = now_iso()
    manifest["status"] = status
    manifest["error"] = error
    manifest.setdefault("coral_version", __version__)
    atomic_write_json(manifest_fp, manifest)

    summary_md = _render_summary(job_dir)
    atomic_write_text(job_dir / SUMMARY_FILENAME, summary_md)


@contextmanager
def run_ledger(
    job_dir: str | Path,
    *,
    tool: str,
    args: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Bracket a command's work with a manifest + ``summary.md`` refresh.

    Opens a run (:func:`start_run`) on entry and finalises it on exit:
    ``completed`` on a clean exit; ``error`` (with the message) on an
    exception or a non-zero ``typer.Exit`` — then re-raises, so the CLI
    still exits as it would. Wrap only the real work (after input
    validation), so a bad-args exit doesn't leave an orphan manifest.

    Args:
        job_dir: The run's output dir (manifest + summary live here).
        tool: Tool name, e.g. ``"coral ingest"``.
        args: The command's resolved args (Paths/None are stringified
            by the manifest writer).

    Yields:
        Nothing — the wrapped block runs inside the ledger.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     with run_ledger(Path(d), tool="demo", args={"x": 1}):
        ...         pass
        ...     len(list((Path(d) / "runs").glob("*.json")))
        1
    """
    import typer

    rid = start_run(job_dir, tool=tool, args=args)
    handler = add_run_logfile(job_dir, rid)
    token = set_run_id(rid)
    try:
        yield
    except typer.Exit as exc:
        code = exc.exit_code
        finalize_run(
            job_dir,
            rid,
            status="completed" if code == 0 else "error",
            error=None if code == 0 else f"exited with code {code}",
        )
        raise
    except Exception as exc:
        finalize_run(job_dir, rid, status="error", error=str(exc))
        raise
    else:
        finalize_run(job_dir, rid, status="completed")
    finally:
        reset_run_id(token)
        remove_run_logfile(handler)


def _render_summary(job_dir: Path) -> str:
    """Render ``summary.md`` from all manifests in ``runs/``.

    Reads every ``runs/*.json`` on each call. Cost is O(n_runs) per
    finalize_run; acceptable for the typical n_runs < 100 case.
    A speed pass may be needed if very long-lived job dirs
    become a real concern.
    """
    runs = _load_all_runs(job_dir)
    # Newest run first (like `git log`), so the latest invocation is visible
    # without scrolling past the whole history. The Cohort block below still
    # leads — it is the current-state rollup, not part of the run log.
    runs.sort(key=lambda m: m.get("started_at", ""), reverse=True)

    header = (
        "# CORAL job summary\n\n"
        "Auto-generated on each `finalize_run` call. One section per"
        " run, newest first.\n\n"
        f"- Per-slide state lives in `<slide>.zarr/state.json`.\n"
        f"- Per-run manifests live in `{RUNS_DIRNAME}/*.json`.\n\n"
    )

    cohort = _render_cohort_block(job_dir)

    if not runs:
        return header + cohort + "_No runs recorded yet._\n"

    sections = [_render_run_section(m) for m in runs]
    return header + cohort + "\n".join(sections)


def _load_all_runs(job_dir: Path) -> list[dict[str, Any]]:
    """Load every manifest under ``runs/`` (best-effort)."""
    runs_dir = job_dir / RUNS_DIRNAME
    if not runs_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for fp in runs_dir.glob("*.json"):
        try:
            out.append(json.loads(fp.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _render_cohort_block(job_dir: Path) -> str:
    """Render the job-level Cohort rollup from every slide's state.

    Reads each ``<slide>.zarr/state.json`` under the job dir
    (:func:`coral.summary.cohort.cohort_status`) into per-stage completion
    counts + a zarr↔state drift summary. Returns ``""`` when the job dir
    holds no stores yet.
    """
    from coral.summary.cohort import STAGES, cohort_status

    rows = cohort_status(job_dir)
    if not rows:
        return ""
    n = len(rows)
    lines = [f"## Cohort — {n} slide(s)", ""]
    for stage in STAGES:
        done = sum(1 for r in rows if r.stages[stage].status == "completed")
        lines.append(f"- {stage}: {done}/{n} completed")
    drift = [f"  - {r.name}: {msg}" for r in rows for msg in r.drift]
    if drift:
        lines.append(f"- drift: {len(drift)} issue(s)")
        lines += drift
    else:
        lines.append("- drift: none")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_run_section(manifest: dict[str, Any]) -> str:
    """Render one run as a markdown section."""
    started = manifest.get("started_at", "unknown")
    finished = manifest.get("finished_at") or "unfinished"
    status = manifest.get("status", "unknown")
    tool = manifest.get("tool", "unknown")
    run_id = manifest.get("run_id", "?")
    coral_ver = manifest.get("coral_version", "unknown")
    error = manifest.get("error")

    args = manifest.get("args") or {}
    args_compact: dict[str, Any] = {}
    for k in sorted(args.keys()):
        v = args[k]
        if isinstance(v, str | int | float | bool) or v is None:
            args_compact[k] = v

    lines: list[str] = []
    lines.append(f"## Run {started} (coral {coral_ver}) — run_id={run_id}")
    lines.append(f"- Tool: `{tool}`")
    lines.append(f"- Status: **{status}**")
    lines.append(f"- Finished: `{finished}`")
    if args_compact:
        lines.append(f"- Args: `{json.dumps(args_compact, sort_keys=True)}`")
    if error:
        # Fence the error so markdown special characters (backticks,
        # asterisks, HTML tags in tracebacks) don't render as markup.
        lines.append("- Error:")
        lines.append("  ```")
        for err_line in str(error).splitlines() or [""]:
            lines.append(f"  {err_line}")
        lines.append("  ```")
    lines.append("")
    return "\n".join(lines)
