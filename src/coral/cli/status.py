"""``coral status`` — live per-step status of a slide or a cohort.

``coral status --job-dir <slide.zarr>`` prints one slide's per-stage
status, in pipeline order, read from its ``state.json``. ``coral status
--job-dir <job_dir>`` prints a cohort table — one row per ``.zarr``
store × stage — plus a
``drift`` column reconciling each store's recorded state against what is
physically present. Both views read state directly (no heavy image open)
and never crash on a corrupt ``state.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from coral.cli._cohort import expand_cohort_zarr_dir
from coral.markers.marker_map import is_qc_channel
from coral.slide.state import SlideState, load_state
from coral.summary.cohort import (
    STAGES,
    SlideStatus,
    StageCell,
    cohort_status,
    collapse_stage,
    reconcile,
)

if TYPE_CHECKING:
    from coral.markers.guardrail import MarkerMapReport

# Each status → (glyph, Rich colour). One source of truth for both views.
_STYLE: dict[str, tuple[str, str]] = {
    "completed": ("✓", "green"),
    "running": ("●", "yellow"),
    "error": ("✗", "red"),
    "pending": ("–", "dim"),
    "skipped": ("·", "cyan"),
    "unreadable": ("?", "red"),
}
# Sub-tasks of patch/extract are dict stages — expanded under their row.
_DICT_STAGES = ("patch", "extract")
_LEGEND = "   ".join(
    f"{_STYLE[s][0]} {s}"
    for s in ("completed", "running", "error", "pending", "skipped")
)


def _glyph(status: str) -> str:
    """The status glyph wrapped in its Rich colour markup."""
    char, colour = _STYLE.get(status, ("?", "red"))
    return f"[{colour}]{char}[/{colour}]"


def status(
    job_dir: Path = typer.Option(
        ...,
        "--job-dir",
        help="Job dir from `coral ingest` (cohort table), or a single "
        ".zarr store (that slide's per-stage view).",
    ),
) -> None:
    """Print live status for one slide, or a whole cohort.

    A ``.zarr`` store path prints that slide's per-stage status; a
    directory prints the cohort table (slide × stage + drift). Read it
    any time — mid-run or after a crash.

    Args:
        job_dir: Job directory from ``coral ingest``, or a single ``.zarr``
            store path for that slide's per-stage view.

    Example:
        Show cohort status after a partial run::

            coral status --job-dir ./processed
    """
    if not job_dir.exists():
        typer.echo(f"{job_dir} does not exist.", err=True)
        raise typer.Exit(code=1)
    if job_dir.suffix == ".zarr" or (job_dir / "0").exists():
        _print_single(job_dir)
    elif job_dir.is_dir():
        _print_cohort(job_dir)
    else:
        typer.echo(f"{job_dir} is not a slide store or a job dir.", err=True)
        raise typer.Exit(code=1)


def _print_single(store: Path) -> None:
    """Print one slide's per-stage status in pipeline order."""
    console = Console()
    try:
        state = load_state(store)
    except Exception as exc:  # noqa: BLE001 — never dump a traceback
        typer.echo(f"{store.name}: state.json is unreadable — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    console.print(_header(store.name, state))
    for stage in STAGES:
        cell = collapse_stage(state, stage)
        expanded = stage in _DICT_STAGES
        console.print(_stage_line(stage, cell, show_error=not expanded))
        if expanded:
            for key, ts in getattr(state.tasks, stage).items():
                console.print(_subtask_line(key, ts))

    drift = reconcile(store, state)
    done = sum(
        1 for s in STAGES if collapse_stage(state, s).status == "completed"
    )
    console.print()
    console.print(_single_summary(done, drift, state.last_run_id))
    for msg in drift:
        console.print(f"  [yellow]⚠[/yellow] {msg}")

    panel = _read_panel(store)
    if panel is not None:
        _render_panel(console, *panel)
    _render_features(
        console, [(store.name, *s) for s in _read_feature_sets(store)]
    )


def _header(name: str, state: SlideState) -> str:
    """Slide name + a one-line context from its recorded metadata."""
    m = state.meta
    bits: list[str] = []
    if m.n_markers is not None:
        bits.append(f"{m.n_markers} markers")
    if m.dimensions is not None:
        bits.append(f"{m.dimensions[0]}×{m.dimensions[1]} px")
    if m.mpp is not None:
        bits.append(f"{m.mpp:g} µm/px")
    context = f"  [dim]({' · '.join(bits)})[/dim]" if bits else ""
    return f"[bold]{name}[/bold]{context}"


def _stage_line(
    stage: str, cell: StageCell, *, show_error: bool = True
) -> str:
    """One stage's roll-up line: glyph, name, duration/count.

    ``show_error`` appends the failure reason — on for scalar stages
    (the line is all there is), off for patch/extract whose sub-task
    lines carry their own reasons.
    """
    detail = ""
    if cell.duration_s is not None:
        detail = f"  [dim]{cell.duration_s:.1f}s[/dim]"
    elif cell.count:
        detail = f"  [dim]{cell.count} set(s)[/dim]"
    if show_error and cell.status == "error" and cell.error:
        detail += f"  [red]{cell.error}[/red]"
    return f"  {_glyph(cell.status)} {stage:<8}{detail}".rstrip()


def _subtask_line(key: str, ts: object) -> str:
    """An indented line for one patch/extract sub-task (no redundancy)."""
    status_str = getattr(ts, "status", "pending")
    dur = getattr(ts, "duration_s", None)
    err = getattr(ts, "error", None)
    if dur is not None:
        detail = f"[dim]{dur:.1f}s[/dim]"
    elif status_str == "error" and err:
        detail = f"[red]{err}[/red]"
    else:
        detail = f"[dim]{status_str}[/dim]"
    return f"      {_glyph(status_str)} {key}  {detail}"


def _single_summary(
    done: int, drift: list[str], last_run_id: str | None
) -> str:
    """The one-line tail: stages done, drift, last run."""
    parts = [f"{done}/{len(STAGES)} stages done"]
    if drift:
        parts.append(f"[yellow]drift: {len(drift)} issue(s)[/yellow]")
    if last_run_id:
        parts.append(f"[dim]last run {last_run_id}[/dim]")
    return " · ".join(parts)


def _read_panel(store: Path) -> tuple[list[dict], str | None] | None:
    """Read a store's frozen channel panel from ``.zattrs``.

    Returns ``(channels, nuclear_name)`` — the per-channel dicts (marker,
    match, keep) and the nuclear stain name — or ``None`` when the store has
    no channels or can't be read (status never crashes on a bad store).
    """
    try:
        import zarr

        root = zarr.open_group(str(store), mode="r")
        channels = list(root.attrs.get("channels", []))
    except Exception:  # noqa: BLE001 — status tolerates an unreadable store
        return None
    if not channels:
        return None
    return channels, root.attrs.get("nuclear_channel")


def _render_panel(
    console: Console, channels: list[dict], nuclear: str | None
) -> None:
    """Print the frozen analysis panel: every channel's marker/status/keep.

    The panel is decided at ingest and stored in ``.zattrs`` — this renders
    it read-only, including the excluded / QC channels the editable
    ``marker_map.csv`` no longer lists.
    """
    console.print()
    console.print("[bold]Panel[/bold]  [dim](frozen at ingest)[/dim]")
    table = Table()
    table.add_column("marker")
    table.add_column("status")
    table.add_column("keep", justify="center")
    table.add_column("note")
    for ch in channels:
        # Resolved marker; fall back to the raw name for an unresolved
        # (REVIEW) channel so every row stays identifiable.
        marker = str(ch.get("marker") or ch.get("raw") or "—")
        kept = bool(ch.get("keep", True))
        keep_cell = "[green]✓[/green]" if kept else "[red]✗[/red]"
        if kept:
            note = "[dim]nuclear[/dim]" if marker == nuclear else ""
        else:
            # Why it was excluded: a QC-type channel (blank/empty/Hoechst)
            # is auto-dropped; any other excluded marker was subset out.
            ident = str(ch.get("raw") or ch.get("marker") or "")
            reason = "QC" if is_qc_channel(ident) else "user"
            note = f"[dim]{reason}[/dim]"
        table.add_row(marker, str(ch.get("match") or "—"), keep_cell, note)
    console.print(table)


def _subdirs(d: Path) -> list[Path]:
    """Sorted sub-directories of ``d`` (a zarr group's members)."""
    return sorted(p for p in d.iterdir() if p.is_dir())


def _read_feature_sets(
    store: Path,
) -> list[tuple[str, str, str, tuple[int, ...]]]:
    """Enumerate a store's stored feature sets.

    Returns ``(patch_slug, extractor, variant, shape)`` per stored output
    array — read from the ``features/<slug>/<encoder>/<variant>`` tree
    (the variant path *is* the array) and its ``.zarray`` header only (no
    feature data). Tolerates a missing/malformed tree; status never
    crashes on a bad store.
    """
    import json

    feats = store / "features"
    if not feats.is_dir():
        return []
    out: list[tuple[str, str, str, tuple[int, ...]]] = []
    for slug in _subdirs(feats):
        for enc in _subdirs(slug):
            for variant in _subdirs(enc):
                meta = variant / ".zarray"
                if not meta.is_file():
                    continue
                try:
                    raw = json.loads(meta.read_text())["shape"]
                except Exception:  # noqa: BLE001 — skip a bad array
                    continue
                shape = tuple(int(d) for d in raw)
                out.append((slug.name, enc.name, variant.name, shape))
    return out


def _render_features(
    console: Console,
    named: list[tuple[str, str, str, str, tuple[int, ...]]],
) -> None:
    """Print the stored feature sets: extractor · patch set · variant · shape.

    ``named`` holds ``(slide, slug, extractor, variant, shape)`` rows; the
    ``slide`` column shows only for a multi-slide cohort. Empty → nothing.
    """
    if not named:
        return
    console.print()
    console.print("[bold]Features[/bold]  [dim](stored embeddings)[/dim]")
    multi = len({row[0] for row in named}) > 1
    table = Table()
    if multi:
        table.add_column("slide")
    table.add_column("extractor")
    table.add_column("patch set")
    table.add_column("variant")
    table.add_column("shape", justify="right")
    for slide, slug, enc, variant, shape in named:
        cells = [enc, slug, variant, " × ".join(str(d) for d in shape)]
        if multi:
            cells.insert(0, slide)
        table.add_row(*cells)
    console.print(table)


def _print_cohort(job_dir: Path) -> None:
    """Render the cohort status table + drift/error details under it."""
    console = Console()
    rows = cohort_status(job_dir)
    if not rows:
        typer.echo(f"no .zarr stores in {job_dir}", err=True)
        raise typer.Exit(code=1)

    console.print(f"[bold]coral status[/bold]  [dim]{job_dir}[/dim]")
    table = Table()
    table.add_column("slide")
    for stage in STAGES:
        table.add_column(stage, justify="center")
    table.add_column("drift", justify="center")
    for row in rows:
        cells = [row.name, *(_cohort_cell(row.stages[s]) for s in STAGES)]
        cells.append(
            f"[yellow]{len(row.drift)}[/yellow]"
            if row.drift
            else "[dim]–[/dim]"
        )
        table.add_row(*cells)
    from coral.markers.guardrail import inspect_marker_map

    stores = expand_cohort_zarr_dir(job_dir)
    report = inspect_marker_map(job_dir, stores)
    console.print(table)
    console.print(f"[dim]{_LEGEND}[/dim]")
    console.print(_cohort_summary(rows, report))
    _print_issues(console, rows)
    _print_marker_map(console, report)
    # The panel is cohort-wide (frozen at ingest); show it from a
    # representative store. A per-store keep mismatch surfaces as drift.
    if stores:
        panel = _read_panel(stores[0])
        if panel is not None:
            _render_panel(console, *panel)
    named = [(st.name, *s) for st in stores for s in _read_feature_sets(st)]
    _render_features(console, named)


def _print_marker_map(console: Console, report: MarkerMapReport) -> None:
    """Print the marker-map health section for a job dir — read-only.

    Reports how many markers are resolved/novel/review, any rows that
    still need a fix, and whether each store is in sync with the map. It
    validates and diffs only; nothing is written.
    """
    console.print()
    if not report.present:
        console.print(
            "[yellow]marker map:[/yellow] no marker_map.csv — run coral ingest"
        )
        return
    console.print(
        f"[bold]marker map[/bold]  [dim]{report.n_markers} markers · "
        f"{report.n_resolved} resolved · {report.n_novel} novel · "
        f"{report.n_review} review[/dim]"
    )
    problems = report.problems
    if problems is not None:
        if problems.unresolved:
            names = ", ".join(o for o, _ in problems.unresolved)
            console.print(
                f"  [red]✗ {len(problems.unresolved)} marker(s) need a "
                f"name in marker_map.csv:[/red] {names}"
            )
        if problems.invalid_names:
            pairs = ", ".join(f"{o} -> {v}" for o, v in problems.invalid_names)
            console.print(
                f"  [red]✗ {len(problems.invalid_names)} invalid "
                f"mapping(s):[/red] {pairs}"
            )
        console.print(
            "  [dim]fix marker_map.csv, then re-run your next step[/dim]"
        )
        return
    if report.drifted:
        console.print(
            f"  [yellow]⧗ {len(report.drifted)} store(s) will sync on the "
            f"next step:[/yellow] {', '.join(report.drifted)}"
        )
        for name, done in report.stale.items():
            console.print(
                f"  [yellow]⚠ {name}: {', '.join(done)} already ran — may "
                f"be stale after sync (delete it or use a fresh "
                f"--job-dir)[/yellow]"
            )
    else:
        console.print("  [green]✓ every store in sync with the map[/green]")


def _cohort_cell(cell: StageCell) -> str:
    """One table cell: glyph + (duration | count) for the cohort view."""
    if cell.duration_s is not None:
        return f"{_glyph(cell.status)} [dim]{cell.duration_s:.0f}s[/dim]"
    if cell.count:
        return f"{_glyph(cell.status)}[dim]({cell.count})[/dim]"
    return _glyph(cell.status)


def _cohort_summary(rows: list[SlideStatus], report: MarkerMapReport) -> str:
    """The cohort tally line: slides, drift, errors, marker readiness."""
    n_drift = sum(len(r.drift) for r in rows)
    n_error = sum(
        1 for r in rows for c in r.stages.values() if c.status == "error"
    )
    parts = [f"{len(rows)} slide(s)"]
    parts.append(
        "drift: none"
        if n_drift == 0
        else f"[yellow]drift: {n_drift} issue(s)[/yellow]"
    )
    if n_error:
        parts.append(f"[red]{n_error} errored stage(s)[/red]")
    parts.append(_readiness(report))
    return " · ".join(parts)


def _readiness(report: MarkerMapReport) -> str:
    """One-word verdict on whether the next step can run — the marker gate.

    Ingest completing (a green tick) does not mean the pipeline can
    proceed: an unresolved or invalid marker map blocks every downstream
    step. This states that readiness plainly, so it is not missed.
    """
    if not report.present:
        # A missing map is not a blocker — stages use each slide's stored
        # marker names — so this is a note, not a "not ready" verdict.
        return "[dim]no marker_map.csv — using stored marker names[/dim]"
    attention = set(report.review_names)
    if report.problems is not None:
        attention |= {o for o, _ in report.problems.invalid_names}
        attention |= {o for o, _ in report.problems.unresolved}
    if attention:
        return (
            f"[yellow]⚠ not ready: {len(attention)} marker(s) need review "
            f"before the next step[/yellow]"
        )
    if report.problems is not None:
        return "[yellow]⚠ not ready: fix marker_map.csv first[/yellow]"
    return "[green]✓ markers ready[/green]"


def _print_issues(console: Console, rows: list[SlideStatus]) -> None:
    """List every errored stage (with reason) and every drift message."""
    for row in rows:
        for stage in STAGES:
            cell = row.stages[stage]
            if cell.status == "error":
                reason = f" — {cell.error}" if cell.error else ""
                console.print(
                    f"  [red]✗ {row.name}: {stage} failed{reason}[/red]"
                )
        for msg in row.drift:
            console.print(f"  [yellow]⚠[/yellow] {row.name}: {msg}")
