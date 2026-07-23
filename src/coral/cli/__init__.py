"""CORAL command-line interface.

Six commands wired into one Typer app:

- ``coral ingest <inputs>`` — Convert raw images to canonical OME-Zarr
- ``coral tissue <slides>`` — Otsu tissue detection or user-mask import
- ``coral patch <slides>`` — Grid patch extraction
- ``coral cell <slides>`` — Cell segmentation or user-mask import
- ``coral extract <slides>`` — FM-based feature extraction
- ``coral status <path>`` — Live per-stage status of a slide or cohort

Every stage command re-applies the edited ``marker_map.csv`` to its
stores automatically before running, so there is no separate apply step.

The package entry point is wired in ``pyproject.toml`` under
``[project.scripts]``.
"""

from __future__ import annotations

import typer

from coral.cli import (
    cell,
    extract,
    ingest,
    patch,
    status,
    tissue,
)

app = typer.Typer(
    name="coral",
    help="CORAL — spatial proteomics CLI.",
    no_args_is_help=True,
)
app.command(name="ingest", help="Convert raw images to canonical OME-Zarr")(
    ingest.ingest
)
app.command(name="tissue", help="Otsu tissue detection or user-mask import")(
    tissue.tissue
)
app.command(name="patch", help="Grid patch extraction")(patch.patch)
app.command(name="cell", help="Cell segmentation or user-mask import")(
    cell.cell
)
app.command(name="extract", help="FM-based feature extraction")(
    extract.extract
)
app.command(name="status", help="Live per-step status of a slide")(
    status.status
)

__all__ = ["app"]
