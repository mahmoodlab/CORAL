"""``coral cell`` — Cellpose segmentation or user-mask import."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import typer

from coral.cli._cohort import (
    expand_cohort_zarr_dir,
    reject_param_conflict,
    resolve_artifact,
    run_marker_guardrail,
    stage_completed,
)
from coral.cli._render import (
    bar_status,
    bar_write,
    fmt_duration,
    log_check_progress,
    per_image_bar,
)
from coral.summary import run_ledger
from coral.utils import CoralError, resolve_device, setup_logging

logger = logging.getLogger(__name__)


def cell(
    job_dir: Path = typer.Option(
        ...,
        "--job-dir",
        help="The ingest job directory: CORAL segments cells in every "
        "<name>.zarr store inside it and records the run in its ledger. "
        "Run `coral ingest` (and `coral tissue`) first.",
    ),
    custom_mask_path: Path | None = typer.Option(
        None,
        "--custom-mask-path",
        help="Skip Cellpose and import your own cell instance masks instead "
        "(pixel = cell_id, 0 = background): a DIRECTORY of masks named like "
        "each store (e.g. A-1.tif for A-1.zarr) — one file for a one-store "
        "job, many for many stores. Imported ids are preserved and the mask "
        "is NOT tissue-restricted.",
    ),
    custom_label_path: Path | None = typer.Option(
        None,
        "--custom-label-path",
        help="Attach per-cell phenotype labels from CSVs (columns: cell_id, "
        "label, optional x, y): a DIRECTORY of CSVs named like each store. "
        "Requires --custom-mask-path (Cellpose relabels cells, so external "
        "ids can only match an imported mask).",
    ),
    label_set: str | None = typer.Option(
        None,
        "--label-set",
        help="Name to file the imported labels under (default: the "
        "--custom-label-path directory name). Lets several label sets "
        "coexist on one slide.",
    ),
    membrane_markers: str | None = typer.Option(
        None,
        "--membrane-markers",
        help="Membrane/structural marker(s) summed into Cellpose's second "
        "channel, COMMA-SEPARATED (e.g. CD45,PanCytokeratin). Each must be a "
        "name present in marker_map.csv (an unknown name errors). Default: "
        "the panel's structural channels.",
    ),
    cell_diameter: float | None = typer.Option(
        None,
        "--cell-diameter",
        help="Expected cell diameter in pixels. Default: cpsam estimates it "
        "per slide — only set this if the automatic estimate looks wrong.",
    ),
    min_cell_size: int = typer.Option(
        15,
        "--min-cell-size",
        help="Discard segmented objects smaller than this many pixels "
        "(removes debris and noise specks). Default 15.",
    ),
    make_cell_viz: bool = typer.Option(
        True,
        "--make-cell-viz/--no-make-cell-viz",
        help="Also save the review overlay to each store "
        "(cells/cell_overlay.png: cell outlines on the nuclear channel, "
        "tissue boundary in green). Use --no-make-cell-viz to write only the "
        "mask and the centroid table.",
    ),
    no_tissue: bool = typer.Option(
        False,
        "--no-tissue",
        help="Segment the WHOLE image with no tissue mask (every cell kept), "
        "instead of restricting to the tissue (the default, which needs "
        "`coral tissue` first). For a core/ROI that fills the frame, or when "
        "tissue detection is unreliable — can be slow on a large WSI. "
        "Segmentation only; not valid with --custom-mask-path.",
    ),
    tissue_method: str | None = typer.Option(
        None,
        "--tissue-method",
        help="Which tissue/tissue_<method>/ mask to restrict to when "
        "several methods coexist (ignored with --no-tissue). Default: "
        "auto (exactly one → use it; several → otsu).",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device for Cellpose: auto (default), cpu, cuda, cuda:N (pick a "
        "GPU by index), or mps (Apple Silicon). Mutually exclusive with "
        "--gpu.",
    ),
    gpu: int | None = typer.Option(
        None,
        "--gpu",
        help="Shorthand for --device cuda:N (a CUDA GPU index, e.g. 0). "
        "Default: auto — cuda if available, else mps, else CPU.",
    ),
) -> None:
    """Segment cells with Cellpose (cpsam), or import your own mask.

    Processes every ``.zarr`` store in ``--job-dir``. By default each store
    is segmented with Cellpose on its nuclear + membrane channels, restricted
    to the tissue mask (run ``coral tissue`` first). Pass
    ``--custom-mask-path`` (a directory of per-store masks) to import
    your own instance masks
    instead (ids preserved, not tissue-restricted), and ``--custom-label-path``
    to attach per-cell phenotypes to an imported mask. Writes a flat
    ``cells/`` folder into each store — ``cells/cell_mask`` (the int32
    instance mask, the source of truth), ``cells/cell_centroids.csv``, and,
    unless ``--no-make-cell-viz``, ``cells/cell_overlay.png``. Already-
    completed stores are skipped (delete the ``cells/`` folder or use a
    fresh ``--job-dir`` to re-run); a failing store is logged and skipped;
    the command exits non-zero if any store failed.

    Args:
        job_dir: Ingest job directory of ``.zarr`` stores.
        custom_mask_path: Optional directory of per-store instance masks.
        custom_label_path: Optional directory of per-cell phenotype CSVs.
        label_set: Name under which imported labels are stored.
        membrane_markers: Comma-separated membrane channels for Cellpose.
        cell_diameter: Expected cell diameter in pixels (optional).
        min_cell_size: Drop objects smaller than this many pixels.
        make_cell_viz: Write ``cell_overlay.png`` review figures.
        no_tissue: Segment the whole image (ignore the tissue mask).
        tissue_method: Which ``tissue/tissue_<method>/`` mask to restrict
            to when several coexist (default: auto).
        device: Device (``auto``, ``cpu``, ``cuda``, ``cuda:N``, ``mps``).
        gpu: CUDA index shorthand (mutually exclusive with ``device``).

    Example:
        Segment cells on an ingested job after tissue detection::

            coral cell --job-dir ./processed
    """
    setup_logging()

    mask_supplied = custom_mask_path is not None
    reject_param_conflict(
        mask_supplied=mask_supplied,
        params={
            "--membrane-markers": membrane_markers,
            "--cell-diameter": cell_diameter,
        },
        mode="Cellpose segmentation",
    )
    for flag, path in (
        ("--custom-mask-path", custom_mask_path),
        ("--custom-label-path", custom_label_path),
    ):
        if path is not None and not path.is_dir():
            raise typer.BadParameter(
                f"{flag} must be a DIRECTORY of per-store files named like "
                f"each store (e.g. A-1.tif for A-1.zarr)."
            )
    if custom_label_path is not None and not mask_supplied:
        raise typer.BadParameter(
            "--custom-label-path needs an imported cell mask "
            "(--custom-mask-path); Cellpose relabels cells, so external "
            "label ids cannot match."
        )
    if no_tissue and mask_supplied:
        raise typer.BadParameter(
            "--no-tissue applies to Cellpose segmentation; an imported mask "
            "is already used as-is (not tissue-restricted)."
        )
    membrane_list = (
        [m.strip() for m in membrane_markers.split(",") if m.strip()]
        if membrane_markers
        else None
    )

    all_slides = expand_cohort_zarr_dir(job_dir)
    if not all_slides:
        logger.error(
            "no .zarr stores in %s — run coral ingest first.", job_dir
        )
        raise typer.Exit(code=1)

    # Cellpose is an environment/install requirement, not a per-slide
    # outcome — fail fast and clearly BEFORE claiming to segment anything.
    # Only when there is actually a slide to segment: a re-run of an
    # already-completed job is a no-op and needs no Cellpose install.
    if not mask_supplied:
        from coral.cells.cellpose import CELLPOSE_AVAILABLE

        has_pending = any(not stage_completed(s, "cells") for s in all_slides)
        if has_pending and not CELLPOSE_AVAILABLE:
            logger.error(
                "coral cell needs Cellpose to segment cells — install the "
                "cells extra with `uv sync --extra cells`, or import "
                "existing masks with --custom-mask-path."
            )
            raise typer.Exit(code=1)

    if device is not None and gpu is not None:
        raise typer.BadParameter("pass one of --device / --gpu, not both")
    device_spec = (
        device
        if device is not None
        else (f"cuda:{gpu}" if gpu is not None else None)
    )

    args = {
        "job_dir": job_dir,
        "custom_mask_path": custom_mask_path,
        "custom_label_path": custom_label_path,
        "label_set": label_set,
        "membrane_markers": membrane_markers,
        "cell_diameter": cell_diameter,
        "min_cell_size": min_cell_size,
        "make_cell_viz": make_cell_viz,
        "no_tissue": no_tissue,
        "tissue_method": tissue_method,
        "device": device_spec,
        "gpu": gpu,
    }

    total = len(all_slides)
    logger.info("Running coral cell on %d slide(s):", total)
    logger.info("  Job dir:    %s", job_dir)
    if mask_supplied:
        method = "import your cell masks (pixel = cell_id)"
    elif no_tissue:
        method = "Cellpose segmentation (cpsam), whole slide (no tissue)"
    else:
        method = "Cellpose segmentation (cpsam), restricted to tissue"
    logger.info("  Method:    %s", method)
    if not mask_supplied:
        logger.info(
            "  Cell diameter (px):    %s",
            cell_diameter if cell_diameter is not None else "auto (cpsam)",
        )
        logger.info("  Min cell size (px):    %s", min_cell_size)

    model = None
    if not mask_supplied:
        from coral.cells import CellposeSegmenter

        try:
            device = resolve_device(device_spec)
        except CoralError as exc:
            logger.error("%s", exc)
            raise typer.Exit(code=1) from exc
        model = CellposeSegmenter(
            diameter=cell_diameter, min_size=min_cell_size, device=device
        )

    run_marker_guardrail(job_dir, all_slides)

    start = time.perf_counter()
    with run_ledger(job_dir, tool="coral cell", args=args):
        logger.info("")
        logger.info(
            "%s %d slide(s)...",
            "Importing cell masks for"
            if mask_supplied
            else "Segmenting cells in",
            total,
        )
        # Load once before [1/N] so download/construction is not charged
        # to the first image's progress bar.
        if model is not None and any(
            not stage_completed(s, "cells") for s in all_slides
        ):
            model.ensure_loaded()
        failures = 0
        for i, item in enumerate(all_slides, start=1):
            bar_write(f"[{i}/{total}] {item.name}")
            with per_image_bar(desc=item.stem, total=1, unit="batch"):
                try:
                    _process_slide(
                        item,
                        model,
                        custom_mask_path=custom_mask_path,
                        custom_label_path=custom_label_path,
                        label_set=label_set,
                        membrane_markers=membrane_list,
                        mask_supplied=mask_supplied,
                        make_cell_viz=make_cell_viz,
                        no_tissue=no_tissue,
                        tissue_method=tissue_method,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    bar_status(f"FAILED — {exc}")

        _log_footer(
            job_dir=job_dir,
            n_ok=total - failures,
            n_total=total,
            elapsed=time.perf_counter() - start,
            mask_supplied=mask_supplied,
            make_cell_viz=make_cell_viz,
        )
        if failures:
            logger.info("")
            logger.error(
                "%d/%d slide(s) failed — see the errors above.",
                failures,
                total,
            )
            raise typer.Exit(code=1)


def _process_slide(
    item: Path,
    model: Any,  # noqa: ANN401 — a BaseCellSegmenter or None (import mode)
    *,
    custom_mask_path: Path | None,
    custom_label_path: Path | None,
    label_set: str | None,
    membrane_markers: list[str] | None,
    mask_supplied: bool,
    make_cell_viz: bool,
    no_tissue: bool,
    tissue_method: str | None,
) -> None:
    """Segment or import cells for one store, then attach any labels.

    The per-store result lines (channels used, cell count) are logged by the
    slide methods, so this only handles the skip / label / dispatch logic.
    """
    from coral.slide import CoralSlide

    cell_path = (
        resolve_artifact(custom_mask_path, item.stem)
        if custom_mask_path is not None
        else None
    )
    label_path = (
        resolve_artifact(custom_label_path, item.stem, ext=".csv")
        if custom_label_path is not None
        else None
    )

    if mask_supplied and cell_path is None:
        bar_status("skipped — no cell mask supplied")
        return
    if stage_completed(item, "cells"):
        bar_status("skipped — already done")
        return

    slide = CoralSlide.open(item)
    if cell_path is not None:
        from coral.io.masks import read_mask_image

        slide.import_cell_mask(read_mask_image(cell_path), viz=make_cell_viz)
    else:
        slide.segment_cells(
            model,
            membrane_markers=membrane_markers,
            viz=make_cell_viz,
            restrict_to_tissue=not no_tissue,
            tissue_method=tissue_method,
        )

    if label_path is not None and custom_label_path is not None:
        import pandas as pd

        chosen = label_set or custom_label_path.name
        slide.import_cell_labels(pd.read_csv(label_path), label_set=chosen)
        bar_status(f"attached labels (set={chosen})")


def _log_footer(
    *,
    job_dir: Path,
    n_ok: int,
    n_total: int,
    elapsed: float,
    mask_supplied: bool,
    make_cell_viz: bool,
) -> None:
    """Log the closing summary (nothing succeeded → no 'Done!')."""
    if not n_ok:
        return  # nothing completed — the caller reports the failure
    logger.info("")
    verb = "Imported cell masks for" if mask_supplied else "Segmented cells in"
    logger.info(
        "Done! %s %d/%d slide(s) in %s",
        verb,
        n_ok,
        n_total,
        fmt_duration(elapsed),
    )
    log_check_progress(job_dir)
    if make_cell_viz:
        logger.info(
            "Review each store's cells/cell_overlay.png to check the result."
        )
