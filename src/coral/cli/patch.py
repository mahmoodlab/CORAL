"""``coral patch`` — grid or cell-centered patch extraction on stores."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import cast

import typer

from coral.cli._cohort import (
    expand_cohort_zarr_dir,
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
from coral.config import PatchConfig, PatchMode
from coral.summary import run_ledger
from coral.utils import setup_logging

logger = logging.getLogger(__name__)


def patch(
    job_dir: Path = typer.Option(
        ...,
        "--job-dir",
        help="The ingest job directory: CORAL extracts patches for every "
        "<name>.zarr store inside it and records the run in its ledger. "
        "Run `coral ingest` (and `coral tissue`) first.",
    ),
    patch_size: int | None = typer.Option(
        None,
        "--patch-size",
        help="Patch side in pixels, at the slide's base resolution. "
        "Default: 256 for grid mode, 64 for cell-centered.",
    ),
    stride: int | None = typer.Option(
        None,
        "--stride",
        help="Step between neighbouring grid patches, in pixels. Default: "
        "the patch size (no overlap). Give this OR --overlap, not both. "
        "Inert in cell-centered mode.",
    ),
    overlap: float = typer.Option(
        0.0,
        "--overlap",
        help="Fractional overlap between neighbouring grid patches "
        "(0.0–1.0; e.g. 0.5 steps by half a patch). Ignored when --stride "
        "is given. Inert in cell-centered mode.",
    ),
    mpp: float | None = typer.Option(
        None,
        "--mpp",
        help="Target microns-per-pixel. Default: the slide's base "
        "resolution (level 0), which is what KRONOS encodes. A coarser "
        "value (downsampling) is a future feature; finer is impossible.",
    ),
    mode: str = typer.Option(
        "grid",
        "--mode",
        help="grid (default): tile the whole slide, keeping every patch "
        "with its tissue fraction (needs `coral tissue`). cell: one patch "
        "per segmented cell (needs `coral cell`; --patch-size defaults "
        "to 64).",
    ),
    make_patch_viz: bool = typer.Option(
        True,
        "--make-patch-viz/--no-make-patch-viz",
        help="Also save the review overlay to each store "
        "(patches/<slug>/patch_overlay.png: the patch grid over the "
        "nuclear channel, kept patches green). Use --no-make-patch-viz to "
        "write only the coordinates and config.",
    ),
    tissue_method: str | None = typer.Option(
        None,
        "--tissue-method",
        help="Which tissue/tissue_<method>/ mask to use when several "
        "methods coexist. Review tissue overlays first, then pass the "
        "chosen method. Default: auto (exactly one → use it; several → "
        "otsu).",
    ),
) -> None:
    """Extract patch coordinates for each slide in a job.

    Processes every ``.zarr`` store in ``--job-dir``. In the default grid
    mode each store is tiled at its base resolution and **every** patch
    is kept, each scored by the fraction of its box under the tissue
    mask (run ``coral tissue`` first), writing
    ``patches/<slug>/{coords, tissue_prop, config.json}`` — pick a
    tissue cut-off downstream against ``tissue_prop``. Pass
    ``--mode cell`` to emit one patch per segmented cell instead
    (run ``coral cell`` first), writing ``{coords, cell_ids, config.json}``;
    ``--stride`` and ``--overlap`` are inert in that
    mode. Unless ``--no-make-patch-viz`` a ``patch_overlay.png`` review
    figure is written to each store. An already-completed patch set is
    skipped (delete ``patches/<slug>/`` to re-run); a failing store is
    logged and the command exits non-zero if any store failed.

    Args:
        job_dir: Ingest job directory of ``.zarr`` stores.
        patch_size: Patch side length in pixels (default 256 grid / 64 cell).
        stride: Grid stride in pixels (default: equal to ``patch_size``).
        overlap: Fractional overlap alternative to an absolute stride.
        mpp: Target microns-per-pixel for the patch-set slug.
        mode: ``grid`` (whole-image grid) or ``cell`` (one patch per cell).
        make_patch_viz: Write ``patch_overlay.png`` review figures.
        tissue_method: Which ``tissue/tissue_<method>/`` mask to score
            patches against (``None`` auto-resolves).

    Example:
        Extract 256 px grid patches after tissue detection::

            coral patch --job-dir ./processed --patch-size 256
    """
    from pydantic import ValidationError

    setup_logging()

    cli_modes = {"grid": "grid", "cell": "cell_centered"}
    if mode not in cli_modes:
        raise typer.BadParameter(
            f"unknown --mode {mode!r}; choose 'grid' or 'cell'"
        )
    norm_mode = cli_modes[mode]
    resolved_size = (
        patch_size
        if patch_size is not None
        else (64 if norm_mode == "cell_centered" else 256)
    )
    try:
        config = PatchConfig(
            patch_size=resolved_size,
            stride=stride,
            overlap=overlap,
            target_mpp=mpp,
            mode=cast(PatchMode, norm_mode),
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc

    all_slides = expand_cohort_zarr_dir(job_dir)
    if not all_slides:
        logger.error(
            "no .zarr stores in %s — run coral ingest first.", job_dir
        )
        raise typer.Exit(code=1)

    args = {
        "job_dir": job_dir,
        "patch_size": patch_size,
        "stride": stride,
        "overlap": overlap,
        "mpp": mpp,
        "mode": mode,
        "make_patch_viz": make_patch_viz,
        "tissue_method": tissue_method,
    }

    total = len(all_slides)
    cell_mode = config.mode == "cell_centered"
    logger.info("Running coral patch on %d slide(s):", total)
    logger.info("  Job dir:    %s", job_dir)
    logger.info(
        "  Method:    %s",
        "cell-centered (one patch per segmented cell)"
        if cell_mode
        else "grid (tiled, every patch kept + tissue-scored)",
    )
    logger.info("  Patch size (px):    %d", config.patch_size)
    if not cell_mode:
        logger.info("  Stride (px):    %d", config.effective_stride)

    run_marker_guardrail(job_dir, all_slides)

    start = time.perf_counter()
    with run_ledger(job_dir, tool="coral patch", args=args):
        logger.info("")
        logger.info("Extracting patches in %d slide(s)...", total)
        failures = 0
        for i, item in enumerate(all_slides, start=1):
            bar_write(f"[{i}/{total}] {item.name}")
            try:
                with per_image_bar(desc=item.stem, total=1, unit="patch"):
                    _process_slide(
                        item,
                        config,
                        make_patch_viz=make_patch_viz,
                        tissue_method=tissue_method,
                    )
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logger.error("%s: %s", item.name, exc)

        _log_footer(
            job_dir=job_dir,
            n_ok=total - failures,
            n_total=total,
            elapsed=time.perf_counter() - start,
            make_patch_viz=make_patch_viz,
        )
        if failures:
            logger.info("")
            logger.error(
                "%d/%d slide(s) failed.",
                failures,
                total,
            )
            raise typer.Exit(code=1)


def _process_slide(
    item: Path,
    config: PatchConfig,
    *,
    make_patch_viz: bool,
    tissue_method: str | None,
) -> None:
    """Extract patches for one store; the slide method logs the result.

    An already-completed patch set is skipped (delete ``patches/<slug>/``
    to re-run). The per-store result line (patch count, overlay) is logged
    by ``CoralSlide.extract_patches``.
    """
    from coral.slide import CoralSlide
    from coral.slide.state import load_state
    from coral.tissue.paths import resolve_tissue_method

    slide_mpp = config.target_mpp or load_state(item).meta.mpp
    if slide_mpp is not None and stage_completed(
        item, "patch", slug=config.resolved_slug(slide_mpp)
    ):
        bar_status("skipped — already done")
        return

    resolved = resolve_tissue_method(item, tissue_method)
    logger.info("  tissue method: %s", resolved)
    slide = CoralSlide.open(item)
    slide.extract_patches(config, viz=make_patch_viz, tissue_method=resolved)


def _log_footer(
    *,
    job_dir: Path,
    n_ok: int,
    n_total: int,
    elapsed: float,
    make_patch_viz: bool,
) -> None:
    """Completion line + a review hint (nothing succeeded → no 'Done!')."""
    if not n_ok:
        return  # nothing completed — the caller reports the failure
    logger.info("")
    logger.info(
        "Done! Extracted patches for %d/%d slide(s) in %s",
        n_ok,
        n_total,
        fmt_duration(elapsed),
    )
    log_check_progress(job_dir)
    if make_patch_viz:
        logger.info(
            "Review each store's patches/<slug>/patch_overlay.png to check "
            "the result."
        )
