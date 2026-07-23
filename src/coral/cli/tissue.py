"""``coral tissue`` — Otsu tissue detection or user-mask import."""

from __future__ import annotations

import logging
import time
from pathlib import Path

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
from coral.utils import setup_logging

logger = logging.getLogger(__name__)


def tissue(
    job_dir: Path = typer.Option(
        ...,
        "--job-dir",
        help="The ingest job directory: CORAL runs on every <name>.zarr "
        "store inside it and records the run in its ledger. Run "
        "`coral ingest` first.",
    ),
    custom_mask_path: Path | None = typer.Option(
        None,
        "--custom-mask-path",
        help="Skip detection and import your own tissue instead: a "
        "DIRECTORY of per-store files named like the store (e.g. A-1.tif "
        "for A-1.zarr). Each may be a binary mask image (pixel > 0 = "
        "tissue, matching the store's full-resolution size) OR a "
        "QuPath-edited tissue.geojson. Written under "
        "tissue/tissue_imported/.",
    ),
    segmentation_method: str = typer.Option(
        "otsu",
        "--segmentation-method",
        help="Tissue segmentation algorithm by registered name (default "
        "otsu). Built-ins: 'otsu' (nuclear + structural threshold) and "
        "'carta' (DeepLabV3; nuclear-only; needs the `carta` extra). "
        "Writes tissue/tissue_<method>/.",
    ),
    structural_markers: str | None = typer.Option(
        None,
        "--structural-markers",
        help="otsu only — the structural marker(s) the Otsu threshold "
        "unions with the nuclear channel, COMMA-SEPARATED (e.g. "
        "PanCytokeratin,Vimentin). CARTA is nuclear-only and ignores this. "
        "Each must be a name present in marker_map.csv (an unknown name "
        "errors). Default: whichever of pan-cytokeratin, vimentin, "
        "collagen IV are on the panel.",
    ),
    max_bridge_distance: float = typer.Option(
        200.0,
        "--max-bridge-distance",
        help="Maximum gap (microns) the tissue boundary bridges before it "
        "concaves inward. Larger → a looser, more convex boundary; "
        "smaller → a tighter, more concave one.",
    ),
    preserve_tissue_holes: bool = typer.Option(
        False,
        "--preserve-tissue-holes/--no-preserve-tissue-holes",
        help="carta only: keep interior holes in the tissue mask (skip the "
        "bounded hole-fill in CARTA's cleanup). Ignored for otsu.",
    ),
    make_boundary_viz: bool = typer.Option(
        True,
        "--make-boundary-viz/--no-make-boundary-viz",
        help="Also save review images under each store's "
        "tissue/tissue_<method>/ folder: the nuclear + detected-boundary "
        "overlay (and, for otsu, the structural max-projection). Use "
        "--no-make-boundary-viz to write only the mask and polygons.",
    ),
) -> None:
    """Detect tissue (Otsu or CARTA), or import your own masks.

    Processes every ``.zarr`` store in ``--job-dir``. Each method writes
    under ``tissue/tissue_<method>/`` so methods can coexist. By default
    each store is Otsu-thresholded on its nuclear + structural channels
    and the resulting cell islands are enclosed in a single **tissue
    region** — an alpha-shape boundary whose one knob is
    ``--max-bridge-distance``. Pass ``--segmentation-method carta`` for the
    nuclear-only DeepLabV3 model (structural markers + the max-projection
    do not apply), or ``--custom-mask-path`` to import your own tissue (a
    mask image or a QuPath-edited ``tissue.geojson`` → ``tissue_imported``).
    ``tissue.geojson`` is the **source of truth**; a ``tissue_mask.png``
    figure (plus, unless ``--no-make-boundary-viz``, ``tissue_overlay.png``
    and, for otsu, ``max_projection.png``) is derived beside it — re-run
    this command after a QuPath edit to refresh that method's figures.
    Already-completed methods are skipped per-method (delete that method's
    folder to re-run); a failing store is logged and skipped; the command
    exits non-zero if any store failed.

    Args:
        job_dir: Ingest job directory of ``.zarr`` stores.
        custom_mask_path: Optional directory of per-store masks/geojson.
        segmentation_method: Registered method name (``otsu`` or ``carta``).
        structural_markers: Comma-separated structural markers (otsu only).
        max_bridge_distance: Max gap (microns) the tissue boundary bridges.
        preserve_tissue_holes: Keep interior holes (carta only).
        make_boundary_viz: Write overlay/max-projection review figures.

    Example:
        Run Otsu tissue detection on an ingested job::

            coral tissue --job-dir ./processed
    """
    setup_logging()
    from coral.tissue import (
        SEGMENTER_REGISTRY,
        CartaTissueSegmenter,
        OtsuTissueSegmenter,
    )

    method = (
        "imported" if custom_mask_path is not None else segmentation_method
    )
    if custom_mask_path is None and segmentation_method not in (
        SEGMENTER_REGISTRY
    ):
        raise typer.BadParameter(
            f"unknown segmentation method {segmentation_method!r}; "
            f"have {sorted(SEGMENTER_REGISTRY)}"
        )
    if segmentation_method == "carta" and structural_markers:
        raise typer.BadParameter(
            "CARTA is nuclear-only and ignores structural markers; drop "
            "--structural-markers (or use --segmentation-method otsu)."
        )

    mask_supplied = custom_mask_path is not None
    if mask_supplied and not custom_mask_path.is_dir():
        raise typer.BadParameter(
            "--custom-mask-path must be a directory of mask images "
            "(one per store, same name as the store)."
            " If store is A-1.zarr, the mask must be A-1.tif."
        )
    reject_param_conflict(
        mask_supplied=mask_supplied,
        params={
            "--structural-markers": structural_markers,
        },
        mode="Otsu tissue detection",
    )
    structural_list = (
        [s.strip() for s in structural_markers.split(",") if s.strip()]
        if structural_markers
        else None
    )

    all_slides = expand_cohort_zarr_dir(job_dir)
    if not all_slides:
        logger.error(
            "no .zarr stores in %s — run coral ingest first.", job_dir
        )
        raise typer.Exit(code=1)

    args = {
        "job_dir": job_dir,
        "custom_mask_path": custom_mask_path,
        "segmentation_method": segmentation_method,
        "structural_markers": structural_markers,
        "max_bridge_distance": max_bridge_distance,
        "preserve_tissue_holes": preserve_tissue_holes,
        "make_boundary_viz": make_boundary_viz,
    }
    total = len(all_slides)
    logger.info("Running coral tissue on %d slide(s):", total)
    logger.info("  Job dir:    %s", job_dir)
    detect_desc = (
        "CARTA tissue detection"
        if segmentation_method == "carta"
        else f"{segmentation_method} tissue detection"
    )
    logger.info(
        "  Method:    %s → tissue/tissue_%s/",
        "import your tissue masks (binarized at > 0)"
        if mask_supplied
        else detect_desc,
        method,
    )
    if not mask_supplied and segmentation_method == "otsu":
        logger.info(
            "  Max bridge distance (um):   %s",
            max_bridge_distance,
        )
    model = None
    if not mask_supplied:
        if segmentation_method == "carta":
            model = CartaTissueSegmenter.build(
                preserve_holes=preserve_tissue_holes
            )
        elif segmentation_method == "otsu":
            model = OtsuTissueSegmenter(
                max_bridge_distance=max_bridge_distance
            )
        else:
            # Future registered methods: zero-arg construct when possible.
            cls = SEGMENTER_REGISTRY[segmentation_method]
            build = getattr(cls, "build", None)
            model = build() if callable(build) else cls()

    run_marker_guardrail(job_dir, all_slides)

    start = time.perf_counter()
    with run_ledger(job_dir, tool="coral tissue", args=args):
        logger.info("")
        logger.info(
            "%s %d slide(s)...",
            "Importing tissue masks for"
            if mask_supplied
            else "Detecting tissue in",
            total,
        )
        failures = 0
        for i, item in enumerate(all_slides, start=1):
            bar_write(f"[{i}/{total}] {item.name}")
            with per_image_bar(desc=item.stem, total=1, unit="batch"):
                try:
                    _process_slide(
                        item,
                        model,
                        method=method,
                        custom_mask_path=custom_mask_path,
                        mask_supplied=mask_supplied,
                        structural_markers=structural_list,
                        make_boundary_viz=make_boundary_viz,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    bar_status(f"FAILED — {exc}")

        _log_footer(
            job_dir=job_dir,
            method=method,
            n_ok=total - failures,
            n_total=total,
            elapsed=time.perf_counter() - start,
            mask_supplied=mask_supplied,
            make_boundary_viz=make_boundary_viz,
        )
        if failures:
            logger.error("%d slide(s) failed.", failures)
            raise typer.Exit(code=1)


def _process_slide(
    item: Path,
    model: object,
    *,
    method: str,
    custom_mask_path: Path | None,
    mask_supplied: bool,
    structural_markers: list[str] | None,
    make_boundary_viz: bool,
) -> None:
    """Detect or import tissue for one store; the slide method logs it."""
    from coral.slide import CoralSlide

    mask_path = (
        resolve_artifact(custom_mask_path, item.stem)
        if custom_mask_path is not None
        else None
    )
    if mask_supplied and mask_path is None:
        bar_status("skipped — no tissue mask supplied")
        return
    if stage_completed(item, "tissue", slug=method):
        slide = CoralSlide.open(item)
        if slide.tissue_figures_stale(method):
            slide.refresh_tissue_figures(method)
            bar_status(
                "tissue figures out of sync — refreshed them from "
                "tissue.geojson (no re-detection)"
            )
        else:
            bar_status("skipped — already done")
        return

    slide = CoralSlide.open(item)
    if mask_path is not None:
        if mask_path.suffix.lower() == ".geojson":
            from coral.tissue.mask import read_tissue_geojson_mask

            _, height, width = slide.image.shape
            user_mask = read_tissue_geojson_mask(mask_path, height, width)
            source_kind = "geojson"
        else:
            from coral.io.masks import read_mask_image

            user_mask = read_mask_image(mask_path)
            source_kind = "image"
        slide.import_tissue_mask(
            user_mask,
            viz=make_boundary_viz,
            source=mask_path,
            source_kind=source_kind,
        )
    else:
        slide.detect_tissue(
            model,
            structural_markers=structural_markers,
            viz=make_boundary_viz,
        )


def _log_footer(
    *,
    job_dir: Path,
    method: str,
    n_ok: int,
    n_total: int,
    elapsed: float,
    mask_supplied: bool,
    make_boundary_viz: bool,
) -> None:
    """Completion line + a review hint."""
    logger.info("")
    if n_ok:
        logger.info(
            "Done! %s %d/%d slide(s) in %s",
            "Imported tissue masks for"
            if mask_supplied
            else "Detected tissue in",
            n_ok,
            n_total,
            fmt_duration(elapsed),
        )
        log_check_progress(job_dir)
    if n_ok and make_boundary_viz:
        logger.info(
            "Review each store's tissue/tissue_%s/tissue_overlay.png to "
            "check the result.",
            method,
        )
    if n_ok and not mask_supplied:
        logger.info("")
        logger.info(
            "To hand-correct a boundary: edit "
            "<store>/tissue/tissue_%s/tissue.geojson in QuPath and "
            "replace it — it is the tissue source of truth. Then re-run "
            "'coral tissue --segmentation-method %s' to refresh the "
            "tissue_mask.png / overlay figures.",
            method,
            method,
        )
