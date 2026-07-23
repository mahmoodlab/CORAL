"""``coral ingest`` — convert raw images to canonical OME-Zarr stores."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import typer
from pydantic import ValidationError

from coral.cli._cohort import (
    expand_cohort_zarr_dir,
    run_marker_guardrail,
    stage_completed,
)
from coral.cli._render import (
    fmt_duration,
    log_check_progress,
    log_marker_map_error,
    print_marker_mapping,
)
from coral.config.subset import Subset
from coral.io.atomic import atomic_write_text
from coral.io.ingest import (
    Resolution,
    _output_stem,
    _parse_channel_names,
    _parse_metadata_csv,
    _per_channel_dir_hint,
    convert_to_canonical,
    resolve_cohort_markers,
)
from coral.markers.marker_map import REVIEW_TOKEN, MarkerMapError
from coral.summary import run_ledger
from coral.tissue.infer import infer_dapi_index
from coral.utils import setup_logging

logger = logging.getLogger(__name__)

_TIFF_SUFFIXES = {".tiff", ".tif"}


def _expand_image_dir(d: Path) -> list[Path]:
    """Return the image inputs found directly inside a directory.

    Yields image files and sub-directories (each treated as one image),
    sorted by name. Other files are reported and skipped.

    Args:
        d: A directory whose immediate children are image sources.

    Returns:
        Sorted list of image-file paths and sub-directory paths.
    """
    entries: list[Path] = []
    for child in sorted(d.iterdir()):
        is_tiff = child.is_file() and child.suffix.lower() in _TIFF_SUFFIXES
        if child.is_dir() or is_tiff:
            entries.append(child)
        elif child.is_file():
            logger.info("%s is not an image — skipping", child.name)
    return entries


def _filter_inputs(
    inputs: list[Path],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[Path]:
    """Keep/drop images by EXACT entry name (no wildcards).

    ``include``/``exclude`` are lists of exact entry names — a
    ``name.extension`` file, or a per-channel directory name — as they
    appear under ``--image-dir``.
    """
    result = list(inputs)
    if include:
        keep = set(include)
        result = [p for p in result if p.name in keep]
    if exclude:
        drop = set(exclude)
        result = [p for p in result if p.name not in drop]
    return result


def _warn_unmatched_filters(
    inputs: list[Path],
    include: list[str] | None,
    exclude: list[str] | None,
) -> None:
    """Warn for each include/exclude name that matches no input image.

    A mistyped filter name (e.g. a wrong extension) would otherwise just
    silently shrink the cohort; surfacing it catches the typo before a
    long run.

    Args:
        inputs: All candidate images found under --image-dir.
        include: The subset's image-include names, or None.
        exclude: The subset's image-exclude names, or None.
    """
    available = {p.name for p in inputs}
    for field, names in (
        ("images.include", include),
        ("images.exclude", exclude),
    ):
        for name in names or []:
            if name not in available:
                logger.warning(
                    "--subset %s %r matches no image under --image-dir "
                    "— ignored.",
                    field,
                    name,
                )


def _review_next_steps(
    job_dir: Path,
    n_review: int,
    resolution: Resolution,
    *,
    print_marker_map: bool = False,
) -> None:
    """Log the marker-review outcome + how to resolve any review rows."""
    logger.info("")
    if print_marker_map:
        print_marker_mapping(resolution)
    if not n_review:
        logger.info(
            "👉 No manual review needed. Continue to your next step "
            "(tissue/cell/patch/extract)."
        )
        return

    review_names = [
        original
        for original, (_, level, _) in resolution.items()
        if level == REVIEW_TOKEN
    ]
    logger.warning(
        "%d marker(s) naming needs to be fixed before moving forward:",
        len(review_names),
    )
    for name in review_names:
        logger.warning("  [red]%s[/red]", name, extra={"no_highlight": True})
    logger.info("")
    logger.info("Fix by: ")
    logger.info("  1. Navigate to %s", job_dir / "marker_map.csv")
    logger.info("  2. For each row with status 'REVIEW', either:")
    logger.info(
        "    • Option 1: find your marker's canonical name in the "
        "registry (src/coral/markers/data/registry_v1.csv) and put it in "
        "'mapped_canonical_name' — the 'status' updates automatically."
    )
    logger.info(
        "    • Option 2: If your marker is not present in our canonical "
        "registry, change 'status' to 'NOVEL' and add your custom name to "
        "'mapped_canonical_name'."
    )
    logger.info(
        "⚠️   Warning: you cannot proceed to tissue detection, cell "
        "segmentation, and FM feature extraction until you fix this."
    )


def ingest(
    image_dir: Path = typer.Option(
        ...,
        "--image-dir",
        help=(
            "Directory of spatial proteomics images to ingest. "
            "Each entry inside it is one image: a single image file, or "
            "a sub-directory holding that image's per-channel images. "
            "Non-image files are skipped. Narrow the set with the "
            "'images' include/exclude in --subset."
        ),
    ),
    job_dir: Path = typer.Option(
        ...,
        "--job-dir",
        help=(
            "Output directory: every processed artifact — the "
            "<name>.zarr stores, the marker map, logs, and the "
            "run summary — is written here."
        ),
    ),
    mpp: float | None = typer.Option(
        None,
        "--mpp",
        help="Fallback microns/pixel for images lacking it. "
        "Applied to all images if no per-image CSV is given.",
    ),
    mpp_csv: Path | None = typer.Option(
        None,
        "--mpp-csv",
        help=(
            "Path to per-image microns-per-pixel overrides: a CSV with "
            "columns 'image,mpp', where 'image' is the exact entry name "
            "as it appears under --image-dir, with its extension "
            "(e.g. 'A-1.ome.tiff'). Takes precedence over the source's "
            "own mpp and over --mpp."
        ),
    ),
    channel_names: Path | None = typer.Option(
        None,
        "--channel-names",
        help=(
            "Path to a text file with one marker name per line, in "
            "channel order. Use ONLY when the image's embedded channel "
            "names are missing or you want to update them: these names "
            "replace the embedded ones in the output store (the source "
            "file is never modified). The number of lines must equal the "
            "image's channel count (cycles x channels for a multi-cycle "
            "image, which is flattened on ingest), or ingest errors."
        ),
    ),
    nuclear_marker: str | None = typer.Option(
        None,
        "--nuclear-marker",
        help=(
            "Force the nuclear channel by marker name (case-insensitive), "
            "overriding auto-inference. Errors if the name is not among the "
            "image's markers."
        ),
    ),
    subset: Path | None = typer.Option(
        None,
        "--subset",
        help=(
            "Path to a YAML narrowing what is processed. 'channels' "
            "(glob include/exclude) seeds which markers stay in the analysis "
            "set; 'images' (exact-name include/exclude) narrows which images "
            "are ingested. The kept channels must include a nuclear marker. "
            "An example YAML file is in example/subset.yaml. "
            "A copy of the YAML is saved to <job_dir>."
        ),
    ),
    keep_hoechst: bool = typer.Option(
        False,
        "--keep-hoechst",
        help=(
            "Keep Hoechst channels. By default they are excluded from the "
            "analysis panel (still stored on disk), since DRAQ5/DAPI "
            "is usually the working nuclear. A --subset channels include "
            "that specifically names them, or --nuclear-marker, also keeps "
            "them."
        ),
    ),
    print_marker_map: bool = typer.Option(
        False,
        "--print-marker-map",
        help=(
            "Print the full auto-mapped marker map "
            "(original marker --> canonical registry name)."
        ),
    ),
) -> None:
    """Convert raw spatial proteomics images to canonical OME-Zarr.

    For each image under ``--image-dir``, ingest reads the pixels,
    standardises them to a canonical ``(channel, y, x)`` layout, resolves
    each channel's marker name against a canonical registry, and writes a
    ``<name>.zarr`` store per image under ``--job-dir``, plus a cohort
    marker map you can review and re-apply.

    An image that fails is logged and skipped; the command exits non-zero
    if any image failed. Re-running re-applies the (edited) marker map
    without re-reading pixels; to re-ingest from pixels, start a fresh
    ``--job-dir`` (or delete the store).

    Args:
        image_dir: Directory of images (files or per-channel subdirs).
        job_dir: Output directory for ``.zarr`` stores and the marker map.
        mpp: Fallback microns/pixel when an image has none.
        mpp_csv: Optional CSV of per-image ``image,mpp`` overrides.
        channel_names: Optional CSV/list of channel names when missing.
        nuclear_marker: Force the nuclear channel by marker name.
        subset: Optional YAML narrowing images and/or channels.
        keep_hoechst: Keep Hoechst channels in the analysis panel.
        print_marker_map: Print the auto-mapped marker table.

    Example:
        Ingest a folder of images into a job directory::

            coral ingest --image-dir ./raw --job-dir ./processed
    """
    setup_logging()

    if not image_dir.is_dir():
        logger.error(
            "--image-dir must be a directory of images: %s", image_dir
        )
        raise typer.Exit(code=1)
    all_inputs = _expand_image_dir(image_dir)

    subset_obj: Subset | None = None
    if subset is not None:
        try:
            subset_obj = Subset.from_yaml(subset)
        except (FileNotFoundError, ValidationError) as exc:
            logger.error("could not read --subset %s: %s", subset, exc)
            raise typer.Exit(code=1) from exc

    img_include = (subset_obj.images.include or None) if subset_obj else None
    img_exclude = (subset_obj.images.exclude or None) if subset_obj else None
    _warn_unmatched_filters(all_inputs, img_include, img_exclude)

    selected = _filter_inputs(all_inputs, img_include, img_exclude)
    selected_set = set(selected)
    excluded = [p for p in all_inputs if p not in selected_set]
    if not selected:
        logger.error(
            "no images to ingest in %s (after the --subset image filter).",
            image_dir,
        )
        raise typer.Exit(code=1)

    # Verbose run header so a first-time user can confirm their inputs +
    # params were understood before any heavy work happens.
    n_files = sum(1 for p in selected if p.is_file())
    n_dirs = len(selected) - n_files
    exts = ", ".join(sorted({p.suffix for p in selected if p.suffix}))
    found = []
    if n_files:
        found.append(f"{n_files} image file(s) [{exts or 'no extension'}]")
    if n_dirs:
        word = "directory" if n_dirs == 1 else "directories"
        found.append(f"{n_dirs} per-channel {word}")
    logger.info("📥 Running coral ingest on %d image(s):", len(selected))
    logger.info("  Image dir:    %s", image_dir)
    logger.info("  Images found:    %s", ", ".join(found))
    for p in selected:
        logger.info("    + %s", p.name)
    hint = _per_channel_dir_hint(selected)
    if hint:
        logger.warning(
            "  %d input(s) look like the per-channel images of one slide "
            "(names match markers: %s). They are being ingested as "
            "separate single-channel slides — if that's wrong, point "
            "--image-dir at the parent folder.",
            len(hint),
            ", ".join(hint[:5]),
        )
    if excluded:
        logger.info("  Excluded by --subset (images):    %d", len(excluded))
        for p in excluded:
            logger.info("    - %s", p.name)
    logger.info("  Job dir:    %s", job_dir)
    name_src = (
        f"read from {channel_names}" if channel_names else "embedded in image"
    )
    logger.info("  Source of marker names:    %s", name_src)
    if subset_obj is not None:
        logger.info(
            "  Subset config:    %s (copied to %s)",
            subset,
            job_dir / "subset.yaml",
        )
        logger.info(
            "    Channels:  include [%s]  exclude [%s]",
            ", ".join(subset_obj.channels.include) or "all",
            ", ".join(subset_obj.channels.exclude) or "none",
        )
        logger.info(
            "    Images:    include [%s]  exclude [%s]",
            ", ".join(subset_obj.images.include) or "all",
            ", ".join(subset_obj.images.exclude) or "none",
        )

    mpp_map = _parse_metadata_csv(mpp_csv) if mpp_csv else None
    if mpp_map is not None:
        logger.info(
            "  mpp-csv provided:    %s (overrides --mpp and source for "
            "%d image(s))",
            mpp_csv,
            len(mpp_map),
        )
    names = _parse_channel_names(channel_names) if channel_names else None

    args = {
        "image_dir": image_dir,
        "job_dir": job_dir,
        "mpp": mpp,
        "mpp_csv": mpp_csv,
        "channel_names": channel_names,
        "nuclear_marker": nuclear_marker,
        "subset": subset,
        "keep_hoechst": keep_hoechst,
        "print_marker_map": print_marker_map,
    }
    with run_ledger(job_dir, tool="coral ingest", args=args):
        if subset is not None:
            # Persist the exact subset used, next to the marker map, so the
            # run's channel + image selection is always recoverable.
            job_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                job_dir / "subset.yaml", Path(subset).read_text()
            )
        try:
            resolution, n_review = resolve_cohort_markers(
                selected,
                job_dir,
                channel_names=names,
                channels=subset_obj.channels if subset_obj else None,
                keep_hoechst=keep_hoechst,
                nuclear_marker=nuclear_marker,
            )
        except MarkerMapError as exc:
            log_marker_map_error(exc)
            raise typer.Exit(code=1) from exc
        except ValueError as exc:
            logger.error("%s", exc)
            raise typer.Exit(code=1) from exc

        # Pre-flight: a kept marker set with no nuclear stain can't ingest
        # at all — fail once, up front, not the same error per image.
        kept = list(dict.fromkeys(r for r, _, k in resolution.values() if k))
        if not kept:
            logger.error(
                "every marker is excluded from the analysis panel — keep "
                "at least one (check --subset / --keep-hoechst)."
            )
            raise typer.Exit(code=1)
        if nuclear_marker is None and infer_dapi_index(kept) is None:
            allm = list(dict.fromkeys(r for r, _, _ in resolution.values()))
            if infer_dapi_index(allm) is not None:
                logger.error(
                    "the nuclear marker is excluded from the analysis "
                    "panel — tissue, cell, and feature extraction need it. "
                    "Keep it (check --subset), or name another with "
                    "--nuclear-marker."
                )
            else:
                logger.error(
                    "no nuclear stain in the kept markers (%s). Every "
                    "image needs one — supply real marker names with "
                    "--channel-names, or name the nuclear channel with "
                    "--nuclear-marker.",
                    ", ".join(kept),
                )
            raise typer.Exit(code=1)

        # A re-run is any run where a target store already exists; used
        # below to re-apply an edited marker_map.csv (a first run skips it).
        expected_stores = [
            job_dir / f"{_output_stem(item)}.zarr" for item in selected
        ]
        had_existing = any(s.exists() for s in expected_stores)

        start = time.perf_counter()
        logger.info("Ingesting %d image(s)...", len(selected))
        failures: list[tuple[str, str]] = []
        total = len(selected)
        for i, item in enumerate(selected, start=1):
            out_store = job_dir / f"{_output_stem(item)}.zarr"
            if stage_completed(out_store, "ingest"):
                logger.info(
                    "[%d/%d] %s skipped — already ingested",
                    i,
                    total,
                    item.name,
                )
                continue
            try:
                convert_to_canonical(
                    item,
                    job_dir,
                    resolution=resolution,
                    mpp=mpp,
                    mpp_map=mpp_map,
                    channel_names=names,
                    nuclear_marker=nuclear_marker,
                    quiet=False,
                )
                logger.info("[%d/%d] %s ingested", i, total, item.name)
            except Exception as exc:
                failures.append((item.name, str(exc)))
                logger.error("[%d/%d] %s FAILED", i, total, item.name)
                logger.error("  %s", exc)

        n_ok = len(selected) - len(failures)
        if had_existing:
            # Re-run: re-apply a possibly-edited marker_map.csv to the
            # already-ingested stores — metadata-only, no pixel re-read,
            # the same sync every downstream stage performs.
            run_marker_guardrail(job_dir, expand_cohort_zarr_dir(job_dir))
        if n_ok:
            logger.info(
                "✅ Done! Ingested %d/%d image(s) in %s",
                n_ok,
                len(selected),
                fmt_duration(time.perf_counter() - start),
            )
            log_check_progress(job_dir)
            _review_next_steps(
                job_dir,
                n_review,
                resolution,
                print_marker_map=print_marker_map,
            )
        if failures:
            logger.error("%d image(s) failed", len(failures))
            raise typer.Exit(code=1)
