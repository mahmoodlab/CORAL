"""``coral ingest-wsi`` — convert a qptiff whole slide to OME-Zarr."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from coral.cli._render import fmt_duration, log_check_progress
from coral.cli.ingest import _filter_inputs, _warn_unmatched_filters
from coral.config.subset import Subset
from coral.dearray import DEFAULT_CONF, dearray_slide, match_cores_file
from coral.io.atomic import atomic_write_text
from coral.io.ingest import (
    Resolution,
    _parse_channel_names,
    _parse_metadata_csv,
    resolve_cohort_markers,
)
from coral.io.ingest_wsi import (
    convert_wsi_to_canonical,
    qptiff_marker_names,
    wsi_output_stem,
)
from coral.io.readers.qptiff import QPTIFF_EXTENSIONS, read_qptiff_metadata
from coral.markers.marker_map import REVIEW_TOKEN, MarkerMapError
from coral.summary import run_ledger
from coral.utils import setup_logging
from coral.utils.errors import ReaderError

logger = logging.getLogger(__name__)

#: Unresolved markers named in a refusal before it says "and N more".
_MAX_NAMED_MARKERS = 8


def _collect_inputs(image_dir: Path) -> list[Path]:
    """The qptiff files under ``--image-dir``, in name order.

    A single qptiff path is accepted too, so pointing at one file does not
    require making a directory for it.
    """
    if image_dir.is_dir():
        return sorted(
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.name.lower().endswith(QPTIFF_EXTENSIONS)
        )
    return [image_dir]


def ingest_wsi(
    image_dir: Path = typer.Option(
        ...,
        "--image-dir",
        help=(
            "Directory of PerkinElmer/Akoya .qptiff whole-slide "
            "scans; one file or many. Each becomes one <name>.zarr "
            "store. A single .qptiff path is accepted too. Every scan in "
            "one run must share a marker panel, since a job dir holds "
            "one marker map."
        ),
    ),
    job_dir: Path = typer.Option(
        ...,
        "--job-dir",
        help=(
            "Output directory: the <name>.zarr store(s), the marker map, "
            "logs, and the run summary are written here."
        ),
    ),
    level: int = typer.Option(
        0,
        "--level",
        min=0,
        help=(
            "Pyramid level to ingest. 0 (default) is full resolution; each "
            "level halves each side. Use a reduced level for a quick "
            "end-to-end check without writing tens of gigabytes."
        ),
    ),
    mpp: float | None = typer.Option(
        None,
        "--mpp",
        help=(
            "Microns/pixel, OVERRIDING whatever the qptiff records. "
            "Required when the file carries none. Use with care: the "
            "scanner normally records this correctly, and morphology, "
            "areas and scalebars all follow from it. --mpp-csv still "
            "takes precedence, so a per-image correction beats this "
            "blanket one."
        ),
    ),
    mpp_csv: Path | None = typer.Option(
        None,
        "--mpp-csv",
        help=(
            "Per-image microns-per-pixel overrides: a CSV with columns "
            "'image,mpp', where 'image' is the qptiff file name with its "
            "extension. Takes precedence over the file's own mpp."
        ),
    ),
    channel_names: Path | None = typer.Option(
        None,
        "--channel-names",
        help=(
            "Text file with one marker name per line, in channel order, "
            "overriding the names the qptiff carries in its <Biomarker> "
            "XML. Only needed when those names are missing or wrong. The "
            "line count must equal the channel count."
        ),
    ),
    subset: Path | None = typer.Option(
        None,
        "--subset",
        help=(
            "YAML narrowing two independent axes, exactly as `coral ingest` "
            "reads it: 'images' selects scans by exact filename, and "
            "'channels' seeds which markers stay in the analysis set by "
            "glob. An example is in example/subset.yaml. A copy is written "
            "to <job-dir>/subset.yaml so a run's selection is recoverable. "
            "Note that de-selecting a channel flags it, it does not drop it: "
            "every channel is still written to the store."
        ),
    ),
    nuclear_marker: str | None = typer.Option(
        None,
        "--nuclear-marker",
        help=(
            "Force the nuclear channel by marker name (case-insensitive). "
            "Without it the stain is inferred exactly as `coral ingest` "
            "infers it (the same resolver, normally DAPI), preferring a "
            "kept channel."
        ),
    ),
    dearray: bool = typer.Option(
        False,
        "--dearray",
        help=(
            "The scan is a TMA: detect its cores after ingesting, with "
            "CARTA's detector, and write dearray/cores.geojson inside the "
            "store. Off by default because a whole slide is a single tissue "
            "section unless you say otherwise."
        ),
    ),
    export_cores: bool = typer.Option(
        False,
        "--export-cores",
        help=(
            "Also cut each core out as its own OME-Zarr under "
            "<job-dir>/cores/<slide>/. Implies --dearray. Off by default: "
            "a core is a region of the slide, and copying every core costs "
            "minutes and gigabytes for pixels you already have."
        ),
    ),
    core_levels: str = typer.Option(
        "0",
        "--core-levels",
        help=(
            "Pyramid levels to give each exported core: '0' (default) for "
            "full resolution alone, or 'all' for every level the parent has. "
            "A core is a few thousand pixels and opens instantly without a "
            "pyramid, so 'all' is for readers that insist on one."
        ),
    ),
    dearray_conf: float = typer.Option(
        DEFAULT_CONF,
        "--dearray-conf",
        min=0.0,
        max=1.0,
        help=(
            "Detector confidence floor. CARTA's default is 0.25. Lower it to "
            "pick up faint cores, at the cost of false positives."
        ),
    ),
    cores_from: Path | None = typer.Option(
        None,
        "--cores-from",
        help=(
            "Directory of GeoJSON core boxes to use instead of detecting: "
            "<slide>.geojson per scan, matched by name. Either QuPath "
            "corrections or boxes from your own tooling; CORAL does not "
            "distinguish. Annotations must be classified as Core. Boxes are "
            "clipped, deduplicated and renumbered, and land under "
            "dearray/dearray_imported/ so a later --dearray cannot overwrite "
            "them."
        ),
    ),
) -> None:
    """Convert a qptiff whole slide to a canonical OME-Zarr store.

    Reads the scan's Baseline series, takes each channel's marker name
    from the file's own ``<Biomarker>`` metadata, and writes the same
    store ``coral ingest`` writes — pixels, channel names, nuclear
    channel, microns-per-pixel, thumbnail and state.

    Pixels are streamed channel by channel, so a full-resolution slide
    does not need to fit in memory. The whole slide becomes one store;
    cores are then located within it as geometry, not copied out, so
    dearraying costs seconds rather than a second write of the pixels.

    Every stage skips itself when its output already exists, so re-running
    is cheap and is how you add work to a finished job: an existing store is
    not re-ingested, existing cores are not re-detected, and an existing core
    crop whose box has not moved is not re-cut. Adding ``--export-cores`` to
    an already-dearrayed job therefore costs only the crops.

    Args:
        image_dir: Directory of ``.qptiff`` scans (or one file).
        job_dir: Output directory for the ``.zarr`` store(s) + marker map.
        level: Pyramid level to ingest (0 = full resolution).
        mpp: Microns/pixel overriding the file's own.
        mpp_csv: Optional CSV of per-image ``image,mpp`` overrides.
        channel_names: Optional file of marker names overriding the file's.
        subset: Optional YAML narrowing scans and/or channels.
        nuclear_marker: Force the nuclear channel by marker name.
        dearray: The scan is a TMA; detect its cores.
        export_cores: Also write each core as its own store under
            ``<job-dir>/cores/<slide>/``. Implies ``dearray``.
        core_levels: ``0`` or ``all`` — pyramid levels per exported core.
        dearray_conf: Detector confidence floor.
        cores_from: Directory of ``<slide>.geojson`` boxes to use rather
            than detecting.

    Example:
        Ingest one scan at full resolution::

            coral ingest-wsi --image-dir ./scans --job-dir ./processed
    """
    setup_logging()

    if not image_dir.exists():
        logger.error("--image-dir not found: %s", image_dir)
        raise typer.Exit(code=1)
    all_inputs = _collect_inputs(image_dir)
    if not all_inputs:
        logger.error("no .qptiff file(s) found at %s", image_dir)
        raise typer.Exit(code=1)

    subset_obj: Subset | None = None
    if subset is not None:
        try:
            subset_obj = Subset.from_yaml(subset)
        except (FileNotFoundError, ValidationError) as exc:
            logger.error("could not read --subset %s: %s", subset, exc)
            raise typer.Exit(code=1) from exc

    img_include = (subset_obj.images.include or None) if subset_obj else None
    img_exclude = (subset_obj.images.exclude or None) if subset_obj else None
    # A mistyped name would otherwise just silently shrink the cohort, and
    # this command's runs are long enough that a typo should surface now.
    _warn_unmatched_filters(all_inputs, img_include, img_exclude)
    selected = _filter_inputs(all_inputs, img_include, img_exclude)
    excluded = [p for p in all_inputs if p not in set(selected)]
    if not selected:
        logger.error(
            "no scans left in %s after the --subset images filter.", image_dir
        )
        raise typer.Exit(code=1)

    logger.info("📥 Running coral ingest-wsi on %d scan(s):", len(selected))
    for p in selected:
        logger.info("    + %s", p.name)
    if excluded:
        logger.info("  Excluded by --subset (images): %d", len(excluded))
        for p in excluded:
            logger.info("    - %s", p.name)
    logger.info("  Image dir:  %s", image_dir)
    logger.info("  Job dir:    %s", job_dir)
    if subset_obj is not None:
        logger.info(
            "  Subset:     %s (copied to %s)",
            subset,
            job_dir / "subset.yaml",
        )
        logger.info(
            "    Channels: include [%s]  exclude [%s]",
            ", ".join(subset_obj.channels.include) or "all",
            ", ".join(subset_obj.channels.exclude) or "none",
        )
        logger.info(
            "    Images:   include [%s]  exclude [%s]",
            ", ".join(subset_obj.images.include) or "all",
            ", ".join(subset_obj.images.exclude) or "none",
        )
    logger.info("  Pyramid level:    %d", level)

    try:
        for p in selected:
            channels, source_mpp, meta = read_qptiff_metadata(p, level=level)
            # One fact per line. This used to be a single wrapped line that
            # no one could scan, and it is the summary a user checks before
            # committing to a 40-minute run.
            logger.info("  %s", p.name)
            logger.info(
                "      slide id:      %s", meta["slide_id"] or "(none)"
            )
            logger.info("      channels:      %d", len(channels))
            logger.info(
                "      names from:    <%s>", meta["channel_name_source"]
            )
            if meta["sample_is_tma"] is not None:
                logger.info(
                    "      is a TMA:      %s (scan profile)",
                    meta["sample_is_tma"],
                )
            logger.info("      level %d shape: %s", level, meta["shape"])
            logger.info(
                "      pyramid:       %d level(s)", meta["pyramid_levels"]
            )
            logger.info(
                "      mpp:           %s",
                f"{source_mpp:.4g} um/px" if source_mpp else "(none in file)",
            )
            if mpp is not None:
                logger.info("      mpp override:  %.4g um/px (--mpp)", mpp)
            elif source_mpp is None:
                logger.error(
                    "  %s carries no mpp; pass --mpp or --mpp-csv", p.name
                )
                raise typer.Exit(code=1)
    except ReaderError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc

    if core_levels.strip().lower() not in {"0", "all"}:
        logger.error("--core-levels takes '0' or 'all', not %r.", core_levels)
        raise typer.Exit(code=1)
    all_levels = core_levels.strip().lower() == "all"

    mpp_map = _parse_metadata_csv(mpp_csv) if mpp_csv else None
    names = _parse_channel_names(channel_names) if channel_names else None

    # Refused before a single pixel is written, like every other check here,
    # and with no override flag. If a scan profile says false about a real
    # TMA the scanner is wrong, and that is worth fixing at the source rather
    # than carrying a flag on every run to talk past it.
    #
    # Only an explicit false counts, and this asymmetry is the point: the
    # ABSENCE of SampleIsTMA is not a claim that a slide IS a TMA. It is no
    # claim at all. TMA_1 carries no such element and is a TMA; older Vectra
    # writers and every non-PerkinElmer source carry none either. Only
    # SampleIsTMA=false is a positive statement, that the operator said at
    # acquisition this slide is one whole tissue section. Nothing here should
    # ever grow a branch that reads a missing element as a TMA.
    if dearray or export_cores:
        sections = [
            p.name
            for p in selected
            if read_qptiff_metadata(p, level=level)[2]["sample_is_tma"]
            is False
        ]
        if sections:
            logger.error(
                "%s: the scan profile says SampleIsTMA=false, so this is a "
                "single tissue section and has no cores to find. Detecting "
                "anyway returns tissue blobs that look like cores%s. Drop "
                "%s.",
                ", ".join(sections),
                " and exports them as if they were cores"
                if export_cores
                else "",
                "--export-cores"
                if export_cores and not dearray
                else "--dearray",
            )
            raise typer.Exit(code=1)

    # Matched before a single pixel is written. A supplied-cores run that
    # discovers halfway through that it has nothing for TMA_2 has already
    # spent forty minutes, and the fix is a file the user has on hand.
    supplied: dict[Path, Path] = {}
    if cores_from is not None:
        if not cores_from.is_dir():
            logger.error(
                "--cores-from must be a directory holding one "
                "<slide>.geojson per scan, not %s. ingest-wsi ingests a "
                "whole directory of scans, so a single file would attach "
                "one slide's boxes to all of them.",
                cores_from,
            )
            raise typer.Exit(code=1)
        missing = []
        for item in selected:
            found = match_cores_file(cores_from, wsi_output_stem(item))
            if found is None:
                missing.append(wsi_output_stem(item))
            else:
                supplied[item] = found
        if missing:
            logger.error(
                "--cores-from %s has no cores for %d scan(s): %s. Expected "
                "<name>.geojson beside the others.",
                cores_from,
                len(missing),
                ", ".join(missing),
            )
            raise typer.Exit(code=1)
        logger.info("  Cores from: %s (%d matched)", cores_from, len(supplied))

    args = {
        "image_dir": image_dir,
        "job_dir": job_dir,
        "level": level,
        "mpp": mpp,
        "mpp_csv": mpp_csv,
        "channel_names": channel_names,
        "subset": subset,
        "nuclear_marker": nuclear_marker,
        "dearray": dearray,
        "export_cores": export_cores,
        "core_levels": core_levels,
        "dearray_conf": dearray_conf,
        "cores_from": cores_from,
    }
    with run_ledger(job_dir, tool="coral ingest-wsi", args=args):
        if subset is not None:
            # Persisted next to the marker map, so a run's scan + channel
            # selection is recoverable from the job dir alone.
            job_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                job_dir / "subset.yaml", Path(subset).read_text()
            )
        try:
            resolution = _resolve(
                selected,
                job_dir,
                names,
                nuclear_marker,
                channels=subset_obj.channels if subset_obj else None,
                export_cores=export_cores,
            )
        except (MarkerMapError, ReaderError, ValueError) as exc:
            logger.error("%s", exc)
            raise typer.Exit(code=1) from exc

        start = time.perf_counter()
        logger.info("Ingesting %d scan(s)...", len(selected))
        failures: list[str] = []
        not_dearrayed: list[str] = []
        for i, item in enumerate(selected, start=1):
            # Announced BEFORE the work, not after. A user waiting on a
            # 40-minute slide needs to know which one is running, and a
            # line that only appears on success says nothing while it runs.
            logger.info("[%d/%d] %s", i, len(selected), item.name)
            try:
                slide = convert_wsi_to_canonical(
                    item,
                    job_dir,
                    resolution=resolution,
                    level=level,
                    mpp=mpp,
                    mpp_map=mpp_map,
                    channel_names=names,
                    nuclear_marker=nuclear_marker,
                )
            except Exception as exc:  # noqa: BLE001 — one bad scan, not all
                failures.append(item.name)
                logger.error("[%d/%d] %s FAILED", i, len(selected), item.name)
                logger.error("  %s", exc)
                continue

            # --export-cores without --dearray would be a flag that silently
            # does nothing, so it turns detection on instead.
            if not (dearray or export_cores or cores_from):
                continue
            # Deliberately does NOT fail the ingest. The pixels are written
            # and the store is valid, so losing a 40-minute write because
            # torch is missing or one slide is a whole section rather than a
            # TMA would be the wrong trade. The failure is recorded on the
            # slide's dearray task, named in the summary below, and the
            # command still exits non-zero only for ingest failures.
            try:
                result = dearray_slide(
                    slide,
                    conf=dearray_conf,
                    from_geojson=supplied.get(item),
                    export=export_cores,
                    export_root=job_dir / "cores",
                    export_all_levels=all_levels,
                )
                # No square brackets: the rich log handler reads them as
                # style markup and silently eats the method name.
                logger.info(
                    "  %d core(s) from %s%s",
                    result.n_cores,
                    result.method,
                    f" -> {result.exported} exported" if export_cores else "",
                )
            except Exception as exc:  # noqa: BLE001 — see above
                not_dearrayed.append(item.name)
                logger.error("  cores not detected: %s", exc)

        n_ok = len(selected) - len(failures)
        if n_ok:
            logger.info(
                "✅ Done! Ingested %d/%d scan(s) in %s",
                n_ok,
                len(selected),
                fmt_duration(time.perf_counter() - start),
            )
            for store in (
                job_dir / f"{wsi_output_stem(p)}.zarr" for p in selected
            ):
                if store.exists():
                    logger.info("  %s", store)
            log_check_progress(job_dir)
        if not_dearrayed:
            logger.warning(
                "%d scan(s) ingested but not dearrayed: %s. The stores are "
                "valid; run `coral ingest-wsi` again, or dearray them later, "
                "once the cause above is fixed.",
                len(not_dearrayed),
                not_dearrayed,
            )
        if failures:
            logger.error("%d scan(s) failed: %s", len(failures), failures)
            raise typer.Exit(code=1)


def _resolve(
    selected: list[Path],
    job_dir: Path,
    names: list[str] | None,
    nuclear_marker: str | None,
    channels: Any = None,  # noqa: ANN401 — a Selection or None
    export_cores: bool = False,
) -> Resolution:
    """Resolve the cohort's marker names from the qptiff panel.

    A qptiff carries its own marker names, so they are read here and
    handed to the shared cohort resolver as an explicit name list — the
    same path a user takes with ``--channel-names``.
    """
    if names is None:
        # Checked before the marker map is built, not after. Fluorophore names
        # written into marker_map.csv would then have to be un-picked by hand.
        for item in selected:
            _, _, meta = read_qptiff_metadata(item)
            if meta.get("channel_name_source") == "Name":
                raise ValueError(
                    f"{item.name} carries no <Biomarker> element, so its "
                    f"channels are named after fluorophores (DAPI, ATTO 550, "
                    f"Cy5) rather than antibody targets. Pass --channel-names "
                    f"with a text file of the real markers, one per line in "
                    f"channel order."
                )
    panel = names or qptiff_marker_names(selected[0])
    for other in selected[1:]:
        other_panel = qptiff_marker_names(other)
        if other_panel != panel:
            raise ValueError(
                f"{other.name} has a different marker panel from "
                f"{selected[0].name}; ingest them into separate job dirs, "
                f"or pass --channel-names to force one panel."
            )
    resolution, n_review = resolve_cohort_markers(
        selected,
        job_dir,
        channel_names=panel,
        channels=channels,
        nuclear_marker=nuclear_marker,
    )
    if n_review:
        review = [
            raw
            for raw, (_, level, _) in resolution.items()
            if level == REVIEW_TOKEN
        ]
        # Refused, not warned about. This is where `ingest-wsi` deliberately
        # diverges from `coral ingest`, and the reason is the cost asymmetry:
        # this command writes tens of gigabytes over tens of minutes, and
        # every one of those bytes would land in a store whose `channels`,
        # OME-XML and omero labels are blank. The check is free, because the
        # marker map is built before a single pixel is read.
        #
        # An exported core is worse still and cannot recover at all: the
        # guardrail re-syncs the stores directly inside the job dir when the
        # map is edited, and cores live one level down in cores/<slide>/, so
        # blanks written into a core stay there, in the one artefact designed
        # to leave the job dir.
        #
        # Named, but capped. A 29-marker panel with every row unresolved
        # printed all 28 names and pushed the actionable sentence off the
        # screen, which is the opposite of what naming them is for.
        shown = ", ".join(review[:_MAX_NAMED_MARKERS])
        if len(review) > _MAX_NAMED_MARKERS:
            shown += f", and {len(review) - _MAX_NAMED_MARKERS} more"
        target = "cores cannot be exported" if export_cores else "nothing is"
        tail = (
            "an exported core carries its panel with it and cannot be "
            "re-synced afterwards"
            if export_cores
            else "a store written now would carry blank channel names into "
            "its attrs, its OME-XML and its omero labels"
        )
        raise ValueError(
            f"{n_review} marker(s) still need review, so {target} "
            f"ingested: {shown}. Resolve them in "
            f"{job_dir / 'marker_map.csv'} (give each a canonical name, or "
            f"set its status to NOVEL), then run again. The map has been "
            f"written, so this is one edit away. Refusing costs nothing "
            f"here: no pixels have been read yet, and {tail}."
        )
    return resolution
