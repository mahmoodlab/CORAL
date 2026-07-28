"""``coral extract`` — feature extraction on canonical slide stores."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

if TYPE_CHECKING:
    from coral.features.kronos2 import Kronos2Extractor
    from coral.processor import CohortResult

from coral.cli._cohort import expand_cohort_zarr_dir, run_marker_guardrail
from coral.cli._render import fmt_duration, log_check_progress
from coral.features import (
    EXTRACTOR_REGISTRY,
    extractor_cli_help,
    listed_extractors,
)
from coral.summary import run_ledger
from coral.utils import CoralError, locks, resolve_device, setup_logging

logger = logging.getLogger(__name__)


def _patch_slugs(slide_path: Path, only: str | None) -> list[str]:
    """Patch-set slugs present on the slide (or just ``only`` if given)."""
    pdir = slide_path / "patches"
    if not pdir.is_dir():
        return []
    slugs = sorted(
        c.name
        for c in pdir.iterdir()
        if c.is_dir() and (c / "config.json").exists()
    )
    if only is not None:
        return [only] if only in slugs else []
    return slugs


def _log_subset_selection(store: Path, channels: Any) -> None:  # noqa: ANN401
    """Log the resolved marker selection for a ``--subset`` run.

    The analysis panel is frozen at ingest, so resolving against any one
    store is representative of the cohort. Reports ``using N of M markers``
    plus the shorter of the excluded/included lists (long lists are omitted
    — the exact names live in each store's ``markers_used``). Fails fast and
    clearly if the selection is empty or drops the nuclear stain, rather
    than failing per-slide mid-run.
    """
    from coral.slide import CoralSlide

    slide = CoralSlide.open(store)
    kept = [slide.markers[i] for i in slide.kept_indices]
    try:
        _, used = slide._resolve_markers(channels)
    except ValueError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc
    excluded = [m for m in kept if m not in used]
    head = f"  subset: using {len(used)} of {len(kept)} markers"
    cap = 8
    if excluded and len(excluded) <= len(used) and len(excluded) <= cap:
        logger.info("%s (excluded: %s)", head, ", ".join(excluded))
    elif used and len(used) <= cap:
        logger.info("%s (only: %s)", head, ", ".join(used))
    else:
        logger.info("%s", head)


def extract(
    job_dir: Path = typer.Option(
        ...,
        "--job-dir",
        help="The ingest job directory: CORAL extracts features for every "
        "<name>.zarr store inside it and records the run in its ledger. "
        "Run `coral ingest`, `coral tissue`, and `coral patch` first.",
    ),
    extractor: str = typer.Option(
        "mean_marker",
        "--extractor",
        help=extractor_cli_help(),
    ),
    patches: str | None = typer.Option(
        None,
        "--patches",
        help="Patch-set slug to extract, e.g. 0.5mpp_256px. Default: every "
        "set present on each store.",
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        help="Patches encoded per forward pass. Default: 16. ",
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        help="Subprocesses reading patches ahead of the forward pass, so "
        "reads overlap the GPU instead of idling it. Gains plateau around "
        "4; higher mostly costs memory. Pass 0 to read inline. "
        "Default: 4. ",
    ),
    subset: str | None = typer.Option(
        None,
        "--subset",
        help="Path to a subset YAML selecting a marker subset by glob — the "
        "same schema as `coral ingest --subset`; only its `channels` apply "
        "here (`images` is ignored). Default: every kept marker. The subset "
        "filename names the output variant folder (markers_<stem>); without "
        "--subset it is markers_all.",
    ),
    additional_markers: Path | None = typer.Option(
        None,
        "--additional-markers",
        help="CSV declaring novel markers absent from KRONOS2's vocabulary "
        "(KRONOS2-class extractors only): Ignored (with a "
        "warning) for other extractors.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device to run on: auto (default), cpu, cuda, cuda:N (pick a "
        "GPU by index, e.g. cuda:1), or mps (Apple Silicon). Mutually "
        "exclusive with --gpu.",
    ),
    gpu: int | None = typer.Option(
        None,
        "--gpu",
        help="Shorthand for --device cuda:N (a CUDA GPU index, e.g. 0). "
        "Default: auto (cuda if available, else mps, else cpu).",
    ),
) -> None:
    r"""Encode stored patch sets into per-patch features.

    For each slide, runs ``extractor`` over every patch set under
    ``patches/`` (or just ``--patches <slug>``), writing
    ``features/<slug>/<extractor>/<variant>/``. Grid patches are read raw;
    cell-centered patches are isolated to their target cell. Markers
    default to all; ``--subset`` selects a subset. ``--batch-size`` sets how
    many patches are encoded per call (tune for GPU memory). Completed sets
    are skipped (delete the ``features/<slug>/<extractor>/<variant>`` folder
    or use a fresh ``--job-dir`` to re-extract). A failing slide is logged
    and skipped; the command exits non-zero if any failed. Concurrent runs
    over one job dir split the cohort collision-free via per-slide locks; a
    crashed run's stale locks are cleared automatically at startup.

    Args:
        job_dir: Ingest job directory of ``.zarr`` stores.
        extractor: Registered extractor name (see ``coral extract --help``).
        patches: Optional patch-set slug; default every set on each store.
        batch_size: Patches encoded per forward pass.
        num_workers: Loader subprocesses reading patches ahead of the
            forward pass (0 reads inline).
        subset: Optional YAML selecting a marker subset (``channels`` only);
            its filename names the output variant (``markers_<stem>``).
        additional_markers: Novel-marker CSV (KRONOS2-class extractors).
        device: Device (``auto``, ``cpu``, ``cuda``, ``cuda:N``, ``mps``).
        gpu: CUDA index shorthand (mutually exclusive with ``device``).

    Example:
        Encode every patch set with the default mean-marker extractor::

            coral extract --job-dir ./processed

        Or a foundation-model encoder on one patch set::

            coral extract --job-dir ./processed --extractor KRONOS2 \
                --patches 0.5mpp_256px --batch-size 16
    """
    from coral.config import PatchConfig
    from coral.config.subset import Subset, variant_name
    from coral.processor import CoralProcessor
    from coral.slide import CoralSlide

    setup_logging()

    if batch_size < 1:
        raise typer.BadParameter("--batch-size must be >= 1")

    if num_workers < 0:
        raise typer.BadParameter("--num-workers must be >= 0")

    extractor_cls = EXTRACTOR_REGISTRY.get(extractor)
    if extractor_cls is None:
        raise typer.BadParameter(
            f"unknown extractor {extractor!r}; have {listed_extractors()}"
        )
    channels = None
    variant = variant_name(None)
    if subset is not None:
        from pydantic import ValidationError

        try:
            subset_obj = Subset.from_yaml(subset)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except ValidationError as exc:
            raise typer.BadParameter(
                f"--subset {subset} is not a valid subset YAML — expected "
                f"'channels:' (and optionally 'images:') with include/exclude "
                f"glob lists."
            ) from exc
        if subset_obj.images.include or subset_obj.images.exclude:
            logger.warning(
                "--subset 'images' selection applies only at `coral ingest`; "
                "ignored here (extract processes every store in --job-dir)."
            )
        channels = subset_obj.channels
        try:
            variant = variant_name(subset)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    all_slides = expand_cohort_zarr_dir(job_dir)
    if not all_slides:
        logger.error(
            "no .zarr stores in %s — run coral ingest first.", job_dir
        )
        raise typer.Exit(code=1)

    if device is not None and gpu is not None:
        raise typer.BadParameter("pass one of --device / --gpu, not both")
    spec = (
        device
        if device is not None
        else (f"cuda:{gpu}" if gpu is not None else None)
    )
    try:
        device = resolve_device(spec)
    except CoralError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc

    args = {
        "job_dir": job_dir,
        "extractor": extractor,
        "subset": subset,
        "additional_markers": additional_markers,
        "variant": variant,
        "patches": patches,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "gpu": gpu,
    }

    total = len(all_slides)
    logger.info("Running coral extract on %d slide(s):", total)
    logger.info("  Job dir:    %s", job_dir)
    logger.info("  Extractor:    %s", extractor)
    logger.info(
        "  Subset:    %s", subset if subset is not None else "all markers"
    )
    logger.info("  Batch size:    %d", batch_size)
    logger.info("  Workers:    %d", num_workers)
    logger.info("  Device:    %s", device)

    run_marker_guardrail(job_dir, all_slides)
    if channels is not None:
        _log_subset_selection(all_slides[0], channels)

    start = time.perf_counter()
    with run_ledger(job_dir, tool="coral extract", args=args):
        # Build the extractor once before [1/N] so Hub download / weight
        # load is not charged to the first image's progress bar.
        logger.info("Loading extractor %s (device=%s)...", extractor, device)
        extractor_obj = extractor_cls.build(device=device)
        logger.info("Extractor ready.")
        patch_counts: list[int] = []

        def _extract_slide(item: Path) -> None:
            """Extract every requested patch set of one slide."""
            from pydantic import ValidationError

            slide = CoralSlide.open(item)
            slugs = _patch_slugs(item, patches)
            if not slugs:
                raise ValueError(
                    "no patch sets found; run `coral patch` first"
                )
            for slug in slugs:
                doc = json.loads(
                    (item / "patches" / slug / "config.json").read_text()
                )
                try:
                    config = PatchConfig.from_stored(doc)
                except ValidationError as exc:
                    msg = (
                        f"patch set {slug!r} has an unreadable config "
                        f"(store predates the current patch format); "
                        f"re-run `coral patch`."
                    )
                    raise ValueError(msg) from exc
                feats = slide.encode_features(
                    extractor_obj,
                    config,
                    channels=channels,
                    batch_size=batch_size,
                    suffix=variant,
                    num_workers=num_workers,
                )
                patch_counts.append(int(feats.shape[0]))

        stale = locks.clear_locks(job_dir, stale_only=True)
        if stale["removed"]:
            logger.info(
                "Reclaimed %d stale lock(s) from a previous run.",
                stale["removed"],
            )
        # Novel-marker prepare pass (KRONOS2-class extractors only):
        # computes/pools the cohort's novel-marker stats and registers them
        # before any extraction, so the z-score uses data-driven (mean, std)
        # instead of the model default. A no-op when nothing is novel; errors
        # (naming the marker) when novel markers lack `--additional-markers`.
        if extractor_obj.supports_novel_markers:
            from coral.features.prepare import prepare_cohort_stats

            try:
                prepare_cohort_stats(
                    cast("Kronos2Extractor", extractor_obj),
                    [CoralSlide.open(p) for p in all_slides],
                    additional_markers,
                    channels=channels,
                )
            except CoralError as exc:
                logger.error("%s", exc)
                raise typer.Exit(code=1) from exc
        elif additional_markers is not None:
            logger.warning(
                "--additional-markers is only used by KRONOS2-class "
                "extractors (novel-marker stats); ignored for %r.",
                extractor,
            )

        logger.info("")
        patch_scope = f"patch set {patches}" if patches else "all patch sets"
        logger.info(
            "Extracting %s features · %d slide(s) · %s",
            extractor,
            total,
            patch_scope,
        )
        # The cohort loop + per-slide lock / skip-done / error-tolerance
        # lives in CoralProcessor; concurrent `coral extract` runs over a
        # shared cohort split it collision-free via the .lock files.
        result = CoralProcessor(all_slides).run(_extract_slide)

        _log_footer(
            result,
            total,
            sum(patch_counts),
            job_dir,
            time.perf_counter() - start,
        )
        if result.failed:
            raise typer.Exit(code=1)


def _log_footer(
    result: CohortResult,
    n_total: int,
    n_patches: int,
    job_dir: Path,
    elapsed: float,
) -> None:
    """Completion summary + next-step hint (mirrors tissue/cell/patch).

    Reports the whole cohort accounting — extracted, skipped (locked by a
    concurrent run), and failed — with the total patches encoded and
    elapsed time. A per-slide store that was already extracted logs its
    own "skipping" line upstream.
    """
    n_ok = len(result.processed)
    logger.info("")
    if n_ok:
        logger.info(
            "Done! Features ready for %d/%d slide(s) (%d patches) in %s",
            n_ok,
            n_total,
            n_patches,
            fmt_duration(elapsed),
        )
        log_check_progress(job_dir)
    if result.skipped_locked:
        logger.info(
            "%d slide(s) skipped — locked by another run.",
            len(result.skipped_locked),
        )
    if result.failed:
        logger.error(
            "%d/%d slide(s) failed — see the errors above.",
            len(result.failed),
            n_total,
        )
