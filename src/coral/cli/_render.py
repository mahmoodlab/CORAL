"""Shared CLI rendering helpers for marker maps and progress bars.

One definition of the duration format, cohort/per-image tqdm helpers,
and the coloured marker-map print — so terminal output stays consistent
by construction.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from tqdm.std import tqdm

from coral.markers.marker_map import (
    NOVEL_TOKEN,
    RESOLVED_TOKEN,
    REVIEW_TOKEN,
    MarkerMapError,
)
from coral.utils.progress import bar_status, bar_write, per_image_bar
from coral.utils.time import fmt_duration

if TYPE_CHECKING:
    from coral.io.ingest import Resolution

logger = logging.getLogger(__name__)

__all__ = [
    "bar_status",
    "bar_write",
    "cohort_progress",
    "fmt_duration",
    "log_check_progress",
    "log_marker_map_error",
    "per_image_bar",
    "print_marker_mapping",
]

_T = TypeVar("_T")


def log_check_progress(job_dir: Path) -> None:
    """One-line hint after a successful stage: how to inspect the job."""
    logger.info("Check progress: coral status --job-dir %s", job_dir)


def cohort_progress(
    items: Sequence[_T],
    *,
    unit: str = "image",
    disable: bool | None = None,
) -> Iterator[tuple[_T, Any]]:
    """Yield ``(item, pbar)`` under one continuous leave=True cohort bar.

    For ingest: ``total=len(items)``; caller sets ``pbar.set_description``
    to the current image name and calls ``pbar.update(1)`` after that
    image's work finishes (including skip/fail).

    Args:
        items: Cohort members to iterate (e.g. image paths).
        unit: tqdm unit label (default ``image``).
        disable: Force-disable the bar; ``None`` lets tqdm decide.

    Yields:
        ``(item, pbar)`` pairs for the caller to update.

    Example:
        >>> items = ["a", "b"]
        >>> pairs = list(cohort_progress(items, unit="image", disable=True))
        >>> len(pairs) == 2 and pairs[0][0] == "a"
        True
    """
    pbar = tqdm(
        total=len(items),
        unit=unit,
        leave=True,
        disable=disable,
    )
    try:
        for item in items:
            yield item, pbar
    finally:
        pbar.close()


def print_marker_mapping(resolution: Resolution) -> None:
    """Print the full marker map, one row per marker, coloured by status.

    Green = mapped to a canonical registry name (or a NOVEL marker);
    red + ``REVIEW`` = still needs review. Backs ``--print-marker-map`` on
    ``coral ingest``.

    Args:
        resolution: Map of ``raw_name → (marker, status, keep)`` from ingest.

    Example:
        After ingest resolves a cohort map::

            from coral.cli._render import print_marker_mapping

            print_marker_mapping(resolution)
    """
    logger.info("")
    logger.info(
        "Marker mapping: original marker -> mapped marker in canonical "
        "registry"
    )
    for original, (mapped, level, _keep) in resolution.items():
        if level == REVIEW_TOKEN:
            # A review row has no mapped name yet — no empty `-->`.
            logger.info(
                "  [red]%s  REVIEW — needs a name[/red]",
                original,
                extra={"no_highlight": True},
            )
        else:
            logger.info(
                "  [green]%s --> %s[/green]",
                original,
                mapped,
                extra={"no_highlight": True},
            )


def log_marker_map_error(exc: MarkerMapError) -> None:
    """Render a :class:`MarkerMapError` as clear, multi-line, red output.

    One block per problem category — each offending value on its own red
    line, then exactly what the user must do about it. Shared so every
    command reports a broken map identically.

    Args:
        exc: The marker-map failure raised by the guardrail.

    Example:
        Catch a broken map and print the same red block every CLI uses::

            from coral.cli._render import log_marker_map_error
            from coral.markers.marker_map import MarkerMapError

            try:
                ...
            except MarkerMapError as exc:
                log_marker_map_error(exc)
    """
    if exc.invalid_names:
        logger.error(
            "the 'mapped_canonical_name' column in marker_map.csv has "
            "%d invalid value(s):",
            len(exc.invalid_names),
        )
        for original, value in exc.invalid_names:
            logger.error(
                "  [red]%s --> %s[/red]",
                original,
                value,
                extra={"no_highlight": True},
            )
        logger.error(
            "Each 'mapped_canonical_name' must be a registry marker name."
        )
        logger.error(
            "Genuinely new marker not in the canonical registry? set "
            "status='NOVEL' and fill 'mapped_canonical_name' with your "
            "custom name."
        )
    if exc.unresolved:
        logger.error(
            "%d marker(s) in marker_map.csv have no name yet in "
            "'mapped_canonical_name' column",
            len(exc.unresolved),
        )
        for original, status in exc.unresolved:
            # A blank name whose status the user already flipped to
            # RESOLVED/NOVEL is the common trap — call it out explicitly.
            detail = (
                f"(status={status}, but the name cell is empty)"
                if status in (RESOLVED_TOKEN, NOVEL_TOKEN)
                else f"(status={status})"
            )
            logger.error(
                "  [red]%s[/red]  %s",
                original,
                detail,
                extra={"no_highlight": True},
            )
        logger.error(
            "Fill 'mapped_canonical_name' with a registry marker name OR"
            " set status='NOVEL' and use your own name for a new marker OR"
            " re-run 'coral ingest' with a --subset that excludes the markers."
        )
