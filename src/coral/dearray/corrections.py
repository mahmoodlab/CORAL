"""Cores supplied by a human: finding the file, and reading its boxes.

A core box the detector got wrong is fixed in QuPath and handed back as
GeoJSON. So are boxes that never came from the detector at all, drawn by hand
or produced by someone's own script. CORAL does not distinguish between those:
they are boxes a human is responsible for, and they land under the
``imported`` method.

**Why there is no edit detection here.** An earlier version wrote imported and
detected cores into the same directory, which meant a re-run could destroy an
afternoon of dragging boxes, which in turn needed a geometry hash to notice
and a ``--force`` flag to override. All of that was managing a collision that
only existed because of the shared directory. Imported cores now have their
own method directory, so detection physically cannot overwrite them, and the
hash, the refusal and the flag are gone. :mod:`coral.tissue` reached the same
answer first: a human-supplied mask is a different method, not a contested
copy of the same one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from coral.dearray._carta.detect import Box, prepare_review_boxes
from coral.dearray.outputs import read_cores
from coral.dearray.paths import CORES_FILE, dearray_dir

logger = logging.getLogger(__name__)

__all__ = ["existing_cores", "load_corrected_boxes", "match_cores_file"]

#: Extensions accepted for a supplied cores file, in preference order.
_SUFFIXES = (".geojson", ".json")


def match_cores_file(directory: Path, slide_stem: str) -> Path | None:
    """The supplied cores file for one slide, matched by name.

    ``--cores-from`` names a directory rather than a file because
    ``ingest-wsi`` ingests a whole directory of scans. Taking one file would
    mean applying one person's boxes for ``TMA_1`` to ``TMA_2`` as well, which
    is how an earlier version behaved and is a silent way to attach the wrong
    geometry to a slide.

    Matching is on the slide's stem: ``TMA_1.qptiff`` takes ``TMA_1.geojson``.

    Args:
        directory: The ``--cores-from`` directory.
        slide_stem: Slide store stem, e.g. ``TMA_1``.

    Returns:
        The matching file, or ``None`` if the slide has none.
    """
    for suffix in _SUFFIXES:
        candidate = directory / f"{slide_stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_corrected_boxes(
    path: Path, *, width: int, height: int, detection_count: int | None = None
) -> list[Box]:
    """Boxes from a supplied GeoJSON, clipped, deduped and sorted.

    Accepts any polygon and takes its bounding box, so a QuPath export whose
    rectangles came back as four points in a different order, or reshaped
    slightly, still reads. Features classified as something other than ``Core``
    are ignored, which is how CARTA's own reader behaves: a QuPath project
    often carries other annotations that are not cores.

    Args:
        path: The supplied ``.geojson``.
        width: Level-0 width, for clipping.
        height: Level-0 height.
        detection_count: What detection originally found, so a suspicious
            explosion in box count can be reported.

    Returns:
        Boxes in level-0 coordinates, in row-major order.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If it holds no usable ``Core`` polygons.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    raw: list[Box] = []
    skipped = 0
    for feature in doc.get("features", []):
        klass = (feature.get("properties") or {}).get("classification") or {}
        name = str(klass.get("name", "Core"))
        if name.lower() != "core":
            skipped += 1
            continue
        rings = (feature.get("geometry") or {}).get("coordinates") or []
        if not rings:
            continue
        xs = [float(pt[0]) for pt in rings[0]]
        ys = [float(pt[1]) for pt in rings[0]]
        raw.append((min(xs), min(ys), max(xs), max(ys)))

    if skipped:
        logger.info("ignored %d annotation(s) not classified as Core", skipped)
    if not raw:
        raise ValueError(
            f"{path} holds no polygons classified as Core. In QuPath, the "
            f"core annotations must carry the Core classification, and the "
            f"objects have to be exported as GeoJSON: saving the .qpproj "
            f"alone is not enough."
        )
    return prepare_review_boxes(
        raw, w=width, h=height, detection_count=detection_count
    )


def existing_cores(store: Path, method: str) -> list[Box]:
    """The boxes already on disk, for an export that is not re-detecting."""
    return read_cores(store, method)


def has_cores(store: Path, method: str) -> bool:
    """Whether this method has already written a ``cores.geojson``."""
    return (dearray_dir(store, method) / CORES_FILE).is_file()
