"""YOLO core-box detection, and the box bookkeeping around it.

Vendored from CARTA ``segmenter/e2e_pipeline.py``, which is CARTA's real dearray
path. Copied verbatim apart from the removals noted below. No algorithm here is
changed.

A core is a **box**, and every coordinate is in the slide's native
full-resolution (level-0) space. The detector runs on a 1280 canvas built by
:mod:`coral.dearray._carta.detect_scale`, and boxes come back through
``remap_boxes_input_to_l0``.

Removals versus upstream:

* ``_run_detect``'s caller in upstream also writes core maps, bundles and
  provenance. None of that is vendored: CORAL writes its own outputs.
* ``_load_review_boxes`` keeps the clip, dedupe and sort but drops upstream's
  ``print`` diagnostics, since CORAL logs through its own logger. The
  ``detection_count`` sanity warning is kept, as the condition it catches is a
  real QuPath re-save artifact.
* ``load_core_boxes_from_geojson`` is not vendored. It lives in CARTA's
  ``geojson_annotations`` module and reads the ``Core`` classification; CORAL
  reads corrections through its own GeoJSON path and passes boxes in.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from coral.dearray._carta.detect_scale import INPUT_SIZE

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]

DETECT_IOU = 0.7
DETECT_AGNOSTIC = True

__all__ = [
    "DETECT_AGNOSTIC",
    "DETECT_IOU",
    "Box",
    "boxes_to_geojson",
    "clip_box_native",
    "dedupe_boxes_native",
    "run_detect",
    "sort_boxes_row_major",
]


def run_detect(rgb: np.ndarray, weights: Path, conf: float) -> list[Box]:
    """Detect core boxes on the 1280 canvas. Returns canvas-space xyxy."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.predict(
        rgb,
        conf=conf,
        imgsz=INPUT_SIZE,
        iou=DETECT_IOU,
        agnostic_nms=DETECT_AGNOSTIC,
        verbose=False,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    return [tuple(map(float, b.xyxy[0].tolist())) for b in res.boxes]


def clip_box_native(box: Box, w: int, h: int) -> Box | None:
    """Clip to the slide, dropping anything that ends up smaller than 8 px."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def sort_boxes_row_major(
    boxes: Sequence[Box], *, row_tol_px: float | None = None
) -> list[Box]:
    """Return boxes in human reading order: top-to-bottom rows, left-to-right
    within row."""
    if not boxes:
        return []
    box_list = list(boxes)
    heights = [max(1.0, float(b[3] - b[1])) for b in box_list]
    tol = (
        float(row_tol_px)
        if row_tol_px is not None
        else max(1.0, float(np.median(heights)) * 0.6)
    )
    remaining = sorted(
        box_list, key=lambda b: (0.5 * (b[1] + b[3]), 0.5 * (b[0] + b[2]))
    )
    rows: list[list[Box]] = []
    for box in remaining:
        cy = 0.5 * (box[1] + box[3])
        if not rows:
            rows.append([box])
            continue
        row = rows[-1]
        row_cy = float(np.mean([0.5 * (b[1] + b[3]) for b in row]))
        if abs(cy - row_cy) <= tol:
            row.append(box)
        else:
            rows.append([box])
    ordered: list[Box] = []
    for row in rows:
        ordered.extend(
            sorted(row, key=lambda b: (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])))
        )
    return ordered


def dedupe_boxes_native(
    boxes: Sequence[Box], *, tol_px: float = 8.0
) -> list[Box]:
    """Drop duplicate/overlapping boxes that share the same centroid (QuPath
    re-save artifact)."""
    unique: list[Box] = []
    for box in boxes:
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        if any(
            abs(cx - 0.5 * (u[0] + u[2])) <= tol_px
            and abs(cy - 0.5 * (u[1] + u[3])) <= tol_px
            for u in unique
        ):
            continue
        unique.append(box)
    return unique


def _box_to_polygon(box: Box) -> list[list[float]]:
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def boxes_to_geojson(boxes_native: Sequence[Box], slide_id: str) -> dict:
    """Core boxes as a QuPath-readable FeatureCollection.

    The ``Core`` classification name and its colour are load-bearing: CARTA's
    reader filters on that class when a corrected file comes back, so a review
    round trip depends on them.
    """
    features = []
    for i, box in enumerate(boxes_native):
        features.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_box_to_polygon(box)],
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": "Core", "colorRGB": -16711936},
                    "core_index": i,
                    "slide_id": slide_id,
                    "coord_space": "native-full-resolution",
                    "source": "yolo",
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def prepare_review_boxes(
    boxes: Sequence[Box],
    *,
    w: int,
    h: int,
    detection_count: int | None = None,
) -> list[Box]:
    """Clip, dedupe and sort boxes that came back from a human.

    Upstream reads them from GeoJSON itself; CORAL reads corrections through its
    own path and passes the boxes in.
    """
    raw = [b for b in (clip_box_native(b, w, h) for b in boxes) if b is not None]
    out = sort_boxes_row_major(dedupe_boxes_native(raw))
    if detection_count is not None and len(out) > max(
        detection_count * 2, detection_count + 4
    ):
        logger.warning(
            "%d exported boxes became %d unique after dedupe (detection had %d); "
            "using the deduped set",
            len(raw),
            len(out),
            detection_count,
        )
    elif len(raw) > len(out):
        logger.info("deduped %d boxes to %d unique cores", len(raw), len(out))
    return out
