"""Writing a dearray result: cores, a QC overlay, and a run record.

Three files per method, following :mod:`coral.tissue`'s convention that
geometry is GeoJSON in level-0 pixel coordinates and lives beside a
picture of itself.

``cores.geojson`` is CARTA's own shape, produced by
:func:`coral.dearray._carta.detect.boxes_to_geojson`, so it opens in QuPath as
``Core`` annotations you can drag and hand back. The class name and colour are
load-bearing for that round trip, which is why this module does not build the
Features itself.

A half-written ``cores.geojson`` would make a slide look dearrayed when it
is not, so the caller records the task only after these files land.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from coral.dearray._carta.detect import Box, boxes_to_geojson
from coral.dearray._carta.detect_scale import remap_boxes_l0_to_input
from coral.dearray.identity import UID_KEY, carry_uids
from coral.dearray.paths import (
    CORES_FILE,
    DEFAULT_DEARRAY_METHOD,
    dearray_dir,
)

logger = logging.getLogger(__name__)

__all__ = ["OVERLAY_FILE", "RECORD_FILE", "read_cores", "write_dearray"]

OVERLAY_FILE = "overlay.png"
RECORD_FILE = "dearray.json"


def _write_overlay(
    path: Path, canvas_rgb: np.ndarray, boxes_canvas: Sequence[Box]
) -> None:
    """The detector canvas with its boxes drawn on, for QC.

    Drawn in canvas coordinates rather than level 0, because the canvas is
    what the detector actually saw. A box that looks wrong here is a
    detection problem; a box that looks wrong only after remapping is a
    coordinate problem. The two are worth telling apart.
    """
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(canvas_rgb)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes_canvas):
        draw.rectangle(
            [box[0], box[1], box[2], box[3]], outline=(0, 255, 0), width=2
        )
        draw.text((box[0] + 3, box[1] + 2), str(i), fill=(255, 255, 0))
    img.save(path)


def write_dearray(
    store: Path,
    method: str,
    *,
    slide_id: str,
    boxes_l0: Sequence[Box],
    record: dict[str, Any],
    canvas_rgb: np.ndarray | None = None,
    mpp: float | None = None,
    letterbox: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write ``cores.geojson``, ``dearray.json``, and an overlay if we can.

    Canvas coordinates for the overlay are derived here from ``boxes_l0``
    rather than passed in. An earlier signature took both lists and trusted
    the caller to keep them in the same order; the caller got it wrong and
    the overlay labelled cores with the wrong indices, while the GeoJSON was
    right. A QC picture that mislabels the thing it is meant to verify is
    worse than no picture, so the ordering is now impossible to get wrong.

    Args:
        store: Slide ``.zarr`` directory.
        method: Method key, ``carta`` or ``imported``.
        slide_id: Written into each Feature, so a GeoJSON that has travelled
            still says which slide it belongs to.
        boxes_l0: Cores in level-0 pixel coordinates, already sorted
            row-major. Their order becomes ``core_index``, so sorting has to
            happen before this.
        record: Run provenance: detector, weights, scales, timings, counts.
        canvas_rgb: The detector canvas, for the overlay. Omitted when there
            is no canvas, which is the case for imported cores: building one
            costs a full level-0 plane read purely to draw a QC picture of
            boxes a human already drew on the real image in QuPath.
        mpp: Level-0 microns per pixel the canvas was built with. Required
            with ``canvas_rgb``.
        letterbox: Letterbox metadata from the same canvas build. Required
            with ``canvas_rgb``.

    Returns:
        Store-relative paths of what was written, for the task's ``outputs``.
    """
    out = dearray_dir(store, method)
    out.mkdir(parents=True, exist_ok=True)

    geojson = boxes_to_geojson(boxes_l0, slide_id)
    # CARTA hardcodes source="yolo", because upstream only ever writes boxes it
    # detected. CORAL also writes boxes a human drew and regions from section
    # mode, and a Feature claiming "yolo" for one of those is a false
    # provenance claim about a scientific object. Stamped here, in CORAL's
    # writer, rather than by editing the vendored function.
    source = str(record.get("source", "yolo"))
    # Identity, alongside the position CARTA already writes. `core_index` is
    # where a core currently sits and changes whenever a neighbour is added or
    # removed; `core_uid` is which core it is and does not. The exporter names
    # stores by the uid, so drawing one box no longer re-cuts every core after
    # it. A first correction inherits the detector's uids rather than minting a
    # new set, or moving one box would orphan every core on the slide.
    uids = carry_uids(
        store,
        method,
        list(boxes_l0),
        inherit_from=(
            None
            if method == DEFAULT_DEARRAY_METHOD
            else DEFAULT_DEARRAY_METHOD
        ),
    )
    for feature, uid in zip(geojson["features"], uids, strict=True):
        feature["properties"]["source"] = source
        feature["properties"][UID_KEY] = uid
    (out / CORES_FILE).write_text(
        json.dumps(geojson, indent=2), encoding="utf-8"
    )

    if canvas_rgb is not None and mpp is not None and letterbox is not None:
        boxes_canvas = remap_boxes_l0_to_input(list(boxes_l0), mpp, letterbox)
        _write_overlay(out / OVERLAY_FILE, canvas_rgb, boxes_canvas)

    (out / RECORD_FILE).write_text(
        json.dumps({**record, "n_cores": len(boxes_l0)}, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "wrote %d cores to %s",
        len(boxes_l0),
        (out / CORES_FILE).relative_to(store),
    )
    rel = out.relative_to(store).as_posix()
    outputs = {
        "cores": f"{rel}/{CORES_FILE}",
        "record": f"{rel}/{RECORD_FILE}",
    }
    if (out / OVERLAY_FILE).is_file():
        outputs["overlay"] = f"{rel}/{OVERLAY_FILE}"
    return outputs


def read_cores(store: Path, method: str) -> list[Box]:
    """Core boxes from a written ``cores.geojson``, in ``core_index`` order.

    Reads the rectangle's bounding box rather than assuming vertex order, so a
    file that has been through QuPath and come back reshaped still reads.

    Args:
        store: Slide ``.zarr`` directory.
        method: Detector key.

    Returns:
        Boxes as ``(x0, y0, x1, y1)`` in level-0 pixel coordinates.

    Raises:
        FileNotFoundError: If the slide has no cores for this method.
    """
    path = dearray_dir(store, method) / CORES_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Detect this slide's cores with "
            f"`coral ingest-wsi --dearray`, or supply them with "
            f"--cores-from."
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    features = sorted(
        doc.get("features", []),
        key=lambda f: f.get("properties", {}).get("core_index", 0),
    )
    boxes: list[Box] = []
    for feature in features:
        rings = feature.get("geometry", {}).get("coordinates") or []
        if not rings:
            continue
        xs = [float(pt[0]) for pt in rings[0]]
        ys = [float(pt[1]) for pt in rings[0]]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes
