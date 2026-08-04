"""Stable identity for a core, independent of where it sits on the slide.

``core_index`` is a POSITION. :func:`prepare_review_boxes` sorts row-major on
every save, so drawing one core in the middle of a TMA shifts the index of
every core after it. Naming anything after that index means an insertion
renames fifty cores, and everything computed from them, for a change that
touched one.

``core_uid`` is an IDENTITY. It is minted the first time a box appears and
carried forward for as long as a box in the same place keeps appearing. It
never changes because a neighbour was added, removed, or resorted.

The two live side by side in ``cores.geojson``: ``core_uid`` says which core
this is, ``core_index`` says where it currently sits. Anything durable, and
the exported core store above all, keys off the uid; only display uses the
index.

**Why matching, rather than threading ids through the boxes.** Boxes reach
:func:`coral.dearray.outputs.write_dearray` as plain tuples, and the function
that clips, dedupes and orders them lives in the vendored ``_carta`` tree,
which CORAL does not modify. So uids are re-attached afterwards by comparing
the boxes about to be written against the ones already on disk. A box that is
recognisably the same core keeps its uid; anything else is new.

A box that has MOVED keeps its uid too, as long as it still overlaps where it
was. That is deliberate: nudging a core's edge is correcting the same core,
not replacing it, so its exported store keeps its identity and only its pixels
are re-cut. A box dragged somewhere else entirely is a different core and gets
a new uid, which is also right.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from coral.dearray._carta.detect import Box
from coral.dearray.paths import CORES_FILE, dearray_dir

__all__ = ["UID_KEY", "carry_uids", "ensure_uids", "new_uid"]

#: Feature property holding the identity. Read by the exporter.
UID_KEY = "core_uid"

#: How much two boxes must overlap to be the same core across a save. Chosen
#: to survive a correction that resizes a core substantially while refusing to
#: pair up two different cores: TMA cores are laid out on a grid with visible
#: gaps, so neighbours do not overlap at all.
_SAME_CORE_IOU = 0.3


def new_uid() -> str:
    """A fresh identity. Short enough to read in a directory listing."""
    return uuid.uuid4().hex[:8]


def _iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if overlap <= 0:
        return 0.0
    union = (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    )
    return overlap / union if union > 0 else 0.0


def _features(store: Path, method: str) -> list[dict[str, Any]]:
    """The features already written for a method, or none."""
    path = dearray_dir(store, method) / CORES_FILE
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(doc.get("features") or [])


def _box_of(feature: dict[str, Any]) -> Box | None:
    """A feature's bounding box, whatever shape its ring came back as."""
    rings = (feature.get("geometry") or {}).get("coordinates") or []
    if not rings:
        return None
    xs = [float(p[0]) for p in rings[0]]
    ys = [float(p[1]) for p in rings[0]]
    return (min(xs), min(ys), max(xs), max(ys))


def carry_uids(
    store: Path,
    method: str,
    boxes: Sequence[Box],
    *,
    inherit_from: str | None = None,
) -> list[str]:
    """The identity of each box about to be written.

    Args:
        store: Slide ``.zarr`` directory.
        method: Method being written.
        boxes: The boxes, in the order they will be written.
        inherit_from: A second method to inherit from when ``method`` has
            nothing yet. The first correction of a slide should keep the
            detector's identities rather than mint a whole new set, or every
            core would be re-cut the first time anybody moved one box.

    Returns:
        One uid per box, in the same order.
    """
    previous = _features(store, method)
    if not previous and inherit_from:
        previous = _features(store, inherit_from)

    known: list[tuple[Box, str]] = []
    for feature in previous:
        box = _box_of(feature)
        uid = (feature.get("properties") or {}).get(UID_KEY)
        if box is not None and uid:
            known.append((box, str(uid)))

    taken: set[int] = set()
    uids: list[str] = []
    for box in boxes:
        best, score = -1, _SAME_CORE_IOU
        for i, (old, _) in enumerate(known):
            if i in taken:
                continue
            overlap = _iou(box, old)
            if overlap > score:
                best, score = i, overlap
        if best < 0:
            uids.append(new_uid())
        else:
            taken.add(best)
            uids.append(known[best][1])
    return uids


def ensure_uids(store: Path, method: str) -> list[str]:
    """The uids a method's cores carry, minting and SAVING any that are absent.

    Written back rather than returned and forgotten. A uid invented fresh on
    every read would differ between two exports, which would rename every core
    store each time and defeat the entire point of having an identity. A store
    written before identities existed therefore gains them the first time
    anything asks, once, and is stable from then on.

    Args:
        store: Slide ``.zarr`` directory.
        method: Method whose cores to read.

    Returns:
        One uid per core, in ``core_index`` order.
    """
    path = dearray_dir(store, method) / CORES_FILE
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    features = sorted(
        doc.get("features") or [],
        key=lambda f: (f.get("properties") or {}).get("core_index", 0),
    )
    uids: list[str] = []
    minted = 0
    for feature in features:
        props = feature.setdefault("properties", {})
        uid = props.get(UID_KEY)
        if not uid:
            uid = new_uid()
            props[UID_KEY] = uid
            minted += 1
        uids.append(str(uid))

    if minted:
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return uids
