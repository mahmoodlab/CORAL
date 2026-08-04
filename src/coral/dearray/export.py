"""``--export-cores``: cut each core out as its own canonical store.

Not the default, and worth restating why. Nothing in the science path needs a
per-core store: patching inside a core is a set of offsets into the parent,
and the parent already carries the whole pyramid. A per-core store earns its
cost in exactly one case, handing a core to something that cannot see the
parent, which is interop rather than pipeline.

**A core store is a CORAL slide store, not a crop in a box.** It carries
everything ``coral ingest`` writes: the pixel arrays, ``channels``,
``nuclear_channel`` by name, ``mpp``, NGFF ``multiscales`` and ``omero``,
``OME/METADATA.ome.xml``, ``thumbnails/nuclear.png``, ``state.json`` and
``structure.txt``. QuPath and napari open a core the way they open a slide,
and pointing ``coral tissue`` or ``coral patch`` at a directory of cores works
with no special case anywhere downstream.

Core provenance (which slide, which box, which method) lives in
``state.meta``, not in the store attributes. That is the rule ``ingest-wsi``
already follows for its own qptiff provenance, and the reason is the same: the
attribute keys stay identical to what ``coral ingest`` produces, so nothing
reading a store has to know which command wrote it.

**Levels.** Level 0 alone by default, because a core is 2000-4000 px and about
100 MB compressed and opens instantly without a pyramid. ``--core-levels all``
copies every level the parent has, cropping each from the parent's own reduced
level rather than downsampling the core's level 0 a second time.

**An unresolved panel refuses.** Every other stage already does, through the
marker-map guardrail. Export is the one path that would write blank channel
names into a store and hand it to someone, and unlike a slide store a core
never recovers: the guardrail re-syncs the stores directly inside the job dir,
and cores live one level down.

**The cores directory is a job directory.** Point `coral tissue` or `coral
cells` at `<job-dir>/cores/<slide>` and it works, because each core is a
complete store. What was missing was the marker map: without one the guardrail
falls back to each store's own names, which is correct but leaves no way to
rename a marker or un-keep a channel across all 56 cores at once. The parent's
map is copied in at export, so the cores directory has exactly the shape
`coral ingest` produces.

**Incremental, and that is the only re-run control.** Each core store records
the geometry hash of the box it was cut from, so a re-export rewrites exactly
the cores whose box moved and skips the rest. There is no ``--force``: a core
whose box has not moved has nothing to re-cut, and one whose box has moved is
re-cut without being asked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from coral.dearray._carta.detect import Box
from coral.dearray.identity import ensure_uids
from coral.io.ingest import (
    _CHUNK_TILE,
    _measured_window,
    _write_ngff_external_attrs,
    _write_nuclear_thumbnail,
)
from coral.slide.core import CoralSlide
from coral.slide.state import SlideMeta, default_state, load_state, save_state
from coral.slide.structure import write_structure_map
from coral.utils.errors import CoralError
from coral.utils.progress import bar_note, channel_bar, writing_channel
from coral.utils.time import now_iso

logger = logging.getLogger(__name__)

#: The cohort's resolved panel, copied in so the cores directory is a job
#: directory in its own right rather than a bag of stores.
MARKER_MAP = "marker_map.csv"

__all__ = ["CoreExport", "box_sha256", "export_cores"]

#: Channel indices named in a refusal before it says "and N more".
_MAX_NAMED = 8


@dataclass(frozen=True)
class CoreExport:
    """What an export did."""

    written: list[str]
    skipped: list[str]
    seconds: float
    #: Core stores whose box no longer exists on the slide, and which were
    #: deleted. Reported rather than silent: this is the one thing an export
    #: removes.
    removed: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """One line for a log, saying what was actually done."""
        removed = f", {len(self.removed)} removed" if self.removed else ""
        return (
            f"{len(self.written)} written, {len(self.skipped)} unchanged"
            f"{removed} in {self.seconds:.1f}s"
        )


def box_sha256(box: Box) -> str:
    """A stable hash of one box, for telling whether a core needs re-cutting.

    Rounded to whole pixels, matching how
    :func:`coral.tissue.mask.geojson_geometry_sha256` normalises rings, so a
    cosmetic re-save cannot look like a moved box.
    """
    payload = json.dumps([int(round(v)) for v in box], separators=(",", ":"))
    return hashlib.sha256(payload.encode(), usedforsecurity=False).hexdigest()


def _core_dir(root: Path, slide_stem: str, uid: str) -> Path:
    """``<root>/<slide>/core_<uid>.zarr``.

    Named by IDENTITY, not by position. ``core_index`` is where a core
    currently sits and shifts whenever a neighbour is added or removed, so a
    store named after it would be renamed, and everything computed from it
    invalidated, by an edit that never touched that core. See
    :mod:`coral.dearray.identity`.

    Outside the slide's own ``.zarr``, deliberately. A core store is a sibling
    of the slide, not a part of it: nesting one canonical store inside another
    makes the parent's directory stop being a clean OME-Zarr, and anything that
    walks a job directory for stores would find cores masquerading as slides.
    """
    return Path(root) / slide_stem / f"core_{uid}.zarr"


def _ome_attrs(store: zarr.Group) -> dict[str, Any]:
    """A store's OME block, whichever namespace this zarr format put it in."""
    attrs = dict(store.attrs)
    ome = attrs.get("ome")
    return dict(ome) if isinstance(ome, dict) else attrs


def _parent_levels(slide: CoralSlide) -> list[str]:
    """The parent's pyramid dataset keys, level order, level 0 first.

    Read from ``multiscales`` rather than by sorting the group's arrays,
    because ``"10"`` sorts before ``"2"`` and a ten-level slide would
    silently come out in the wrong order.
    """
    multiscales = _ome_attrs(slide.store).get("multiscales") or []
    if multiscales:
        paths = [
            str(d["path"])
            for d in multiscales[0].get("datasets", [])
            if "path" in d
        ]
        if paths:
            return paths
    return ["0"]


def _parent_colors(slide: CoralSlide) -> dict[int, str]:
    """The parent's per-channel omero colours, so a core opens the same way."""
    omero = _ome_attrs(slide.store).get("omero") or {}
    entries = omero.get("channels") or []
    return {
        i: str(entry["color"])
        for i, entry in enumerate(entries)
        if entry.get("color")
    }


def _clipped_level_box(
    box: Box, base: tuple[int, int], level: tuple[int, int]
) -> tuple[int, int, int, int]:
    """One box in a reduced level's coordinates, clipped to that level.

    Scaled by the level's measured size ratio rather than by ``2 ** level``. A
    qptiff's reduced levels are usually but not always exact halves, and an
    assumed factor puts the crop a few pixels off at the deepest level, which
    is where it is least likely to be noticed.

    Clamped rather than trusted, because a box that came back from a human can
    name pixels outside the slide and a zarr slice would silently return a
    smaller array rather than complain.
    """
    fy, fx = level[0] / base[0], level[1] / base[1]
    x0, y0, x1, y1 = (
        int(round(box[0] * fx)),
        int(round(box[1] * fy)),
        int(round(box[2] * fx)),
        int(round(box[3] * fy)),
    )
    return max(0, x0), max(0, y0), min(level[1], x1), min(level[0], y1)


def _write_core_levels(
    root: zarr.Group,
    slide: CoralSlide,
    box: Box,
    *,
    level_keys: list[str],
    labels: list[str],
    name: str,
) -> list[tuple[int, int, int]]:
    """Crop this core out of each parent level into datasets ``0..N-1``.

    Each level is cropped from the parent's own reduced level, not downsampled
    from the core's level 0. The scanner already reduced these pixels, and
    reducing them a second time by a different method would make a core's
    pyramid disagree with the slide's at the same zoom.

    Returns:
        The ``(c, y, x)`` shape of each dataset written, in level order.
    """
    written: list[tuple[int, int, int]] = []
    base = slide.store[level_keys[0]]
    base_hw = (int(base.shape[1]), int(base.shape[2]))
    n_channels = int(base.shape[0])

    for offset, key in enumerate(level_keys):
        source = slide.store[key]
        level_hw = (int(source.shape[1]), int(source.shape[2]))
        x0, y0, x1, y1 = _clipped_level_box(box, base_hw, level_hw)
        height, width = y1 - y0, x1 - x0
        if height <= 0 or width <= 0:
            # A deep level can shrink a small core to nothing. Stopping here
            # keeps the pyramid contiguous: a store with levels 0, 1, 3 is not
            # a pyramid any reader can walk.
            logger.debug(
                "%s: level %d is empty after scaling; pyramid stops at %d",
                name,
                offset,
                offset - 1,
            )
            break

        chunks = (1, min(_CHUNK_TILE, height), min(_CHUNK_TILE, width))
        shape = (n_channels, height, width)
        dataset = root.create_dataset(
            str(offset), shape=shape, dtype=source.dtype, chunks=chunks
        )
        # Transient and indented under the cores bar. leave=False is what
        # keeps fifty-six cores times N levels from leaving a wall of
        # finished bars behind; only the cores bar survives.
        with channel_bar(
            desc=f"    level {offset}",
            total=n_channels,
            leave=False,
            position=1,
        ) as bar:
            # Channel at a time, as the ingest streams, so peak memory is one
            # core-sized plane rather than the whole crop.
            for c in range(n_channels):
                writing_channel(bar, labels[c])
                dataset[c] = source[c, y0:y1, x0:x1]
                bar.update(1)
        written.append(shape)
    return written


def _copy_marker_map(job_dir: Path, out_dir: Path) -> None:
    """Give the cores directory the panel its cores were cut with.

    Without it every stage command run against the cores warns and falls back
    to the names inside each store. Those names are right, so nothing is
    broken, but there is then no single place to rename a marker or un-keep a
    channel for all of them.

    **An existing copy is never overwritten.** If it differs from the parent's,
    someone resolved something in the cores directory and re-exporting is not a
    reason to discard that. The difference is reported instead, because two
    maps for one panel silently drifting apart is the failure worth catching.
    """
    src = Path(job_dir) / MARKER_MAP
    dst = Path(out_dir) / MARKER_MAP
    if not src.is_file():
        return
    if dst.is_file():
        if dst.read_bytes() != src.read_bytes():
            logger.warning(
                "%s differs from the one in %s. Keeping the copy that is "
                "here: a re-export is not a reason to discard a resolution "
                "made against the cores. Delete it to take the parent's.",
                dst,
                job_dir,
            )
        return
    shutil.copyfile(src, dst)
    logger.info("marker map -> %s", dst)


#: What `export_cores` itself writes into a core store. Anything else was put
#: there by a later stage, from pixels this export is about to replace.
_EXPORT_WRITES = frozenset(
    {"OME", "thumbnails", "state.json", "structure.txt"}
)


def _clear_derived(out: Path) -> None:
    """Remove results computed from the pixels this core is about to lose.

    An allowlist rather than a list of stages, so a stage added later cannot
    be forgotten here. Pyramid levels are named by digits and are overwritten
    in place; everything else that is not in :data:`_EXPORT_WRITES` is a
    downstream result and is removed.
    """
    if not out.is_dir():
        return
    for child in out.iterdir():
        if child.name in _EXPORT_WRITES or child.name.isdigit():
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _remove_orphans(folder: Path, uids: Sequence[str]) -> list[str]:
    """Delete core stores whose box no longer exists on the slide.

    A core is cut from a box. Delete the box and the store left behind is not
    an out-of-date core, it is a core that is not on the slide at all, and
    nothing else will ever clean it up: cutting only ever writes or skips, so
    without this an orphan sits in the cores directory forever, is listed by
    anything that scans the directory, and is handed to the next stage as if
    it were tissue somebody chose.

    Only ever removes stores this export could have written. A directory that
    is not `core_<uid>.zarr` is left alone, and an empty uid list removes
    nothing at all rather than emptying the folder.

    Args:
        folder: `<root>/<slide>`, the slide's core directory.
        uids: Identities the current boxes carry.

    Returns:
        Names of the stores removed.
    """
    if not uids or not folder.is_dir():
        return []
    keep = {f"core_{uid}.zarr" for uid in uids}
    gone: list[str] = []
    for child in sorted(folder.glob("core_*.zarr")):
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child, ignore_errors=True)
            gone.append(child.stem)
    if gone:
        logger.info(
            "removed %d core(s) whose box no longer exists: %s",
            len(gone),
            ", ".join(gone),
        )
    return gone


def _recorded_digest(out: Path) -> str | None:
    """The box hash a written core says it was cut from, if any."""
    try:
        return getattr(load_state(out).meta, "core_box_sha256", None)
    except Exception:  # noqa: BLE001 - a broken store is re-cut
        return None


def _channel_labels(markers: list[str], channels: list[Any]) -> list[str]:
    """Display names: resolved marker, else the raw name, else the index."""
    labels = []
    for i, marker in enumerate(markers):
        raw = ""
        if i < len(channels) and isinstance(channels[i], dict):
            raw = str(channels[i].get("raw") or "")
        labels.append(marker or raw or f"channel_{i}")
    return labels


def export_cores(
    slide: CoralSlide,
    boxes: list[Box],
    *,
    method: str,
    root: Path,
    all_levels: bool = False,
) -> CoreExport:
    """Cut each core out of the parent into its own canonical store.

    Args:
        slide: The parent store.
        boxes: Cores in level-0 coordinates, row-major. Index is ``core_NNN``.
        method: Method key the boxes came from, recorded in each core's state,
            so a core says whether a human or the detector defined it.
        root: Parent directory for the per-slide core folder, normally
            ``<job-dir>/cores``.
        all_levels: Copy every pyramid level the parent has. Level 0 only by
            default.

    Returns:
        A :class:`CoreExport`.

    Raises:
        CoralError: If any channel has no resolved marker name. Checked here
            as well as in the CLI, because a core store leaves the job dir and
            a blank panel travelling inside one is not recoverable.
    """
    started = time.perf_counter()
    written: list[str] = []
    skipped: list[str] = []
    # Identity per box, from the method's own cores.geojson. A store written
    # before identities existed gains them here, once and saved, so a second
    # export agrees with the first instead of renaming every core.
    uids = ensure_uids(slide.path, method)
    if len(uids) != len(boxes):
        from coral.dearray.identity import new_uid

        uids = (uids + [new_uid() for _ in boxes])[: len(boxes)]
    level_keys = _parent_levels(slide) if all_levels else ["0"]
    markers = slide.markers
    blank = [i for i, m in enumerate(markers) if not str(m).strip()]
    if blank:
        shown = ", ".join(str(i) for i in blank[:_MAX_NAMED])
        if len(blank) > _MAX_NAMED:
            shown += f", and {len(blank) - _MAX_NAMED} more"
        raise CoralError(
            f"{slide.path.name} has {len(blank)} channel(s) with no resolved "
            f"marker name (index {shown}), so its cores cannot be exported. "
            f"Resolve them in marker_map.csv and re-run: a core carries its "
            f"panel with it and cannot be re-synced afterwards."
        )
    channels = list(dict(slide.store.attrs).get("channels") or [])
    labels = _channel_labels(markers, channels)
    nuclear_idx = slide.nuclear_channel
    mpp = slide._mpp()  # noqa: SLF001 - the validated public-boundary reader

    out_dir = Path(root) / slide.path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_marker_map(slide.path.parent, out_dir)

    logger.info(
        "cutting %d core(s) at %d level(s) into %s",
        len(boxes),
        len(level_keys),
        out_dir,
    )
    # Always shown, including for a level-0-only export. Fifty-six cores is
    # minutes of work, and the previous version silenced every bar when there
    # was one level, which left the terminal saying nothing at all.
    with channel_bar(
        desc="  cores", total=len(boxes), unit="core", position=0
    ) as cores_bar:
        for index, box in enumerate(boxes):
            uid = uids[index]
            name = f"core_{uid}"
            out = _core_dir(root, slide.path.stem, uid)
            digest = box_sha256(box)
            bar_note(cores_bar, name)
            if out.is_dir() and _recorded_digest(out) == digest:
                skipped.append(name)
                cores_bar.update(1)
                continue
            # The pixels are about to change, so anything derived from the old
            # ones is now wrong. Nothing here deletes, and a stage records its
            # results in a directory rather than only in state.json, so a
            # tissue mask cut from the previous box would survive the re-cut,
            # be found by a scan, and be paired with pixels it does not
            # describe. Export writes the levels, OME, thumbnails, state.json
            # and structure.txt; everything else in a core store came from a
            # later stage and goes.
            _clear_derived(out)
            shapes = _export_one(
                slide,
                box,
                out=out,
                name=name,
                index=index,
                digest=digest,
                uid=uid,
                method=method,
                level_keys=level_keys,
                labels=labels,
                markers=markers,
                channels=channels,
                nuclear_idx=nuclear_idx,
                mpp=mpp,
            )
            if shapes is not None:
                written.append(name)
            cores_bar.update(1)

    removed = _remove_orphans(root / slide.path.stem, uids)
    result = CoreExport(
        written=written,
        skipped=skipped,
        removed=removed,
        seconds=time.perf_counter() - started,
    )
    logger.info("cores -> %s: %s", root / slide.path.stem, result.summary)
    return result


def _export_one(  # noqa: PLR0913 - one store's worth of metadata
    slide: CoralSlide,
    box: Box,
    *,
    out: Path,
    name: str,
    index: int,
    digest: str,
    uid: str,
    method: str,
    level_keys: list[str],
    labels: list[str],
    markers: list[str],
    channels: list[Any],
    nuclear_idx: int | None,
    mpp: float,
) -> list[tuple[int, int, int]] | None:
    """Write one core as a complete canonical store.

    Returns:
        The shape of each level written, or ``None`` if the box clipped away
        to nothing.
    """
    started = now_iso()
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(out), mode="w")
    shapes = _write_core_levels(
        root, slide, box, level_keys=level_keys, labels=labels, name=name
    )
    if not shapes:
        logger.warning("%s is empty after clipping, skipped", name)
        return None

    nuclear_name = markers[nuclear_idx] if nuclear_idx is not None else None
    # Exactly the attribute keys `coral ingest` writes, so nothing downstream
    # can tell a core from a slide by its attrs.
    root.attrs["channels"] = channels
    root.attrs["nuclear_channel"] = nuclear_name
    root.attrs["mpp"] = mpp
    root.attrs["source_pyramid_levels"] = len(shapes)

    # Measured from the coarsest level written: those pixels are already on
    # disk and small, so this costs a fraction of a second and saves every
    # reader from inventing its own contrast.
    coarsest = root[str(len(shapes) - 1)]
    windows = {}
    for i in range(int(shapes[0][0])):
        measured = _measured_window(np.asarray(coarsest[i]))
        if measured:
            windows[i] = measured
    _write_ngff_external_attrs(
        root,
        labels,
        mpp,
        nuclear_marker=nuclear_name,
        level_shapes=shapes,
        dtype=np.dtype(root["0"].dtype),
        windows=windows,
        channel_colors=_parent_colors(slide),
    )
    if nuclear_idx is not None:
        _write_nuclear_thumbnail(out, np.asarray(root["0"][nuclear_idx]))

    state = default_state(out, image_path=str(slide.path.resolve()))
    state.tasks.ingest.status = "completed"
    state.tasks.ingest.started_at = started
    state.tasks.ingest.completed_at = now_iso()
    state.tasks.ingest.outputs = {"image": "0"}
    state.meta = SlideMeta(
        dimensions=(shapes[0][1], shapes[0][2]),
        n_markers=len(labels),
        mpp=mpp,
        # Core provenance lives here rather than in .zattrs, so the store's
        # attribute keys stay identical to `coral ingest`. Same rule
        # ingest-wsi follows for its own qptiff provenance.
        source_format="core",
        # Position, for display and for ordering a listing.
        core_index=index,
        # Identity, and what this store is named after. Survives a neighbour
        # being added or removed; `core_index` does not.
        core_uid=uid,
        core_box=[int(round(v)) for v in box],
        core_box_sha256=digest,
        parent_slide=slide.path.name,
        detector=method,
    )
    save_state(out, state)
    write_structure_map(out)
    logger.debug(
        "  %s %dx%d px (%.2f x %.2f mm), %d level(s)",
        name,
        shapes[0][2],
        shapes[0][1],
        shapes[0][2] * mpp / 1000,
        shapes[0][1] * mpp / 1000,
        len(shapes),
    )
    return shapes
