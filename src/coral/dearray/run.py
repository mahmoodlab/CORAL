"""``dearray_slide``: find a TMA's cores, or take the ones you were given.

Ties together the canvas, CARTA's detector, the importer and the exporter, and
records a ``dearray[<method>]`` task on the slide the way ``detect_tissue``
records ``tissue[<method>]``.

Detection is:

    level-0 nuclear plane -> 1280 canvas -> YOLO -> boxes back to level 0
        -> clip to the slide -> dedupe -> sort row-major -> write

Only the clip, dedupe and sort are worth a note, and they are CARTA's own,
in that order for a reason: clipping first stops an off-slide box from
surviving as a sliver, deduping before sorting keeps the QuPath re-save
artifact out of the numbering, and sorting last makes ``core_index``
reading order.

**Three ways in, and each skips what it does not need.**

``--cores-from`` writes to the ``imported`` method and never builds a canvas:
the slide's width, height and mpp are in the store's attrs, and reading a
full level-0 plane to rediscover them would cost about ten seconds to draw a
QC picture of boxes a human already checked in QuPath.

Detection writes to ``carta``, and is skipped outright when that method
already has a ``cores.geojson``. That is the same contract ingest already
keeps for its stores, and it is what makes ``--export-cores`` on an
already-dearrayed slide cost only the export.

Export cuts whichever cores are authoritative: ``imported`` if a human has
supplied any, otherwise the detector's.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from coral.dearray._carta.detect import (
    Box,
    clip_box_native,
    dedupe_boxes_native,
    run_detect,
    sort_boxes_row_major,
)
from coral.dearray._carta.detect_scale import (
    INPUT_SIZE,
    TARGET_UM_PER_PX,
    remap_boxes_input_to_l0,
)
from coral.dearray.canvas import build_detector_canvas
from coral.dearray.corrections import has_cores, load_corrected_boxes
from coral.dearray.export import export_cores
from coral.dearray.outputs import read_cores, write_dearray
from coral.dearray.paths import (
    DEFAULT_DEARRAY_METHOD,
    IMPORTED_METHOD,
    dearray_dir,
)
from coral.slide.core import CoralSlide
from coral.slide.state import TaskState
from coral.utils.errors import CoralError
from coral.utils.time import now_iso

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONF",
    "DearrayResult",
    "dearray_slide",
    "export_method_for",
]

#: CARTA's own detector confidence (``DearrayConfig.yolo_conf``).
DEFAULT_CONF = 0.25

_DEFAULT_REF = "hf_hub:MahmoodLab/CARTA"
_WEIGHTS_FILE = "carta_core.pt"
_INSTALL_MSG = (
    "TMA dearraying needs the optional `dearray` extra (ultralytics + torch). "
    "Install it with `uv sync --extra dearray` (or `pip install "
    "coral[dearray]`), or drop --dearray. Weights download automatically from "
    "the public Hugging Face repo MahmoodLab/CARTA, or point at a local "
    "checkpoint with CARTA_CORE_WEIGHTS."
)


@dataclass(frozen=True)
class DearrayResult:
    """What a dearray run found and where it put it."""

    method: str
    boxes: list[Box]
    outputs: dict[str, str]
    seconds: float
    detected: bool = True
    exported: int = 0

    @property
    def n_cores(self) -> int:
        """How many cores this slide has."""
        return len(self.boxes)


def export_method_for(
    store: Path, method: str = DEFAULT_DEARRAY_METHOD
) -> str:
    """Which method's cores are authoritative for this slide.

    A human's boxes outrank the detector's. Detection is left on disk so the
    two can be compared, but it is not what gets cut.

    Args:
        store: Slide ``.zarr`` directory.
        method: The detector method to fall back to.

    Returns:
        ``imported`` if a human has supplied cores, else ``method``.
    """
    return IMPORTED_METHOD if has_cores(store, IMPORTED_METHOD) else method


def _resolve_weights(ref: str = _DEFAULT_REF) -> Path:
    """The detector checkpoint, from a local path or the Hugging Face hub.

    Mirrors :class:`coral.tissue.carta.CartaTissueSegmenter`'s resolution
    rather than inventing a second mechanism: ``CARTA_CORE_WEIGHTS`` wins,
    then an ``hf_hub:`` ref, then a plain path.
    """
    local = os.environ.get("CARTA_CORE_WEIGHTS")
    if local:
        path = Path(local)
        if not path.is_file():
            raise FileNotFoundError(
                f"CARTA_CORE_WEIGHTS points at {path}, which is not a file."
            )
        return path
    if not ref.startswith("hf_hub:"):
        return Path(ref)
    repo_id = ref.split("hf_hub:", 1)[1]
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id, filename=_WEIGHTS_FILE, token=token
            )
        )
    except Exception as exc:  # noqa: BLE001 - surface an actionable hint
        raise RuntimeError(
            f"Could not fetch the CARTA core detector from hf_hub:{repo_id} "
            f"({_WEIGHTS_FILE}). The repo is public, so check the network, or "
            f"set CARTA_CORE_WEIGHTS to a local .pt."
        ) from exc


def _import_cores(
    slide: CoralSlide, from_geojson: Path, *, corrected_by: str | None
) -> tuple[list[Box], dict[str, object]]:
    """Read supplied boxes, using the store's attrs instead of a canvas."""
    height, width = (int(v) for v in slide.image.shape[-2:])
    boxes = load_corrected_boxes(
        Path(from_geojson),
        width=width,
        height=height,
        detection_count=None,
    )
    record: dict[str, object] = {
        "source": "imported",
        "imported_from": Path(from_geojson).name,
        "imported_by": corrected_by,
        "imported_at": now_iso(),
        "native_mpp": slide._mpp(),  # noqa: SLF001 - validated reader
        "slide_width": width,
        "slide_height": height,
    }
    logger.info(
        "imported %d core(s) from %s", len(boxes), Path(from_geojson).name
    )
    return boxes, record


def _detect_cores(
    slide: CoralSlide, *, conf: float, weights: str | Path, method: str
) -> tuple[list[Box], dict[str, object], object]:
    """Run CARTA's detector and map its boxes back to level 0."""
    canvas = build_detector_canvas(slide)
    record: dict[str, object] = {
        "detector": method,
        "detection_level": 0,
        "native_mpp": canvas.mpp,
        "nuclear_channel": canvas.nuclear_index,
        "target_um_per_px": TARGET_UM_PER_PX,
        "input_size": INPUT_SIZE,
        "letterbox_scale": canvas.letterbox["letterbox_scale"],
        "read_seconds": round(canvas.read_seconds, 2),
        "canvas_seconds": round(canvas.canvas_seconds, 2),
    }

    weights_path = _resolve_weights(str(weights))
    detect_started = time.perf_counter()
    try:
        canvas_boxes = run_detect(canvas.rgb, weights_path, conf)
    except ImportError as exc:  # ultralytics absent
        raise ImportError(_INSTALL_MSG) from exc
    detect_seconds = time.perf_counter() - detect_started

    # Level 0 straight out of the remap, because the canvas was built from the
    # slide's own level-0 mpp. Then CARTA's order: clip, dedupe, sort.
    on_slide = [
        box
        for box in (
            clip_box_native(b, canvas.width, canvas.height)
            for b in remap_boxes_input_to_l0(
                canvas_boxes, canvas.mpp, canvas.letterbox
            )
        )
        if box is not None
    ]
    boxes = sort_boxes_row_major(dedupe_boxes_native(on_slide))
    record.update(
        {
            "source": "yolo",
            "weights": weights_path.name,
            "conf": conf,
            "n_detected": len(canvas_boxes),
            "n_on_slide": len(on_slide),
            "detect_seconds": round(detect_seconds, 2),
        }
    )
    logger.info(
        "detected %d cores (%d raw, %d on the slide) in %.1f s",
        len(boxes),
        len(canvas_boxes),
        len(on_slide),
        detect_seconds,
    )
    if not boxes:
        logger.warning(
            "no cores detected on %s. Inspect %s to see what the detector "
            "saw. A single tissue section looks like this, and does not need "
            "dearraying at all.",
            slide.path.name,
            dearray_dir(slide.path, method).name,
        )
    return boxes, record, canvas


def dearray_slide(
    slide: CoralSlide,
    *,
    method: str = DEFAULT_DEARRAY_METHOD,
    conf: float = DEFAULT_CONF,
    weights: str | Path = _DEFAULT_REF,
    from_geojson: Path | None = None,
    export: bool = False,
    export_root: Path | None = None,
    export_all_levels: bool = False,
    corrected_by: str | None = None,
) -> DearrayResult:
    """Give a slide its cores, and optionally cut them out.

    Only a TMA is dearrayed. A whole slide that is a single tissue section is
    simply not passed here, which is why there is no section mode: writing one
    region covering the whole slide would be a core that is not a core.

    Args:
        slide: An open canonical store.
        method: Detector output key. Only ``carta`` exists. Supplied cores
            always go to ``imported`` regardless of this.
        conf: Detector confidence floor. CARTA's default is 0.25.
        weights: ``hf_hub:<repo>``, or a path to a local ``.pt``.
        from_geojson: Cores supplied for this slide. Detection is not run.
        export: Also cut each core out as its own store.
        export_all_levels: Give each exported core the parent's full pyramid
            rather than level 0 alone.
        export_root: Where those stores go, normally ``<job-dir>/cores``. A
            core store is a sibling of the slide, not a part of it, so it does
            not live inside the slide's own ``.zarr``.
        corrected_by: Recorded against supplied cores, so a changed boundary
            says who changed it. A core box is a scientific claim.

    Returns:
        A :class:`DearrayResult`.

    Raises:
        CoralError: If there are no cores to export and none can be produced.
        ImportError: If the ``dearray`` extra is not installed.
    """
    started = time.perf_counter()
    write_method = IMPORTED_METHOD if from_geojson else method

    # Nothing to do but export. Detection is skipped exactly the way ingest
    # skips a finished store, which is what makes a second run with
    # --export-cores cost only the crops.
    if from_geojson is None and has_cores(slide.path, method):
        logger.info(
            "  cores already on disk (%s), detection skipped", write_method
        )
        return _export_only(
            slide,
            method=method,
            export=export,
            export_root=export_root,
            all_levels=export_all_levels,
        )

    slot = slide.state.tasks.dearray.setdefault(write_method, TaskState())
    with slide._task(slot):  # noqa: SLF001 - the shared lifecycle recorder
        canvas = None
        if from_geojson is not None:
            boxes, record = _import_cores(
                slide, from_geojson, corrected_by=corrected_by
            )
        else:
            boxes, record, canvas = _detect_cores(
                slide, conf=conf, weights=weights, method=method
            )

        record["total_seconds"] = round(time.perf_counter() - started, 2)
        outputs = write_dearray(
            slide.path,
            write_method,
            slide_id=slide.path.stem,
            boxes_l0=boxes,
            record=record,
            canvas_rgb=canvas.rgb if canvas else None,
            mpp=canvas.mpp if canvas else None,
            letterbox=canvas.letterbox if canvas else None,
        )
        (dearray_dir(slide.path, write_method) / "dearray.json").write_text(
            json.dumps({**record, "n_cores": len(boxes)}, indent=2),
            encoding="utf-8",
        )

        n_exported = 0
        if export:
            # Not necessarily the method we just wrote. Detecting on a slide
            # whose cores a human has already supplied leaves the detection on
            # disk for comparison, but exports the human's boxes.
            use = export_method_for(slide.path, write_method)
            n_exported = _run_export(
                slide,
                boxes if use == write_method else read_cores(slide.path, use),
                method=use,
                root=export_root,
                slot=slide.state.tasks.dearray.setdefault(use, TaskState()),
                all_levels=export_all_levels,
            )

        slot.outputs = outputs
        slot.n_cores = len(boxes)

    return DearrayResult(
        method=write_method,
        boxes=boxes,
        outputs=outputs,
        seconds=time.perf_counter() - started,
        exported=n_exported,
    )


def _export_only(
    slide: CoralSlide,
    *,
    method: str,
    export: bool,
    export_root: Path | None,
    all_levels: bool = False,
) -> DearrayResult:
    """Cut cores that already exist, without re-deriving them."""
    started = time.perf_counter()
    use = export_method_for(slide.path, method)
    boxes = read_cores(slide.path, use)
    slot = slide.state.tasks.dearray.setdefault(use, TaskState())
    n_exported = 0
    if export:
        with slide._task(slot):  # noqa: SLF001 - the shared lifecycle recorder
            n_exported = _run_export(
                slide,
                boxes,
                method=use,
                root=export_root,
                slot=slot,
                all_levels=all_levels,
            )
    return DearrayResult(
        method=use,
        boxes=boxes,
        outputs=slot.outputs or {},
        seconds=time.perf_counter() - started,
        detected=False,
        exported=n_exported,
    )


def _run_export(
    slide: CoralSlide,
    boxes: list[Box],
    *,
    method: str,
    root: Path | None,
    slot: TaskState,
    all_levels: bool = False,
) -> int:
    """Export ``boxes`` and record where they went on the task."""
    if not boxes:
        raise CoralError(
            f"{slide.path.name} has no cores to export. Detect them with "
            f"--dearray, or supply them with --cores-from."
        )
    target = Path(root) if root else slide.path.parent / "cores"
    exported = export_cores(
        slide, boxes, method=method, root=target, all_levels=all_levels
    )
    slot.exported = len(exported.written) + len(exported.skipped)
    slot.cores_dir = str(target / slide.path.stem)
    return slot.exported
