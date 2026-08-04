"""The detector canvas: a CORAL store becomes the 1280 image YOLO expects.

This is the one place CARTA's dearray path needs real adaptation rather
than copying. Upstream opens a qptiff with tifffile and reads a plane out
of it; CORAL already has the slide as an OME-Zarr store with marker names,
an ``mpp`` and a resolved nuclear channel, so it reads through
:class:`~coral.slide.core.CoralSlide` and never reopens the source file.

Everything about the canvas itself is CARTA's, in
:mod:`coral.dearray._carta.detect_scale`: the locked 26.8 µm/px target, the
percentile stretch, the letterbox into 1280 and the grayscale-to-RGB
replicate. The detector was trained on that exact chain, so reproducing it
is the whole job.

**Level 0, deliberately.** CARTA's ``choose_detection_read_level`` says
*"Always read detection input from pyramid level 0, then downsample to
target."* Reading a reduced level would be cheaper and would hand the
detector a canvas built by a different resampling path than the one it was
validated on. Measured on a 34560 x 34560 uint8 slide: the level-0 nuclear
read is about 1.2 s and the resample about 8.6 s, so the read is not the
expensive part and there is nothing to gain by degrading it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, NamedTuple

import numpy as np

from coral.dearray._carta.detect_scale import (
    TARGET_UM_PER_PX,
    prepare_slide_rgb,
)
from coral.slide.core import CoralSlide
from coral.utils.errors import CoralError

logger = logging.getLogger(__name__)

__all__ = ["DetectorCanvas", "build_detector_canvas"]


class DetectorCanvas(NamedTuple):
    """The detector's input, and everything needed to map boxes back.

    Attributes:
        rgb: ``(1280, 1280, 3)`` uint8, what the detector sees.
        letterbox: Metadata from CARTA's letterbox, consumed by
            ``remap_boxes_input_to_l0`` to return level-0 coordinates.
        mpp: The slide's level-0 microns per pixel. The remap needs the *same*
            value that built the canvas, which is why it travels with it.
        height: Level-0 height, for clipping boxes to the slide.
        width: Level-0 width.
        nuclear_index: Channel the canvas was built from.
        read_seconds: Time spent reading the level-0 plane.
        canvas_seconds: Time spent resampling and letterboxing.
    """

    rgb: np.ndarray
    letterbox: dict[str, Any]
    mpp: float
    height: int
    width: int
    nuclear_index: int
    read_seconds: float
    canvas_seconds: float


def build_detector_canvas(slide: CoralSlide) -> DetectorCanvas:
    """Read a slide's nuclear channel at level 0 and build the detector canvas.

    Args:
        slide: An open canonical store.

    Returns:
        A :class:`DetectorCanvas`.

    Raises:
        CoralError: If the slide has no nuclear channel. Core detection is
            nuclear-only, so there is nothing to fall back to and guessing a
            channel would silently detect on the wrong stain.
    """
    index = slide.nuclear_channel
    if index is None:
        raise CoralError(
            f"{slide.path.name} has no nuclear channel, so its cores "
            f"cannot be detected: CARTA's detector is nuclear-only. Set "
            f"one at ingest with --nuclear-marker."
        )
    mpp = slide._mpp()  # noqa: SLF001 - the validated public-boundary reader

    started = time.perf_counter()
    # Materialised rather than left lazy: the resample below is a single global
    # operation over the whole plane, so it would pull every chunk anyway.
    plane = np.asarray(slide.image[index])
    read_seconds = time.perf_counter() - started
    height, width = int(plane.shape[0]), int(plane.shape[1])
    logger.info(
        "read level 0 nuclear plane '%s' %dx%d (%.0f MB) in %.1f s",
        slide.markers[index] if slide.markers else index,
        width,
        height,
        plane.nbytes / 1e6,
        read_seconds,
    )

    started = time.perf_counter()
    rgb, letterbox = prepare_slide_rgb(plane, mpp)
    canvas_seconds = time.perf_counter() - started
    logger.info(
        "canvas %dx%d at %.4f um/px, letterbox scale %.3f, in %.1f s",
        letterbox["scaled_w"],
        letterbox["scaled_h"],
        TARGET_UM_PER_PX,
        letterbox["letterbox_scale"],
        canvas_seconds,
    )

    return DetectorCanvas(
        rgb=rgb,
        letterbox=letterbox,
        mpp=mpp,
        height=height,
        width=width,
        nuclear_index=index,
        read_seconds=read_seconds,
        canvas_seconds=canvas_seconds,
    )
