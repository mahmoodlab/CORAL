"""Vendored CARTA TMA dearraying core, inference only.

Copy of the **core-detection** path of CARTA's ``segmenter`` package (CC-BY-4.0,
see ``LICENSE`` / ``NOTICE``), embedded so CORAL can dearray a TMA without
depending on the (private) CARTA repository. Sibling of
:mod:`coral.tissue._carta`, which vendors the same project's tissue segmenter.

This picks up what that vendoring deliberately left behind.
``coral.tissue._carta.seg_scale`` records that it absorbed ``resample_to_target``
from CARTA's ``core_detection_scale.py`` and dropped "its YOLO letterbox /
box-remap / dearray-target helpers". Those helpers are exactly what core
detection needs, so they arrive here.

Only inference is vendored. No training or loss code, no QuPath launching
(CARTA's ``qupath_hitl.py``, 816 lines, is not copied: CORAL round-trips GeoJSON
instead), no per-core OME export, no bundle writers, and no e2e pipeline.

The tissue segmenter is **not** vendored a second time. CARTA's ``split_yolo``
segments each detected box with its own segmenter; the CORAL adaptation calls
:class:`coral.tissue.carta.CartaTissueSegmenter`, which is already here.

**Scope.** Only CARTA's real detection path. ``core_split.py`` is deliberately
NOT vendored: grid fitting, k-means row/column clustering and watershed splitting
are the legacy heuristics CARTA was built to replace, and ``e2e_pipeline.py`` never
calls them. An earlier version of this package vendored that whole file and then
presented its methods as CORAL features, which wasted the point of CARTA. Removed.

**Faithful, not improved.** Algorithms are copied as they are so this tree can be
diffed against CARTA and so behaviour matches the tool that was validated on real
slides. Where upstream looks wrong it is recorded in a comment and left alone. An
earlier version of this vendoring changed three algorithms and a fallback; that
was a mistake and was reverted.

Edits made to the upstream files, all structural and kept as small as possible:

  * ``detect.py`` (from ``e2e_pipeline.py``) is the YOLO box detection and the box
    bookkeeping around it, copied verbatim. Not vendored from that file: core-map
    rendering, bundle writing, provenance, and everything CORAL writes itself.
    ``_load_review_boxes`` becomes ``prepare_review_boxes`` taking boxes rather
    than reading GeoJSON, because CORAL reads corrections through its own path,
    and its ``print`` diagnostics go through a logger. Upstream's
    ``load_core_boxes_from_geojson`` is not vendored.

  * ``detect_scale.py`` (from ``core_detection_scale.py``) renames
    ``prepare_slide_rgb`` to
    :func:`~coral.dearray._carta.detect_scale.prepare_detector_rgb`, and drops
    the deprecated ``FIXED_NORM_LO`` / ``FIXED_NORM_HI`` constants along with
    CARTA's per-cohort µm/px constants, since CORAL reads mpp from the store.

    The ``remap_boxes_*`` pair keeps its upstream names, and the ``l0`` in them
    is a misnomer worth knowing about. Both functions are scale-agnostic: they
    undo the resample and letterbox relative to whatever ``native_um_per_px``
    the caller declared. CORAL declares a **reduced pyramid level**, so they
    return that level's coordinates, not level 0's. Reading the names as a
    promise of level-0 coordinates is the cheapest possible route to a
    whole-slide coordinate offset that still looks plausible on screen; see
    :func:`coral.dearray.canvas.build_detector_canvas` for the scale the
    caller actually passes.

The tissue segmenter is CORAL's, not CARTA's. ``coral.tissue._carta`` is the
authoritative copy: CARTA's own may be behind, and the two already infer at
different scales, 1.0 µm/px here against 2.027 there. Anything here that needs
a tissue mask goes through :class:`coral.tissue.carta.CartaTissueSegmenter`.

Two scale facts that are easy to get wrong. The detector canvas is
**26.8 µm/px**, which is ``34301 µm / 1280``: the widest training slide fitted
into YOLO's input, so a 1 mm core is about 37 px. And for a 17.5 mm slide the
resample produces about 654 px, which the letterbox then **upscales roughly 2x**
to fill 1280 rather than leaving it as an island on black padding.

This package is excluded from ruff / pyright / pytest / coverage (see
``pyproject.toml``), as third-party code kept close to verbatim. Modules that
import torch or ultralytics do so lazily, so importing this package is safe
without the ``dearray`` extra installed.
"""
