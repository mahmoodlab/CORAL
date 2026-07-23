"""Vendored CARTA DAPI tissue-segmentation core — inference only (DeepLabV3-ResNet50).

Copy of the **inference** path of CARTA's ``segmenter`` package (CC-BY-4.0 — see
``LICENSE`` / ``NOTICE``), embedded here so CORAL can run the CARTA tissue segmenter
without depending on the (private) CARTA repository. This mirrors how
``coral.features._eva`` / ``coral.features._kronos1`` vendor their upstream models.

Only the DAPI-nuclear → tissue-mask inference path is vendored. NO training or
loss code, and none of CARTA's dearray / TMA / YOLO / per-core OME export code is
copied. The thin adapter that plugs this into CORAL's ``BaseTissueSegmenter`` lives
in ``coral.tissue.carta`` (``CartaTissueSegmenter``).

Edits made to the upstream files (kept as small as possible):
  * ``deeplabv3.py`` — trimmed to inference: removed ``train_from_patches`` and
    its ``trainer`` / ``batching`` / ``losses`` imports + the ``freeze_backbone``
    hook (so ``trainer.py`` / ``batching.py`` / ``losses.py`` are not vendored);
    also dropped the unused ``segment_whole_slide`` method and build the backbone
    with ``weights=None`` (the checkpoint replaces them — no torchvision download).
  * ``interface.py`` — trimmed to the ``Segmenter`` ABC (dropped
    build_segmenter / postprocess_mask / segment_tissue / dearray helpers).
  * ``config.py`` — trimmed to the inference-only dataclasses with defaults (no
    YAML loader, no training/eval/dearray/otsu/postprocess sub-configs); defaults
    match the published checkpoint (replicate / 512 tile / 0.5 confidence).
  * ``dapi_rgb.py`` — ``normalize_percentile`` inlined instead of imported from
    CARTA ``segmenter/data.py`` (drops the tifffile / imagecodecs / PIL pull-in).
  * ``mask_cleanup.py`` — ``close_mask`` keeps only the scikit-image path; the
    optional ``cv2.morphologyEx`` fast path is removed (CORAL stays OpenCV-free).
  * ``tiled_infer.py`` — dropped the unused whole-slide ``tiled_segment_dapi``
    path (CORAL segments per-core via ``tiled_segment``).
  * ``core_mask_cleanup.py`` — dropped the unused ``CoreMaskCleanup.log_line``.
  * ``seg_scale.py`` — absorbs ``resample_to_target`` + the inference-scale
    constant from CARTA ``core_detection_scale.py`` (that file is not vendored
    separately; its YOLO letterbox / box-remap / dearray-target helpers are
    dropped), and adds ``mask_seg_to_native`` (from CARTA ``e2e_pipeline.py``) so
    a seg-scale mask can be upsampled to native.

Copied verbatim: ``device.py``.

This package is excluded from ruff / pyright / pytest / coverage (see
``pyproject.toml``) — third-party code kept close to verbatim. Several modules
import torch at module load, so import them lazily (the adapter does).
"""
