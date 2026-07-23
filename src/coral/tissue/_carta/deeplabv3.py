"""DeepLabV3 / ResNet50 tissue segmenter — inference only.

Vendored from CARTA ``segmenter/deeplabv3.py``, trimmed to inference: the
``train_from_patches`` path and its ``trainer`` / ``batching`` / ``losses``
imports (and the ``freeze_backbone`` training hook) are removed — the CORAL embed
only runs inference.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

from .config import Config, load_config
from .dapi_rgb import dapi_to_rgb
from .device import get_torch_device
from .interface import Segmenter
from .tiled_infer import tiled_segment


def _build_deeplab(backbone: str = "resnet50") -> nn.Module:
    if backbone != "resnet50":
        raise ValueError(f"only resnet50 backbone implemented, got {backbone!r}")
    # weights=None + weights_backbone=None: the CARTA checkpoint supplies
    # every weight, so skip torchvision's COCO + ImageNet downloads (~160MB;
    # also works offline). aux_loss=True keeps the aux_classifier the
    # checkpoint includes, so the strict load_state_dict matches.
    model = deeplabv3_resnet50(
        weights=None, weights_backbone=None, aux_loss=True
    )
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    return model


class DeepLabV3Segmenter(Segmenter):
    def __init__(self, cfg: Config = None, weights_path: str = None):
        self.cfg = cfg or load_config()
        self.weights_path = weights_path or self.cfg.weights_path("deeplabv3")
        self._device = get_torch_device(self.cfg.training.device)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = _build_deeplab(self.cfg.training.backbone).to(self._device)
            if self.weights_path and os.path.isfile(self.weights_path):
                state = torch.load(self.weights_path, map_location=self._device,
                                   weights_only=True)
                self._model.load_state_dict(state)
            self._model.eval()

    def _predict_batch(self, batch_imgs):
        with torch.inference_mode():
            x = batch_imgs.to(self._device, dtype=torch.float32)
            logits = self._model(x)["out"]
            return torch.sigmoid(logits[:, 0])

    def segment(self, image: np.ndarray, source: str = None) -> np.ndarray:
        self._ensure_model()
        rgb = dapi_to_rgb(
            image, mode=self.cfg.inference.dapi_rgb_mode,
            lo_pct=self.cfg.data.norm_lo_pct, hi_pct=self.cfg.data.norm_hi_pct,
        )
        return tiled_segment(
            rgb, self._predict_batch,
            tile_size=self.cfg.inference.tile_size,
            overlap=self.cfg.inference.overlap,
            batch_size=self.cfg.inference.batch_size,
            threshold=self.cfg.inference.confidence,
        )
