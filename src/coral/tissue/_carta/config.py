"""Trimmed CARTA config — inference-only dataclasses (defaults, no YAML).

Vendored + trimmed from CARTA ``segmenter/config.py``: only the fields the
inference path reads are kept (DAPI→RGB normalization percentiles, the DeepLab
backbone/device, and the tiled-inference knobs). Training / eval / dearray /
otsu / postprocess / slide sub-configs and the YAML loader are dropped; the
defaults match CARTA's canonical config and the published checkpoint (replicate
DAPI→RGB, 512 tile, 0.5 confidence, resnet50 backbone).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    norm_lo_pct: float = 1.0
    norm_hi_pct: float = 99.0


@dataclass
class TrainingConfig:
    model: str = "deeplabv3"
    backbone: str = "resnet50"
    device: str = "auto"
    weights_dir: str = "segmenter_weights"


@dataclass
class InferenceConfig:
    tile_size: int = 512
    overlap: int = 0
    batch_size: int = 4
    dapi_rgb_mode: str = "replicate"
    confidence: float = 0.5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def resolve_path(self, path: str) -> str:
        if not path:
            return ""
        p = Path(path)
        return str(p) if p.is_absolute() else str(Path.cwd() / p)

    def weights_path(self, model: Optional[str] = None) -> str:
        model = model or self.training.model
        if model == "deeplabv3":
            return os.path.join(
                self.training.weights_dir,
                f"deeplabv3_{self.training.backbone}.pt",
            )
        raise ValueError(f"no weights path for model {model!r}")


def load_config(path: Optional[str] = None) -> Config:
    """Return CARTA inference defaults (the YAML loader is dropped in the embed)."""
    return Config()
