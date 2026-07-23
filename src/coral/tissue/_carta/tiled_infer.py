"""
Tiled inference for large 2-D images with overlap averaging.

Vendored from CARTA ``segmenter/tiled_infer.py``. Edit vs upstream: the
whole-slide ``tiled_segment_dapi`` path is dropped — CORAL segments per-core
via ``DeepLabV3Segmenter.segment`` (which calls ``tiled_segment``), so the
per-tile-normalizing whole-slide variant is unused. When a CORAL progress
bar is active, its ``total`` is reset to the tile-batch count and advanced
per finished batch.
"""
from __future__ import annotations

from math import ceil

import numpy as np
import torch


def tiled_segment(rgb: np.ndarray, predict_fn, tile_size: int = 512,
                  overlap: int = 64, batch_size: int = 4,
                  threshold: float = 0.5) -> np.ndarray:
    """
    Run predict_fn on overlapping tiles and stitch into a full mask.

    Parameters
    ----------
    rgb         : (H, W, 3) uint8
    predict_fn  : callable(batch_tensor) -> (B, H, W) bool/uint8 tensor on any device
    """
    H, W = rgb.shape[:2]
    stride = max(1, tile_size - overlap)
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(H - tile_size, 0) + 1, stride))
    xs = list(range(0, max(W - tile_size, 0) + 1, stride))
    if not ys or ys[-1] + tile_size < H:
        ys.append(max(0, H - tile_size))
    if not xs or xs[-1] + tile_size < W:
        xs.append(max(0, W - tile_size))

    tiles, coords = [], []
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            patch = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            ph, pw = y1 - y0, x1 - x0
            patch[:ph, :pw] = rgb[y0:y1, x0:x1]
            tiles.append(patch)
            coords.append((y0, x0, ph, pw))

    n_batches = ceil(len(tiles) / batch_size) if tiles else 0
    try:
        from coral.utils.progress import get_active_bar

        bar = get_active_bar()
    except Exception:  # noqa: BLE001 — progress is optional
        bar = None
    if bar is not None and not bar.disable and n_batches:
        bar.unit = "batch"
        bar.reset(total=n_batches)

    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start: start + batch_size]
        batch_coords = coords[start: start + batch_size]
        imgs = torch.stack([
            torch.from_numpy(t).permute(2, 0, 1).float() / 255.0
            for t in batch_tiles
        ])
        preds = predict_fn(imgs)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        for pred, (y0, x0, ph, pw) in zip(preds, batch_coords):
            mask = pred[:ph, :pw].astype(np.float32)
            acc[y0:y0 + ph, x0:x0 + pw] += mask
            cnt[y0:y0 + ph, x0:x0 + pw] += 1.0
        if bar is not None and not bar.disable and n_batches:
            bar.update(1)

    cnt = np.maximum(cnt, 1.0)
    return (acc / cnt) >= threshold
