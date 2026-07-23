"""``CellposeSegmenter`` — Cellpose 4.x (cpsam) cell segmentation.

Cellpose is an **opt-in** dependency (the ``cells`` extra): the import
is guarded so this module loads without it, and the model is built
lazily on first :meth:`segment` (or eagerly via :meth:`ensure_loaded`).
cpsam takes a 2-channel ``[membrane, nuclear]`` image and returns an
instance-label mask; built-in tiling (``bsize``/``tile_overlap``)
flow-stitches across tiles, so whole-slide cells don't fragment at tile
seams. When a CORAL progress bar is active, its ``total`` is reset to the
Cellpose tile-batch count and advanced per finished GPU batch.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, ClassVar

import numpy as np

from coral.cells.base import BaseCellSegmenter
from coral.cells.registry import register

logger = logging.getLogger(__name__)

# cellpose lives in the opt-in `cells` extra; check without importing it
# (keeps this module importable + type-checkable without the extra).
CELLPOSE_AVAILABLE = importlib.util.find_spec("cellpose") is not None

# cpsam default tile size (cellpose rejects other values for sam_vitl).
_DEFAULT_BSIZE = 256
_DEFAULT_BATCH_SIZE = 8


def _n_tile_batches(
    height: int,
    width: int,
    *,
    bsize: int = _DEFAULT_BSIZE,
    tile_overlap: float = 0.1,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    diameter: float | None = None,
) -> int:
    """Tile-batch count matching ``cellpose.core.run_net`` for one 2-D image.

    Used to size the progress bar before Cellpose starts. When ``diameter``
    is set, Cellpose rescales so cells are ~30 px — the same factor is
    applied here so the batch count tracks the resized canvas.
    """
    rescale = (
        (30.0 / diameter) if diameter is not None and diameter > 0 else 1.0
    )
    lyr = int(height * rescale)
    lxr = int(width * rescale)
    ypad1, ypad2 = _pad_axis(lyr, bsize)
    xpad1, xpad2 = _pad_axis(lxr, bsize)
    ly = lyr + ypad1 + ypad2
    lx = lxr + xpad1 + xpad2
    ny = (
        1
        if ly <= bsize
        else int(np.ceil((1.0 + 2 * tile_overlap) * ly / bsize))
    )
    nx = (
        1
        if lx <= bsize
        else int(np.ceil((1.0 + 2 * tile_overlap) * lx / bsize))
    )
    ntiles = ny * nx
    return max(1, int(np.ceil(ntiles / max(1, batch_size))))


def _pad_axis(length: int, min_len: int) -> tuple[int, int]:
    """Mirror ``cellpose.transforms.get_pad_yx`` for one axis."""
    div, extra = 16, 1
    if length >= min_len:
        lpad = int(div * np.ceil(length / div) - length)
    else:
        lpad = max(0, min_len - length - (extra * div))
    pad1 = extra * div // 2 + lpad // 2
    pad2 = extra * div // 2 + lpad - lpad // 2
    return pad1, pad2


@register("cellpose")
class CellposeSegmenter(BaseCellSegmenter):
    """Cellpose 4.x (cpsam) whole-cell instance segmenter.

    Runs the cpsam model (GPU when available, else CPU) on a 2-channel
    ``[membrane, nuclear]`` image and returns an ``int32`` instance
    mask. The model is loaded lazily (and its weights downloaded +
    cached) on first :meth:`segment`, or upfront via
    :meth:`ensure_loaded` (preferred for CLI so load is not attributed
    to the first image's progress bar).

    Args:
        model_type: Cellpose pretrained model (default ``"cpsam"``).
        diameter: Expected cell diameter in px; ``None`` lets cpsam
            estimate it.
        flow_threshold: Flow-error threshold.
        cellprob_threshold: Cell-probability threshold.
        min_size: Drop cells smaller than this many pixels.
        use_gpu: Use the GPU when one is available.
        device: Explicit torch device (e.g. ``"cuda:1"``); overrides
            ``use_gpu``. ``None`` → auto (cuda if available).
        bsize: Tile block size in px; ``None`` uses the cellpose default
            (256 for cpsam).
        tile_overlap: Fractional overlap between tiles.
        batch_size: Tiles per GPU forward (Cellpose default 8).

    Example:
        >>> seg = CellposeSegmenter(diameter=30, use_gpu=False)
        >>> seg.name
        'cellpose'
        >>> seg.diameter
        30
    """

    name: ClassVar[str] = "cellpose"

    def __init__(
        self,
        *,
        model_type: str = "cpsam",
        diameter: float | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
        use_gpu: bool = True,
        device: str | None = None,
        bsize: int | None = None,
        tile_overlap: float = 0.1,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        """Bind cpsam hyperparameters (see the class docstring)."""
        self.model_type = model_type
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self.use_gpu = use_gpu
        self.device = device
        self.bsize = bsize
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self._model: Any = None

    def ensure_loaded(self) -> None:
        """Load (and cache) the cpsam weights before the first slide.

        Call this once from the CLI before the per-image loop so model
        download / construction is not charged to image ``[1/N]``.

        Example:
            Prefetch weights once before a cohort loop::

                from coral.cells import CellposeSegmenter

                seg = CellposeSegmenter()
                seg.ensure_loaded()
        """
        self._load_model()

    def _load_model(self) -> Any:  # noqa: ANN401 — a cellpose model
        """Build (and cache) the cpsam model; GPU if available."""
        if self._model is not None:
            return self._model
        if not CELLPOSE_AVAILABLE:
            msg = (
                "Cellpose is not installed — install the cells extra: "
                "`pip install coral[cells]` (or `uv sync --extra cells`)."
            )
            raise ImportError(msg)
        import torch  # pyright: ignore[reportMissingImports]
        from cellpose import models  # pyright: ignore[reportMissingImports]

        if self.device is not None:
            dev = torch.device(self.device)
            gpu = dev.type in ("cuda", "mps")
        else:
            dev = None
            gpu = bool(self.use_gpu and torch.cuda.is_available())
        logger.info(
            "loading cellpose model %r (gpu=%s, device=%s)",
            self.model_type,
            gpu,
            dev,
        )
        self._model = models.CellposeModel(
            gpu=gpu, device=dev, pretrained_model=self.model_type
        )
        return self._model

    def segment(
        self,
        nuclear: np.ndarray,
        membrane: np.ndarray,
        *,
        mpp: float,
    ) -> np.ndarray:
        """Segment cells from the nuclear + membrane channels.

        Args:
            nuclear: 2-D nuclear channel.
            membrane: 2-D composite membrane channel (same shape).
            mpp: Microns-per-pixel (informational; cpsam uses
                ``diameter`` for the cell-size prior).

        Returns:
            ``int32`` ``(y, x)`` instance mask, 0 = background.

        Example:
            Prefer the slide driver (loads channels + writes outputs)::

                from coral import CoralSlide
                from coral.cells import CellposeSegmenter

                slide = CoralSlide.open("slide.zarr")
                slide.segment_cells(CellposeSegmenter())
        """
        model = self._load_model()
        # cpsam input: (H, W, 2) with channel 0 = membrane, 1 = nuclear.
        img = np.stack(
            [
                np.asarray(membrane, dtype=np.float32),
                np.asarray(nuclear, dtype=np.float32),
            ],
            axis=-1,
        )
        bsize = self.bsize if self.bsize is not None else _DEFAULT_BSIZE
        n_batches = _n_tile_batches(
            int(img.shape[0]),
            int(img.shape[1]),
            bsize=bsize,
            tile_overlap=self.tile_overlap,
            batch_size=self.batch_size,
            diameter=self.diameter,
        )
        eval_kwargs: dict[str, Any] = {
            "channel_axis": 2,
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "min_size": self.min_size,
            "tile_overlap": self.tile_overlap,
            "batch_size": self.batch_size,
            "bsize": bsize,
        }
        masks, _flows, _styles = self._eval_with_progress(
            model, img, n_batches=n_batches, eval_kwargs=eval_kwargs
        )
        mask = masks[0] if isinstance(masks, list) else masks
        mask = np.asarray(mask, dtype=np.int32)
        logger.debug(
            "%s: %d cells (mpp=%.3g)", self.name, int(mask.max()), mpp
        )
        return mask

    def _eval_with_progress(
        self,
        model: Any,  # noqa: ANN401 — cellpose CellposeModel
        img: np.ndarray,
        *,
        n_batches: int,
        eval_kwargs: dict[str, Any],
    ) -> tuple[Any, Any, Any]:
        """Run ``model.eval``, ticking the active bar per tile batch."""
        from cellpose import core  # pyright: ignore[reportMissingImports]

        try:
            from coral.utils.progress import get_active_bar

            bar = get_active_bar()
        except Exception:  # noqa: BLE001 — progress is optional
            bar = None
        if bar is not None and not bar.disable and n_batches:
            bar.unit = "batch"
            bar.reset(total=n_batches)

        orig_forward = core._forward  # noqa: SLF001

        def _forward_tick(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            out = orig_forward(*args, **kwargs)
            if bar is not None and not bar.disable and n_batches:
                bar.update(1)
            return out

        core._forward = _forward_tick  # noqa: SLF001
        try:
            return model.eval(img, **eval_kwargs)
        finally:
            core._forward = orig_forward  # noqa: SLF001
