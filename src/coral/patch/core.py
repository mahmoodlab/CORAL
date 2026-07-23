"""``Patcher`` — grid patch-coordinate generation + tissue scoring.

The patcher produces **coordinates, not pixels**: a vectorized grid of
level-0 ``(x, y)`` top-left coordinates at the slide's base mpp, each
tagged with the fraction of its box under the tissue mask. Pixels are
read later, at feature-extraction time.

Every grid position is returned — the patcher scores tissue coverage, it
never filters on it. Callers that want an on-tissue subset apply their
own cut-off to the returned fractions.

The numerics live in two pure module-level helpers so they can be
unit-tested without a slide:

- :func:`_grid_coords` — vectorized grid (``np.meshgrid``) with the
  kronos/TRIDENT edge-cover so right/bottom tissue isn't dropped.
- :func:`_tissue_proportions` — per-patch tissue fraction via a
  bbox-prefilter + an integral image (summed-area table), so coverage
  for every patch is O(1); optional progress ticks advance a live bar
  in chunks without changing the result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from coral.config import PatchConfig
    from coral.slide import CoralSlide

logger = logging.getLogger(__name__)


def _axis_starts(extent: int, size: int, stride: int) -> np.ndarray:
    """Top-left starts along one axis, with an edge-cover start.

    Returns ``arange(0, extent - size + 1, stride)`` plus the final
    ``extent - size`` start when the regular grid misses the border —
    matching kronos/TRIDENT so the right/bottom strip is still patched.
    Every start keeps the box in-bounds (``start + size <= extent``).
    When ``extent < size`` no patch fits, so the axis is empty.
    """
    if extent < size:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, extent - size + 1, stride, dtype=np.int64)
    last = extent - size
    if starts.size == 0 or int(starts[-1]) != last:
        starts = np.append(starts, np.int64(last))
    return starts


def _grid_coords(
    height: int, width: int, size: int, stride: int
) -> np.ndarray:
    """Vectorized grid of level-0 ``(x, y)`` top-left coordinates.

    Args:
        height: Level-0 image height (y extent), in pixels.
        width: Level-0 image width (x extent), in pixels.
        size: Patch side in pixels.
        stride: Step between patch starts, in pixels.

    Returns:
        ``(N, 2)`` int64 array of ``(x, y)`` coords, row-major, each
        with ``x + size <= width`` and ``y + size <= height``. Empty
        ``(0, 2)`` when no patch fits.

    Example:
        >>> _grid_coords(64, 64, size=32, stride=32).tolist()
        [[0, 0], [32, 0], [0, 32], [32, 32]]
    """
    xs = _axis_starts(width, size, stride)
    ys = _axis_starts(height, size, stride)
    xx, yy = np.meshgrid(xs, ys)  # indexing="xy" → (len(ys), len(xs))
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.int64)


def _tissue_proportions(
    coords: np.ndarray,
    size: int,
    mask: np.ndarray,
    *,
    progress: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Fraction of each patch box under ``mask`` (level-0 coords).

    Uses a bbox-prefilter (patches outside the mask's bounding box read
    exactly 0) plus an integral image, so every patch's coverage is
    O(1). Optional ``progress(n)`` advances a live bar as patches are
    scored in chunks (same math, visible per-patch progress).

    Args:
        coords: ``(N, 2)`` int64 ``(x, y)`` top-left coords, in-bounds.
        size: Patch side in pixels.
        mask: Boolean ``(y, x)`` tissue mask at level 0.
        progress: Optional ``callable(int)`` called with the number of
            patches just scored.

    Returns:
        ``(N,)`` float32 tissue fractions in ``[0, 1]``.

    Example:
        >>> import numpy as np
        >>> mask = np.zeros((64, 64), dtype=bool)
        >>> mask[:32, :32] = True
        >>> coords = _grid_coords(64, 64, size=32, stride=32)
        >>> _tissue_proportions(coords, 32, mask).tolist()
        [1.0, 0.0, 0.0, 0.0]
    """
    n = int(coords.shape[0])
    prop = np.zeros(n, dtype=np.float32)
    if n == 0:
        return prop
    m = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    if rows.size == 0:  # empty mask → every patch reads 0
        if progress is not None:
            progress(n)
        return prop

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    x, y = coords[:, 0], coords[:, 1]
    # bbox-prefilter: only boxes intersecting the tissue bbox can be >0.
    inb = (x < x1) & (x + size > x0) & (y < y1) & (y + size > y0)
    if not inb.any():
        if progress is not None:
            progress(n)
        return prop

    # Summed-area table, zero-padded. Built in place (cumsum twice with
    # out=integ) to avoid two full-size intermediates, and int32 while
    # the pixel count fits (max value = total tissue pixels ≤ h*w), which
    # halves the memory + bandwidth vs int64.
    h, w = m.shape
    dtype = np.int32 if h * w < (1 << 31) else np.int64
    integ = np.zeros((h + 1, w + 1), dtype=dtype)
    integ[1:, 1:] = m
    integ.cumsum(axis=0, out=integ)
    integ.cumsum(axis=1, out=integ)

    in_idx = np.flatnonzero(inb)
    n_out = n - int(in_idx.size)
    if progress is not None and n_out:
        progress(n_out)

    area = float(size * size)
    chunk = 4096
    for start_i in range(0, int(in_idx.size), chunk):
        sel = in_idx[start_i : start_i + chunk]
        cx, cy = x[sel], y[sel]
        box_sum = (
            integ[cy + size, cx + size]
            - integ[cy, cx + size]
            - integ[cy + size, cx]
            + integ[cy, cx]
        )
        prop[sel] = (box_sum / area).astype(np.float32)
        if progress is not None:
            progress(int(sel.size))
    return prop


def _cell_coords(
    centroids: np.ndarray, patch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centered top-left coords + ids for **every** cell (keep-edge).

    Each cell's patch box is centered on its centroid
    (``round(centroid) - patch_size // 2``). **No cell is dropped:** a box
    that runs off the level-0 image is kept with its raw — possibly
    negative or overflowing — top-left, and the overhang is zero-padded
    when the pixels are read at feature-extraction time. (Reversed the
    earlier drop-at-border behaviour so border cells still get an
    embedding.)

    Args:
        centroids: ``(M, 3)`` float array of ``(cell_id, x, y)`` rows at
            level 0 — the schema of ``cells/cell_centroids.csv``.
        patch_size: Patch side in pixels.

    Returns:
        ``(coords, cell_ids, dropped_xy)`` — ``(M, 2)`` int64 ``(x, y)``
        top-left coords and ``(M,)`` int64 cell ids for **all** cells (in
        order), plus an always-empty ``(0, 2)`` ``dropped_xy`` (kept for
        signature stability; no cell is dropped under keep-edge).

    Example:
        >>> import numpy as np
        >>> cents = np.array([[7.0, 50.0, 40.0], [9.0, 1.0, 1.0]])
        >>> coords, ids, dropped = _cell_coords(cents, 16)
        >>> coords.tolist(), ids.tolist(), dropped.tolist()
        ([[42, 32], [-7, -7]], [7, 9], [])
    """
    ids = centroids[:, 0].astype(np.int64)
    half = patch_size // 2
    x0 = np.round(centroids[:, 1]).astype(np.int64) - half
    y0 = np.round(centroids[:, 2]).astype(np.int64) - half
    coords = np.stack([x0, y0], axis=1)
    empty = np.empty((0, 2), dtype=centroids.dtype)
    return coords, ids, empty


class Patcher:
    """Generate grid patch coordinates + tissue fractions for a slide.

    Patches are produced at the slide's **base mpp** (level 0) — the
    resolution KRONOS encodes at — as vectorized ``(x, y)`` coords, each
    scored by the fraction of its box under the slide's tissue mask.
    Every grid position is returned; nothing is dropped for having too
    little tissue.
    """

    def __init__(
        self,
        slide: CoralSlide,
        config: PatchConfig,
        *,
        tissue_method: str | None = None,
    ) -> None:
        """Bind the patcher to a slide + patch config.

        Args:
            slide: An ``CoralSlide`` instance (tissue must be detected).
            config: A ``PatchConfig`` describing patch size, mode, and
                stride/overlap.
            tissue_method: Resolved method name under
                ``tissue/tissue_<method>/`` (``None`` auto-resolves).
        """
        self._slide = slide
        self._config = config
        self._tissue_method = tissue_method

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Every grid coord + the tissue fraction of its box.

        The full level-0 grid at base mpp, each patch scored against the
        slide's tissue mask. **Nothing is filtered** — a patch with no
        tissue under it is returned like any other, carrying a
        ``tissue_prop`` of 0. Apply a cut-off downstream (e.g.
        ``coords[prop >= 0.1]``) so the threshold stays a decision of
        the analysis, not of the stored patch set. Grid mode only.

        Returns:
            ``(coords, tissue_prop)`` — ``(N, 2)`` int64 level-0
            ``(x, y)`` coords and ``(N,)`` float32 tissue fractions, for
            every grid position.

        Raises:
            ValueError: If ``mode == "cell_centered"`` (grid-only; use
                ``cell_centered_coords``), or tissue detection has not
                run on the slide.

        Example:
            >>> import shutil, tempfile
            >>> from pathlib import Path
            >>> from coral import CoralSlide
            >>> from coral.config import PatchConfig
            >>> from coral.tissue import OtsuTissueSegmenter
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dst = Path(d) / "s.zarr"
            ...     _ = shutil.copytree("tests/data/tiny_slide.zarr", dst)
            ...     slide = CoralSlide.open(dst)
            ...     _ = slide.detect_tissue(OtsuTissueSegmenter(), viz=False)
            ...     cfg = PatchConfig(patch_size=16)
            ...     coords, prop = Patcher(slide, cfg).generate()
            ...     coords.shape[1], prop.shape == (coords.shape[0],)
            (2, True)
        """
        cfg = self._config
        if cfg.mode == "cell_centered":
            msg = (
                "generate is grid-only; for mode='cell_centered' use "
                "cell_centered_coords()."
            )
            raise ValueError(msg)

        _, height, width = self._slide.store["0"].shape
        coords = _grid_coords(
            int(height), int(width), cfg.patch_size, cfg.effective_stride
        )
        n = int(coords.shape[0])
        progress = None
        try:
            from coral.utils.progress import get_active_bar

            bar = get_active_bar()
        except Exception:  # noqa: BLE001 — progress is optional
            bar = None
        if bar is not None and not bar.disable:
            bar.unit = "patch"
            bar.reset(total=max(n, 1))

            def _tick(k: int) -> None:
                bar.update(k)

            progress = _tick

        mask = self._slide._tissue_mask_np(self._tissue_method)
        prop = _tissue_proportions(
            coords, cfg.patch_size, mask, progress=progress
        )
        if progress is not None and n == 0:
            progress(1)
        logger.info(
            "%s: %d patches (size=%d, stride=%d, %d with tissue)",
            self._slide.path.name,
            int(coords.shape[0]),
            cfg.patch_size,
            cfg.effective_stride,
            int((prop > 0).sum()),
        )
        return coords, prop

    def cell_centered_coords(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell-centered patch coords + ids for **every** cell (keep-edge).

        Centers a ``config.patch_size`` box on each stored cell centroid
        (``cells/cell_centroids.csv``). **No cell is dropped** — a box that
        runs off the level-0 image is kept and zero-padded when its pixels
        are read at feature-extraction time (CP-01 fix). Cell-centered
        patches are not tissue-filtered — the cells are already restricted
        to tissue by ``segment_cells``.

        Returns:
            ``(coords, cell_ids, dropped_xy)`` — ``(M, 2)`` int64 level-0
            ``(x, y)`` top-left coords, ``(M,)`` int64 cell ids, and an
            always-empty ``dropped_xy`` (no drops under keep-edge). See
            :func:`_cell_coords`.

        Raises:
            ValueError: If cell segmentation has not run on the slide.

        Example:
            >>> import shutil, tempfile
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> from coral import CoralSlide
            >>> from coral.config import PatchConfig
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dst = Path(d) / "s.zarr"
            ...     _ = shutil.copytree("tests/data/tiny_slide.zarr", dst)
            ...     slide = CoralSlide.open(dst)
            ...     (dst / "cells").mkdir()
            ...     pd.DataFrame(
            ...         {"cell_id": [1], "x": [16.0], "y": [16.0]}
            ...     ).to_csv(dst / "cells" / "cell_centroids.csv", index=False)
            ...     cfg = PatchConfig(patch_size=8, mode="cell_centered")
            ...     coords, ids, dropped = Patcher(
            ...         slide, cfg
            ...     ).cell_centered_coords()
            ...     coords.tolist(), ids.tolist(), dropped.tolist()
            ([[12, 12]], [1], [])
        """
        cfg = self._config
        centroids = self._slide._cell_centroids()
        coords, cell_ids, dropped_xy = _cell_coords(centroids, cfg.patch_size)
        n = int(coords.shape[0])
        try:
            from coral.utils.progress import get_active_bar

            bar = get_active_bar()
        except Exception:  # noqa: BLE001 — progress is optional
            bar = None
        if bar is not None and not bar.disable:
            bar.unit = "patch"
            bar.reset(total=max(n, 1))
            bar.update(n if n else 1)
        logger.debug(
            "%s: %d cell patches (patch_size=%d, keep-edge)",
            self._slide.path.name,
            n,
            cfg.patch_size,
        )
        return coords, cell_ids, dropped_xy
