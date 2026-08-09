"""``CoralDataset`` — lazy patch reader for feature extraction.

An index-into-the-slide dataset: :meth:`~CoralDataset.__getitem__` reads one
patch's level-0 pixels at its coords (grid raw box, or cell-isolated when
``cell_ids`` is given) and optionally **dtype-scales** it to ``float32
[0, 1]`` — the data-driven "dtype-normed patch" the patcher hands out (the
divisor is set by the *image* dtype, never the encoder).

It deliberately does **not** subclass ``torch.utils.data.Dataset``, so it
imports without the model extras; a torch ``DataLoader`` still wraps it (it
only needs ``__len__`` + ``__getitem__``). The zarr arrays are opened
per-call so the dataset survives ``DataLoader`` worker processes.

Mirrors TRIDENT's ``WSIPatcherDataset`` (read in ``__getitem__``), but CORAL
does only the I/O-bound lazy read + the data-driven scale here; the
per-encoder GPU transform runs in the encoder's ``forward`` (matrix doc).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from coral.dtypes import scaling_factor


class CoralDataset:
    """Lazily read (and optionally dtype-scale) slide patches by index.

    Args:
        slide_path: Path to the slide ``.zarr`` (reads level-0 ``"0"`` and,
            for cell patches, ``"cells/cell_mask"``).
        coords: ``(n, 2)`` array of ``(x, y)`` level-0 top-left coords.
        patch_size: Patch side in level-0 pixels.
        idxs: Channel indices to select (the marker subset), in order.
        scale: When ``True`` (default), divide by the data-driven
            ``scaling_factor(dtype)`` → ``float32 [0, 1]``; when ``False``,
            return the raw box (mean_marker scales in its own float64
            reduction).
        cell_ids: Optional ``(n,)`` cell ids; when given, each box is
            isolated to its cell (neighbours + background zeroed).

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import numpy as np, zarr
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "s.zarr"
        ...     r = zarr.open_group(str(p), mode="w")
        ...     _ = r.create_dataset(
        ...         "0", data=np.full((2, 16, 16), 65535, dtype="uint16")
        ...     )
        ...     ds = CoralDataset(p, np.array([[0, 0]]), 8, [0, 1])
        ...     patch, coord = ds[0]
        ...     (len(ds), patch.shape, patch.dtype, float(patch.max()))
        (1, (2, 8, 8), dtype('float32'), 1.0)
    """

    def __init__(
        self,
        slide_path: str | Path,
        coords: np.ndarray,
        patch_size: int,
        idxs: Any,  # noqa: ANN401 — Sequence[int] | np.ndarray
        *,
        scale: bool = True,
        cell_ids: np.ndarray | None = None,
    ) -> None:
        """Store read params; zarr opens lazily per :meth:`__getitem__`."""
        self._path = Path(slide_path)
        self._coords = np.asarray(coords)
        self._patch_size = int(patch_size)
        self._idxs = np.asarray(idxs)
        self._scale = scale
        self._cell_ids = None if cell_ids is None else np.asarray(cell_ids)

    def __len__(self) -> int:
        """Number of patches."""
        return len(self._coords)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read patch ``index`` → ``(patch, coord)`` (scaled iff ``scale``)."""
        store0 = zarr.open_array(str(self._path / "0"), mode="r")
        _, height, width = store0.shape
        x, y = int(self._coords[index][0]), int(self._coords[index][1])
        box = self._read_box(store0, x, y, height, width)
        if self._cell_ids is not None:
            mask = zarr.open_array(
                str(self._path / "cells" / "cell_mask"), mode="r"
            )
            footprint = self._read_mask_box(mask, x, y, height, width) == int(
                self._cell_ids[index]
            )
            box = box * footprint[None]
        if self._scale:
            box = box.astype(np.float32) / scaling_factor(box.dtype)
        return box, self._coords[index]

    def _read_box(
        self,
        store0: Any,  # noqa: ANN401 — a zarr array
        x: int,
        y: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """``(n_sel, size, size)`` channel box at ``(x, y)``, 0-padded."""
        size, sel = self._patch_size, self._idxs
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + size), min(height, y + size)
        out = np.zeros((len(sel), size, size), dtype=store0.dtype)
        if x1 > x0 and y1 > y0:
            # Index the subset in zarr, not after: reading `[:, ...]`
            # decompresses every channel just to discard most of them,
            # which dominates the read when the panel is a subset. Zarr
            # preserves `sel` order, so the channel order is unchanged.
            #
            # A fancy index is *slower* than a plain slice when it skips
            # nothing, though, so keep the slice when every channel is
            # wanted in order — the no-subset default.
            if len(sel) == store0.shape[0] and np.array_equal(
                sel, np.arange(len(sel))
            ):
                win = np.asarray(store0[:, y0:y1, x0:x1])
            else:
                win = np.asarray(store0[sel, y0:y1, x0:x1])
            out[
                :, y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)
            ] = win
        return out

    def _read_mask_box(
        self,
        mask: Any,  # noqa: ANN401 — a zarr array
        x: int,
        y: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """``(size, size)`` instance-label box at ``(x, y)``, 0-padded."""
        size = self._patch_size
        out = np.zeros((size, size), dtype=np.int64)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + size), min(height, y + size)
        if x1 > x0 and y1 > y0:
            out[y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)] = (
                np.asarray(mask[y0:y1, x0:x1])
            )
        return out
