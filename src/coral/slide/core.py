"""``CoralSlide`` — data container + operations for one spatial slide.

The slide owns its on-disk store, exposes data + workflow state, and
hosts the pipeline operations as methods — tissue detection, patching,
cell segmentation, and feature extraction, each landed in its sprint.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from coral.markers.additional import SlideStat
from coral.slide.state import SlideState, TaskState, load_state, save_state
from coral.slide.structure import write_structure_map
from coral.utils.errors import CoralError
from coral.utils.progress import activate_bar, bar_status, bar_write
from coral.utils.time import fmt_duration, now_iso

logger = logging.getLogger(__name__)


def _numpy_collate(items: list[tuple[Any, Any]]) -> tuple[Any, Any]:
    """Stack ``(patch, coord)`` items into ``(patches, coords)`` numpy.

    Keeps the batch as numpy (not torch tensors) so each encoder controls
    its own dtype/device — KRONOS2's z-score stays a numpy op, bit-exact.
    """
    patches = np.stack([p for p, _ in items])
    coords = np.stack([c for _, c in items])
    return patches, coords


def _patch_batches(
    ds: Any,  # noqa: ANN401
    batch_size: int,
    num_workers: int = 0,
) -> Any:  # noqa: ANN401
    """Yield ``(patch_batch, coord_batch)`` numpy batches over ``ds``.

    Uses a torch ``DataLoader`` (numpy collate) when torch is available so
    the lazy reads can overlap the GPU forward; falls back to a plain loop
    otherwise (mean_marker is torch-free). Single-pass + ordered.

    Args:
        ds: An ``CoralDataset`` (or anything with ``__len__`` +
            ``__getitem__``).
        batch_size: Patches per yielded batch.
        num_workers: Loader subprocesses reading patches ahead of the
            forward pass. ``0`` (default) reads inline in the calling
            process. Ignored by the torch-free fallback, which has no
            worker concept.
    """
    if importlib.util.find_spec("torch") is not None:
        from torch.utils.data import (  # pyright: ignore[reportMissingImports]
            DataLoader,
        )

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_numpy_collate,
        )
        yield from loader
    else:
        for start in range(0, len(ds), batch_size):
            stop = min(start + batch_size, len(ds))
            yield _numpy_collate([ds[i] for i in range(start, stop)])


def _clear_output_dir(path: Path) -> None:
    """Remove ``path`` and its contents if it exists (idempotent).

    Called before a stage rewrites its output directory so a rerun
    replaces the previous result cleanly rather than merging with stale
    files left by an earlier run (e.g. a ``*_overlay.png`` from a
    ``viz=True`` pass, or a half-written folder from a crashed run).
    """
    if path.exists():
        shutil.rmtree(path)


# OME-NGFF metadata for the single-level boolean tissue label substore.
class _TissueChannels(NamedTuple):
    """What detect_tissue loads + records for one slide.

    ``stack`` is ``[nuclear, *structural]`` at level 0 (a zero nuclear
    plane when no nuclear stain is found); ``backdrop`` is the nuclear
    channel (or structural max-projection) drawn under the overlay.
    """

    stack: np.ndarray
    backdrop: np.ndarray
    mpp: float
    nuclear_name: str | None
    nuclear_index: int | None
    structural_names: list[str]
    structural_source: str


# detect_tissue warns when coverage falls outside this band — almost
# certainly a failed threshold (all-tissue / all-background / wrong
# channel) rather than a real mask. Conservative so a small TMA core in
# a large frame does not false-warn (coverage is of the whole frame).
_MIN_PLAUSIBLE_COVERAGE = 0.01
_MAX_PLAUSIBLE_COVERAGE = 0.99

# Bumped when the on-disk features/<slug>/<encoder>/<variant> layout
# changes.
# v2: per-extractor output array renamed to a uniform "features"
# (mean_marker was "marker_features").
# v3: the encoder folder gets a `_{N}markers` (or custom --suffix)
# suffix so marker variants of one encoder don't collide.
# v4: three-level features/<slug>/<encoder>/<variant>/ — the encoder
# folder is its clean name and the marker variant (markers_{N}, or a
# custom --suffix) is a nested leaf, so model names stay clean.
# v5: the single output array is stored directly AT the variant path
# (its `.zarray` + provenance `.zattrs` live there) — the redundant
# inner "features" child is gone; marker names travel in the attrs and
# cell ids join from patches/<slug>/cell_ids.
_FEATURE_SCHEMA_VERSION = "5"


class CoralSlide:
    """A single spatial proteomics slide backed by an OME-Zarr store.

    Pipeline operations are methods on this class (TRIDENT-style).

    Attributes:
        path: On-disk path to the ``.zarr`` directory store.
        store: Opened Zarr root group.
        state: Workflow state loaded from ``slide.zarr/state.json``.
        markers: List of marker names from the slide's attrs.

    Example:
        >>> from coral import CoralSlide
        >>> slide = CoralSlide.open("tests/data/tiny_slide.zarr")
        >>> slide.markers
        ['dapi', 'cd3', 'cd20']
        >>> slide.state.tasks.ingest.status
        'pending'
    """

    def __init__(self, path: Path, store: zarr.Group) -> None:
        """Construct a slide bound to an open Zarr group.

        Most callers should use :meth:`CoralSlide.open` instead.
        """
        self._path = path
        self._store = store
        self._state: SlideState = load_state(path)

    @classmethod
    def open(cls, path: str | Path) -> CoralSlide:
        """Open an existing CORAL slide store.

        Args:
            path: Path to a ``.zarr`` directory.

        Returns:
            Opened ``CoralSlide`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.

        Example:
            >>> from coral import CoralSlide
            >>> slide = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> slide.path.name
            'tiny_slide.zarr'
        """
        path = Path(path)
        if not path.exists():
            msg = f"Slide store not found: {path}"
            raise FileNotFoundError(msg)
        if not path.is_dir():
            msg = (
                f"Slide store must be a directory (Zarr store), "
                f"got file: {path}"
            )
            raise NotADirectoryError(msg)
        store = zarr.open_group(str(path), mode="a")
        return cls(path=path, store=store)

    @property
    def path(self) -> Path:
        """On-disk path to the slide's ``.zarr`` directory.

        Returns:
            Absolute or relative ``Path`` passed to :meth:`open`.

        Example:
            >>> from coral import CoralSlide
            >>> CoralSlide.open("tests/data/tiny_slide.zarr").path.name
            'tiny_slide.zarr'
        """
        return self._path

    @property
    def store(self) -> zarr.Group:
        """The opened Zarr root group.

        Returns:
            Writable ``zarr.Group`` for this slide's store.

        Example:
            >>> from coral import CoralSlide
            >>> store = CoralSlide.open("tests/data/tiny_slide.zarr").store
            >>> "channels" in store.attrs
            True
        """
        return self._store

    @property
    def state(self) -> SlideState:
        """Slide workflow state loaded from ``state.json``.

        Returns:
            The in-memory :class:`~coral.slide.state.SlideState`.

        Example:
            >>> from coral import CoralSlide
            >>> st = CoralSlide.open("tests/data/tiny_slide.zarr").state
            >>> st.slide.name
            'tiny_slide'
        """
        return self._state

    @property
    def markers(self) -> list[str]:
        """Resolved marker names, in channel order.

        Derived from the ``channels`` attr (the single source of truth);
        empty if the slide hasn't been ingested yet.

        Returns:
            Marker name per channel, cleaned for downstream matching.

        Example:
            >>> from coral import CoralSlide
            >>> CoralSlide.open("tests/data/tiny_slide.zarr").markers
            ['dapi', 'cd3', 'cd20']
        """
        channels = self._store.attrs.get("channels")
        if not channels:
            return []
        return [ch["marker"] for ch in channels]

    @property
    def raws(self) -> list[str]:
        """Raw (original) channel names, in channel order.

        The exact names as written in ``marker_map.csv`` — for terminal
        display. (``markers`` are cleaned/lower-cased for downstream
        matching; ``raws`` keeps the user's casing.)

        Returns:
            Original channel labels, one per channel.

        Example:
            >>> from coral import CoralSlide
            >>> len(CoralSlide.open("tests/data/tiny_slide.zarr").raws)
            3
        """
        channels = self._store.attrs.get("channels")
        if not channels:
            return []
        return [ch.get("raw") or ch["marker"] for ch in channels]

    @property
    def nuclear_channel(self) -> int | None:
        """Index of the slide's nuclear-stain channel.

        The store records the nuclear stain by **name** (e.g. ``"dapi"``);
        this resolves it back to a channel index for array indexing. Set
        at ingest (inferred from the markers, or user-given) so tissue
        detection, cell segmentation, and the overlays read it instead of
        re-inferring. Falls back to on-the-fly inference when the name is
        absent or no longer on the panel. ``None`` means no nuclear stain.

        Returns:
            Zero-based channel index, or ``None`` if no nuclear stain.

        Example:
            >>> from coral import CoralSlide
            >>> s = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> s.nuclear_channel
            0
        """
        attrs = self._store.attrs
        if "nuclear_channel" in attrs:
            name = attrs["nuclear_channel"]
            if name is None:
                return None
            markers = self.markers
            if name in markers:
                return markers.index(name)
        from coral.tissue.infer import infer_dapi_channel

        return infer_dapi_channel(self)

    @property
    def kept_indices(self) -> list[int]:
        """Channel indices the user kept (``keep`` not ``no``), in order.

        ``keep`` is the per-channel analysis panel, frozen at ingest (from
        ``--subset`` + the QC auto-drop): excluded channels (blanks,
        controls) are dropped from tissue, cell, and feature extraction. A
        channel with no flag defaults to kept.

        Returns:
            Kept channel indices in panel order.

        Example:
            A 4-channel slide with its 2nd channel excluded
            (``keep="no"``) returns ``[0, 2, 3]``.
        """
        channels = self._store.attrs.get("channels")
        if not channels:
            return []
        return [i for i, ch in enumerate(channels) if ch.get("keep", True)]

    def _nuclear_image(self) -> np.ndarray:
        """Level-0 nuclear-channel image (zeros plane if no nuclear stain).

        The backdrop for the cell overlays — uses :attr:`nuclear_channel`
        directly so it never needs a structural/membrane channel.
        """
        idx = self.nuclear_channel
        if idx is None:
            _, height, width = self._store["0"].shape
            return np.zeros((int(height), int(width)), dtype=np.float32)
        return np.asarray(self.image[idx], dtype=np.float32)

    def _mpp(self) -> float:
        """Microns-per-pixel from the slide attrs, validated.

        The scale every micron-based operation (tissue morphology radii,
        the overlay scalebar, patch sizing) converts against. Reading it
        is a public-boundary check: a store without a usable ``mpp``
        raises a clear :class:`CoralError` rather than a raw
        ``KeyError``/``TypeError``.

        Returns:
            The slide's microns-per-pixel as a ``float``.

        Raises:
            CoralError: If the slide has no ``mpp`` attr, or its value is
                not a positive, finite number.

        Example:
            >>> from coral import CoralSlide
            >>> s = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> s._mpp()
            0.5
        """
        attrs = self._store.attrs
        if "mpp" not in attrs:
            raise CoralError(
                f"{self._path.name}: slide has no 'mpp' "
                "(microns-per-pixel) attribute, which is required to "
                "scale micron-based operations. Re-ingest with --mpp or "
                "a metadata CSV — typical imaging platforms run "
                "~0.25-0.5 um/pixel."
            )
        try:
            mpp = float(attrs["mpp"])
        except (TypeError, ValueError):
            raise CoralError(
                f"{self._path.name}: slide 'mpp' attribute is not a "
                f"number (got {attrs['mpp']!r})."
            ) from None
        if not np.isfinite(mpp) or mpp <= 0:
            raise CoralError(
                f"{self._path.name}: slide 'mpp' must be a positive, "
                f"finite number (got {mpp!r})."
            )
        return mpp

    @property
    def image(self) -> xr.DataArray:
        """Full-resolution image as a lazy ``(c, y, x)`` DataArray.

        Dask-backed over the level-0 Zarr dataset, with marker names as
        the ``c`` coordinate when present. CORAL's canonical pixel-read
        surface.

        Returns:
            Lazy ``xr.DataArray`` with dims ``("c", "y", "x")``.

        Example:
            >>> from coral import CoralSlide
            >>> slide = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> img = slide.image
            >>> img.dims
            ('c', 'y', 'x')
            >>> img.shape
            (3, 32, 32)
            >>> img.coords["c"].values.tolist()
            ['dapi', 'cd3', 'cd20']
        """
        data = da.from_zarr(self._store["0"])
        coords = {"c": self.markers} if self.markers else None
        return xr.DataArray(data, dims=("c", "y", "x"), coords=coords)

    @property
    def tissue_mask(self) -> xr.DataArray:
        """Level-0 boolean tissue mask as a ``(y, x)`` DataArray.

        Rasterized from ``tissue/tissue_<method>/tissue.geojson`` — the
        single source of truth for the tissue boundary
        (``tissue_mask.png`` is only a figure). When several methods
        coexist, resolution follows
        :func:`~coral.tissue.paths.resolve_tissue_method`
        (exactly one → that one; several → ``otsu`` by default). A
        boundary hand-edited in QuPath is therefore respected immediately.

        Returns:
            Boolean ``(y, x)`` mask at level 0.

        Raises:
            ValueError: If tissue detection has not run on this slide.

        Example:
            >>> from coral import CoralSlide
            >>> s = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> try:
            ...     s.tissue_mask
            ... except ValueError:
            ...     print("not detected")
            not detected
        """
        return xr.DataArray(self._tissue_mask_np(), dims=("y", "x"))

    def _tissue_mask_np(self, method: str | None = None) -> np.ndarray:
        """Level-0 tissue mask as an eager ``(y, x)`` bool numpy array.

        Rasterizes ``tissue/tissue_<method>/tissue.geojson`` (the source
        of truth). ``method=None`` uses
        :func:`~coral.tissue.paths.resolve_tissue_method`.
        """
        from coral.tissue.mask import read_tissue_geojson_mask
        from coral.tissue.paths import resolve_tissue_method, tissue_dir

        resolved = resolve_tissue_method(self._path, method)
        path = tissue_dir(self._path, resolved) / "tissue.geojson"
        _, height, width = self._store["0"].shape
        return read_tissue_geojson_mask(path, int(height), int(width))

    def write_novel_marker_stats(
        self, stats: dict[str, SlideStat], region_key: str
    ) -> None:
        """Persist this slide's novel-marker partials in the store attrs.

        The prepare pass stashes the per-marker ``(n, μ, s²)``
        it computed plus the ``region_key`` reuse signature under the root
        ``novel_marker_stats`` attr, so a resumed run can pool every slide's
        stored partials and skip recompute when the mask is unchanged. A
        later write replaces the whole record.

        Args:
            stats: Per-marker :class:`~coral.markers.additional.SlideStat`.
            region_key: The mask + scaling signature the stats were
                computed under (see
                :func:`coral.markers.additional.region_key`).

        Example:
            >>> import shutil, tempfile
            >>> from pathlib import Path
            >>> from coral import CoralSlide
            >>> from coral.markers.additional import SlideStat
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dst = Path(d) / "s.zarr"
            ...     _ = shutil.copytree("tests/data/tiny_slide.zarr", dst)
            ...     s = CoralSlide.open(dst)
            ...     s.write_novel_marker_stats(
            ...         {"FoxA1": SlideStat(9, 0.25, 0.01)}, "(5, 5):9:65535.0"
            ...     )
            ...     s.read_novel_marker_stats()[1]
            '(5, 5):9:65535.0'
        """
        self._store.attrs["novel_marker_stats"] = {
            "region_key": region_key,
            "stats": {
                m: {"n_pixels": s.n_pixels, "mean": s.mean, "var": s.var}
                for m, s in stats.items()
            },
        }

    def read_novel_marker_stats(
        self,
    ) -> tuple[dict[str, SlideStat], str] | None:
        """Read back the stored novel-marker partials + their region key.

        Returns ``None`` when the slide has no persisted partials (no
        prepare pass has run); otherwise ``(stats, region_key)`` with stats
        as :class:`~coral.markers.additional.SlideStat` per marker. The
        prepare pass reuses these only when the key matches the current
        mask and they cover the target markers.

        Example:
            >>> from coral import CoralSlide
            >>> s = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> s.read_novel_marker_stats()
        """
        raw = self._store.attrs.get("novel_marker_stats")
        if not raw:
            return None
        stats = {
            m: SlideStat(int(d["n_pixels"]), float(d["mean"]), float(d["var"]))
            for m, d in raw["stats"].items()
        }
        return stats, str(raw["region_key"])

    @property
    def cells(self) -> xr.DataArray:
        """Level-0 cell instance mask as a lazy ``(y, x)`` int32 DataArray.

        Reads the ``cells/cell_mask`` array written by
        :meth:`segment_cells` (the source of truth for cells). cellID =
        label value (0 = background).

        Returns:
            Lazy ``(y, x)`` int32 instance mask.

        Raises:
            ValueError: If cell segmentation has not run on this slide.

        Example:
            >>> from coral import CoralSlide
            >>> s = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> try:
            ...     s.cells
            ... except ValueError:
            ...     print("not segmented")
            not segmented
        """
        try:
            array = self._store["cells/cell_mask"]
        except KeyError as exc:
            msg = f"{self._path.name}: no cell mask — run segment_cells first."
            raise ValueError(msg) from exc
        return xr.DataArray(da.from_zarr(array), dims=("y", "x"))

    def _cell_centroids(self) -> np.ndarray:
        """Per-cell ``(cell_id, x, y)`` centroids at level 0, as an array.

        Reads the ``cells/cell_centroids.csv`` table written by
        :meth:`segment_cells` and returns its ``cell_id, x, y`` columns
        (level-0 pixels) as an ``(M, 3)`` float array — the input form
        for cell-centered patching. ``cell_id`` matches the
        ``cells/cell_mask`` instance label, so it joins straight back onto
        the stored mask.

        Raises:
            ValueError: If cell segmentation has not run on this slide.

        Example:
            >>> from coral import CoralSlide
            >>> s = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> try:
            ...     s._cell_centroids()
            ... except ValueError:
            ...     print("not segmented")
            not segmented
        """
        import pandas as pd

        path = self._path / "cells" / "cell_centroids.csv"
        if not path.exists():
            msg = (
                f"{self._path.name}: no cell centroids — run "
                "segment_cells first."
            )
            raise ValueError(msg)
        df = pd.read_csv(path)
        return df[["cell_id", "x", "y"]].to_numpy(dtype=float)

    def _save_state(self) -> None:
        """Atomically write the current state to disk."""
        save_state(self._path, self._state)

    @contextmanager
    def _task(self, task: TaskState) -> Iterator[TaskState]:
        """Record a step's live lifecycle around its work in ``state.json``.

        On enter: ``running`` + ``started_at`` (saved, so a concurrent
        reader sees in-progress). On clean exit: ``completed`` +
        ``completed_at``. On exception: ``error`` + the message, saved, then
        **re-raised** (so a crash is visible, not a silent ``pending``). The
        yielded ``task`` is the slot to attach ``outputs`` to before exit;
        the caller must have placed it (e.g. ``tasks.patch[slug] =
        TaskState()``) so dict slots target the right entry.
        """
        task.status = "running"
        task.started_at = now_iso()
        task.completed_at = None
        task.error = None
        self._save_state()
        try:
            yield task
        except Exception as exc:
            task.status = "error"
            task.error = str(exc)
            self._save_state()
            raise
        else:
            task.status = "completed"
            task.completed_at = now_iso()
            self._save_state()
            write_structure_map(self._path)

    def status(self) -> dict[str, dict[str, Any]]:
        """Live per-step status for this slide (read from ``state.json``).

        Returns a mapping ``{step: {status, started_at, completed_at,
        duration_s, error}}`` in pipeline order — ingest, each
        ``tissue[method]``, each per-slug ``patch[...]``, cells, then each
        ``extract[...]`` — the image's live status. Read it any time,
        mid-run or after a crash.

        Example:
            >>> import tempfile, zarr
            >>> from pathlib import Path
            >>> import numpy as np
            >>> with tempfile.TemporaryDirectory() as d:
            ...     p = Path(d) / "s.zarr"
            ...     r = zarr.open_group(str(p), mode="w")
            ...     _ = r.create_dataset(
            ...         "0", data=np.zeros((1, 4, 4), dtype="uint16")
            ...     )
            ...     r.attrs["mpp"] = 0.5
            ...     CoralSlide.open(p).status()["ingest"]["status"]
            'pending'
        """
        t = self._state.tasks
        view: dict[str, dict[str, Any]] = {
            "ingest": self._task_view(t.ingest),
        }
        for method, ts in t.tissue.items():
            view[f"tissue[{method}]"] = self._task_view(ts)
        for slug, ts in t.patch.items():
            view[f"patch[{slug}]"] = self._task_view(ts)
        view["cells"] = self._task_view(t.cells)
        for key, ts in t.extract.items():
            view[f"extract[{key}]"] = self._task_view(ts)
        return view

    @staticmethod
    def _task_view(ts: TaskState) -> dict[str, Any]:
        """One task's status fields as a plain dict (for status()/CLI)."""
        return {
            "status": ts.status,
            "started_at": ts.started_at,
            "completed_at": ts.completed_at,
            "duration_s": ts.duration_s,
            "error": ts.error,
        }

    def release(self) -> None:
        """Release resources held by this slide.

        Currently a no-op; future versions may hold open file handles,
        lazy dask arrays, GPU memory etc.

        Example:
            >>> from coral import CoralSlide
            >>> slide = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> slide.release()  # safe to call; currently a no-op
        """

    def summary(self) -> str:
        """Return a one-line human-readable summary.

        Example:
            >>> from coral import CoralSlide
            >>> s = CoralSlide.open("tests/data/tiny_slide.zarr")
            >>> "3 markers" in s.summary()
            True
        """
        st = self._state
        tasks_done = 0
        if st.tasks.ingest.status == "completed":
            tasks_done += 1
        tasks_done += sum(
            1 for t in st.tasks.tissue.values() if t.status == "completed"
        )
        tasks_done += sum(
            1 for t in st.tasks.patch.values() if t.status == "completed"
        )
        tasks_done += sum(
            1 for t in st.tasks.extract.values() if t.status == "completed"
        )
        return (
            f"{st.slide.name}{st.slide.ext}: "
            f"{len(self.markers)} markers; "
            f"{tasks_done} tasks completed"
        )

    def __repr__(self) -> str:
        """Return a compact ``CoralSlide(name=..., markers=N)`` repr."""
        n = len(self.markers)
        return f"CoralSlide(name={self._state.slide.name!r}, markers={n})"

    # ──────────── Pipeline operations ────────────

    def detect_tissue(
        self,
        model: Any,  # noqa: ANN401 — any BaseTissueSegmenter
        *,
        structural_markers: list[str] | None = None,
        viz: bool = True,
    ) -> xr.DataArray:
        """Detect tissue and write the mask, polygons, and overlay.

        Runs ``model`` at full resolution (level 0). Otsu loads the
        nuclear + structural channels (structural inferred unless
        overridden); a nuclear-only segmenter (e.g. CARTA) loads just the
        nuclear stain chosen at ingest. Writes under
        ``tissue/tissue_<method>/`` (``method`` = ``model.name``):

        - ``tissue_mask.png`` — the boolean mask (level 0)
        - ``tissue.geojson`` — polygons in level-0 coordinates
        - ``tissue_overlay.png`` — nuclear + detected boundary, when ``viz``
        - ``max_projection.png`` — the structural max-projection, only
          when ``viz`` and the segmenter uses structural channels

        Args:
            model: A tissue segmenter (e.g. ``OtsuTissueSegmenter``).
            structural_markers: Marker names to force as the structural
                channels.
            viz: Write the overlay PNG (default ``True``).

        Returns:
            The level-0 tissue mask (see :attr:`tissue_mask`).

        Example:
            >>> import shutil, tempfile
            >>> from pathlib import Path
            >>> from coral import CoralSlide
            >>> from coral.tissue import OtsuTissueSegmenter
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dst = Path(d) / "s.zarr"
            ...     _ = shutil.copytree("tests/data/tiny_slide.zarr", dst)
            ...     slide = CoralSlide.open(dst)
            ...     mask = slide.detect_tissue(OtsuTissueSegmenter())
            ...     mask.dims
            ('y', 'x')
        """
        method = model.name
        slot = self._state.tasks.tissue.setdefault(method, TaskState())
        with self._task(slot):
            start = time.perf_counter()
            chans = self._tissue_channels(
                structural_markers, uses_structural=model.uses_structural
            )
            # Preprocess once (per-channel percentile normalization), so the
            # max-projection review image shows exactly what the segmenter
            # thresholds on — the normalized nuclear + structural channels.
            normalized = np.asarray(model.preprocess(chans.stack))
            mask = np.asarray(
                model.forward(normalized, mpp=chans.mpp), dtype=bool
            )
            # The max-projection review image only means something for a
            # segmenter that thresholds on structural channels (Otsu); a
            # nuclear-only model (CARTA) has nothing to project.
            detection_max = (
                normalized.max(axis=0) if model.uses_structural else None
            )
            coverage = float(mask.mean())
            if (
                not _MIN_PLAUSIBLE_COVERAGE
                <= coverage
                <= _MAX_PLAUSIBLE_COVERAGE
            ):
                logger.warning(
                    "      tissue coverage %.1f%% looks degenerate — the "
                    "threshold may have failed (wrong nuclear channel, or "
                    "a unimodal all-tissue / all-background slide). Inspect "
                    "the overlay before trusting downstream patches.",
                    100.0 * coverage,
                )
            inputs: dict[str, Any] = {
                "nuclear_marker": chans.nuclear_name,
                "nuclear_index": chans.nuclear_index,
            }
            if model.uses_structural:
                inputs["structural_markers"] = chans.structural_names
                inputs["structural_source"] = chans.structural_source
            inputs["mpp"] = chans.mpp
            config_base = {
                "method": method,
                "segmenter": {
                    "name": method,
                    "params": model.params,
                },
                "inputs": inputs,
            }
            self._write_tissue_outputs(
                mask,
                chans.backdrop,
                chans.mpp,
                method=method,
                viz=viz,
                nuclear_name=chans.nuclear_name,
                detection_max=detection_max,
                config_base=config_base,
                elapsed=time.perf_counter() - start,
            )
        return xr.DataArray(self._tissue_mask_np(method), dims=("y", "x"))

    def _tissue_channels(
        self,
        structural_markers: list[str] | None,
        *,
        uses_structural: bool,
    ) -> _TissueChannels:
        """Pick + load the channels detect_tissue segments on.

        A nuclear-only segmenter (``uses_structural=False``, e.g. CARTA)
        loads just the nuclear stain and ignores ``structural_markers``.
        Otsu (``uses_structural=True``) also loads structural markers: the
        user's ``--structural-markers`` (validated against the panel — an
        unknown name errors), or by default whichever of pan-cytokeratin /
        vimentin / collagen IV are present. Only the needed channels are
        read from level 0. Logs the channels used and where they came from.
        """
        nuclear_idx = self.nuclear_channel
        raws = self.raws

        if not uses_structural:
            # Nuclear-only: structural markers are irrelevant to this
            # segmenter, so never resolve, load, or log them.
            if nuclear_idx is None:
                raise ValueError(
                    "no nuclear stain to segment — this segmenter is "
                    "nuclear-only, but the slide has no nuclear channel "
                    "(set one at ingest)."
                )
            nuclear_name = raws[nuclear_idx]
            bar_write(f"      nuclear={nuclear_name}")
            level0 = np.asarray(self.image[nuclear_idx])[None]
            return _TissueChannels(
                stack=level0,
                backdrop=level0[0],
                mpp=self._mpp(),
                nuclear_name=nuclear_name,
                nuclear_index=nuclear_idx,
                structural_names=[],
                structural_source="",
            )

        from coral.tissue.infer import (
            default_structural_channels,
            resolve_marker_indices,
        )

        if structural_markers is not None:
            # the user's explicit choice — resolve_marker_indices raises a
            # clear error if a name is not on the panel (the marker_map)
            struct_idxs = resolve_marker_indices(self, structural_markers)
            source = "you specified"
        else:
            kept = set(self.kept_indices)
            struct_idxs = [
                i for i in default_structural_channels(self) if i in kept
            ]
            source = "default"

        wanted = (
            [nuclear_idx] if nuclear_idx is not None else []
        ) + struct_idxs
        if not wanted:
            raise ValueError(
                "no nuclear or structural channel to segment — every "
                "usable channel is excluded in the marker_map. Keep at "
                "least one (the nuclear stain is set at ingest), or pass "
                "--structural-markers."
            )

        nuclear_name = raws[nuclear_idx] if nuclear_idx is not None else None
        structural_names = [raws[i] for i in struct_idxs]
        bar_write(
            f"      nuclear={nuclear_name or 'none'} · "
            f"structural={', '.join(structural_names) or 'none'} "
            f"({source})"
        )
        if nuclear_idx is None:
            logger.warning(
                "      no nuclear stain — segmenting on structural "
                "channels only; inspect the overlay before trusting it."
            )

        # Load only the needed channels at full resolution (level 0).
        image = self.image
        level0 = np.stack([np.asarray(image[i]) for i in wanted])
        mpp = self._mpp()

        if nuclear_idx is not None:
            stack, backdrop = level0, level0[0]
        else:
            zeros = np.zeros(level0.shape[1:], dtype=level0.dtype)
            stack = np.concatenate([zeros[None], level0], axis=0)
            backdrop = level0.max(axis=0)
        return _TissueChannels(
            stack=stack,
            backdrop=backdrop,
            mpp=mpp,
            nuclear_name=nuclear_name,
            nuclear_index=nuclear_idx,
            structural_names=structural_names,
            structural_source=source,
        )

    def _write_tissue_outputs(
        self,
        mask: np.ndarray,
        backdrop: np.ndarray,
        mpp: float,
        *,
        method: str,
        viz: bool,
        config_base: dict[str, Any],
        nuclear_name: str | None = None,
        detection_max: np.ndarray | None = None,
        elapsed: float | None = None,
    ) -> None:
        """Write ``tissue/tissue_<method>/`` outputs.

        ``tissue.geojson`` (polygons), ``tissue_mask.png``, and
        ``config.json`` (the resolved run config from ``config_base`` plus
        coverage, polygon count, and a geometry hash) always; when
        ``viz``, also ``tissue_overlay.png`` and optionally
        ``max_projection.png``.
        """
        from coral import __version__
        from coral.tissue.mask import (
            geojson_geometry_sha256,
            mask_to_geopandas,
            render_tissue_overlay,
            save_binary_mask_png,
            save_stretched_png,
            write_tissue_geojson,
        )
        from coral.tissue.paths import tissue_dir, tissue_rel

        out = tissue_dir(self._path, method)
        _clear_output_dir(out)  # rerun: replace, don't merge stale figures
        out.mkdir(parents=True, exist_ok=True)
        # Write the geojson (the source of truth) first, then derive the
        # figures from it — so on a fresh run the figures are newer than the
        # geojson, and a later geojson-only edit reads as newer than them
        # (see tissue_figures_stale).
        gdf = mask_to_geopandas(mask)
        write_tissue_geojson(gdf, out / "tissue.geojson")
        save_binary_mask_png(mask, out / "tissue_mask.png")
        if viz:
            render_tissue_overlay(
                backdrop,
                mask,
                mpp=mpp,
                nuclear_name=nuclear_name,
                save_to=out / "tissue_overlay.png",
            )
            if detection_max is not None:
                save_stretched_png(detection_max, out / "max_projection.png")

        config = {
            **config_base,
            "result": {
                "coverage_fraction": float(np.asarray(mask).mean()),
                "n_polygons": len(gdf),
            },
            "geojson_sha256": geojson_geometry_sha256(out / "tissue.geojson"),
            "coral_version": __version__,
            "created_at": now_iso(),
        }
        (out / "config.json").write_text(json.dumps(config, indent=2))

        rel = tissue_rel(method)
        task = self._state.tasks.tissue.setdefault(method, TaskState())
        task.status = "completed"
        task.completed_at = now_iso()
        task.outputs = {
            "mask": f"{rel}/tissue_mask.png",
            "geojson": f"{rel}/tissue.geojson",
            "config": f"{rel}/config.json",
            **({"overlay": f"{rel}/tissue_overlay.png"} if viz else {}),
        }
        self._save_state()
        bar_status(
            f"      {100.0 * float(np.asarray(mask).mean()):.1f}% tissue · "
            f"{len(gdf)} polygon(s)"
            f"{' · saved tissue overlay' if viz else ''}"
            f"{f' · {fmt_duration(elapsed)}' if elapsed is not None else ''}"
        )

    def import_tissue_mask(
        self,
        mask: np.ndarray,
        *,
        viz: bool = True,
        source: str | Path | None = None,
        source_kind: str | None = None,
    ) -> xr.DataArray:
        """Ingest a user-provided binary tissue mask.

        Writes under ``tissue/tissue_imported/``.

        For users who bring their own tissue segmentation instead of
        running :meth:`detect_tissue`. The mask is binarized (``> 0``),
        validated against the slide's level-0 geometry, and written via
        the same path as detection (``tissue_mask.png`` + GeoJSON +
        a boundary overlay), so downstream code sees an identical surface.

        Args:
            mask: A 2-D ``(y, x)`` mask at level 0; any ``> 0`` pixel is
                tissue. Must match the slide's level-0 height/width.
            viz: Write the boundary-overlay PNG (default ``True``).
            source: Where the mask came from, recorded in
                ``config.json`` provenance (optional).
            source_kind: ``"geojson"`` or ``"image"`` — the source's kind,
                recorded in the config provenance (optional).

        Returns:
            The level-0 tissue mask (see :attr:`tissue_mask`).

        Raises:
            ValueError: If the mask shape doesn't match level 0.

        Example:
            >>> import shutil, tempfile
            >>> from pathlib import Path
            >>> import numpy as np
            >>> from coral import CoralSlide
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dst = Path(d) / "s.zarr"
            ...     _ = shutil.copytree("tests/data/tiny_slide.zarr", dst)
            ...     s = CoralSlide.open(dst)
            ...     m = np.zeros((32, 32), dtype="uint8")
            ...     m[8:24, 8:24] = 1  # a tissue block
            ...     _ = s.import_tissue_mask(m, viz=False)
            ...     bool(s.tissue_mask.values.any())
            True
        """
        from coral.io.masks import validate_mask_shape

        method = "imported"
        _, height, width = self._store["0"].shape
        validate_mask_shape(np.asarray(mask), int(height), int(width))
        binary = np.asarray(mask) > 0
        backdrop = (
            self._patch_backdrop()[0]
            if viz
            else np.zeros((1, 1), dtype=np.uint8)
        )
        mpp = self._mpp()
        nuc_idx = self.nuclear_channel
        nuc_name = self.raws[nuc_idx] if nuc_idx is not None else None
        config_base = {
            "method": method,
            "segmenter": None,
            "import": {
                "source": str(source) if source is not None else None,
                "type": source_kind,
            },
            "inputs": {
                "nuclear_marker": nuc_name,
                "nuclear_index": nuc_idx,
                "mpp": mpp,
            },
        }
        slot = self._state.tasks.tissue.setdefault(method, TaskState())
        with self._task(slot):
            self._write_tissue_outputs(
                binary,
                backdrop,
                mpp,
                method=method,
                viz=viz,
                nuclear_name=nuc_name,
                config_base=config_base,
            )
        write_structure_map(self._path)
        return xr.DataArray(self._tissue_mask_np(method), dims=("y", "x"))

    def tissue_figures_stale(self, method: str) -> bool:
        """Whether ``tissue_<method>/`` figures are stale vs its geojson.

        ``tissue.geojson`` is the source of truth; the PNGs derive from
        it. ``True`` when the geojson has been edited (e.g. in QuPath)
        since ``tissue_mask.png`` was written, **or** when the geojson is
        present but its ``tissue_mask.png`` is missing (a figure was
        deleted) — either way a refresh is due. ``False`` when there is no
        geojson yet.

        Returns:
            ``True`` if the figures are stale or missing relative to the
            geojson.

        Example:
            >>> from coral import CoralSlide
            >>> CoralSlide.open(
            ...     "tests/data/tiny_slide.zarr"
            ... ).tissue_figures_stale("otsu")
            False
        """
        from coral.tissue.paths import tissue_dir

        out = tissue_dir(self._path, method)
        geojson = out / "tissue.geojson"
        mask_png = out / "tissue_mask.png"
        if not geojson.exists():
            return False
        if not mask_png.exists():
            return True  # geojson present but its figure is gone -> stale
        return geojson.stat().st_mtime > mask_png.stat().st_mtime

    def tissue_edited(self, method: str) -> bool:
        """Whether ``tissue_<method>/tissue.geojson`` differs from last write.

        Compares the geometry hash recorded in that method's
        ``config.json`` (``geojson_sha256``) against a fresh hash of the
        current geojson. ``True`` means the boundary was hand-edited
        (e.g. in QuPath) since CORAL wrote it. Unlike
        :meth:`tissue_figures_stale` (an mtime check), this is robust to
        file copies and version control, and ignores cosmetic re-saves
        that don't move the boundary. ``False`` when the config or the
        geojson is absent, or no hash was recorded.

        Returns:
            ``True`` if the geojson geometry hash no longer matches.

        Example:
            >>> from coral import CoralSlide
            >>> CoralSlide.open("tests/data/tiny_slide.zarr").tissue_edited(
            ...     "otsu"
            ... )
            False
        """
        import json

        from coral.tissue.mask import geojson_geometry_sha256
        from coral.tissue.paths import tissue_dir

        out = tissue_dir(self._path, method)
        config = out / "config.json"
        geojson = out / "tissue.geojson"
        if not config.exists() or not geojson.exists():
            return False
        recorded = json.loads(config.read_text()).get("geojson_sha256")
        if not recorded:
            return False
        return geojson_geometry_sha256(geojson) != recorded

    def refresh_tissue_figures(self, method: str) -> bool:
        """Rebuild PNG figures from ``tissue/tissue_<method>/tissue.geojson``.

        ``tissue.geojson`` is the single source of truth for the tissue
        boundary; ``tissue_mask.png`` and ``tissue_overlay.png`` are only
        figures derived from it. This re-rasterizes the (possibly
        QuPath-edited) geojson and rewrites those two figures — without
        touching the geojson itself.

        Returns:
            ``True`` if a ``tissue.geojson`` was present and refreshed,
            ``False`` otherwise.

        Example:
            >>> from coral import CoralSlide
            >>> CoralSlide.open(
            ...     "tests/data/tiny_slide.zarr"
            ... ).refresh_tissue_figures("otsu")
            False
        """
        from coral.tissue.mask import (
            read_tissue_geojson_mask,
            render_tissue_overlay,
            save_binary_mask_png,
        )
        from coral.tissue.paths import tissue_dir

        out = tissue_dir(self._path, method)
        geojson = out / "tissue.geojson"
        if not geojson.exists():
            return False
        _, height, width = self._store["0"].shape
        mask = read_tissue_geojson_mask(geojson, int(height), int(width))
        save_binary_mask_png(mask, out / "tissue_mask.png")
        if (out / "tissue_overlay.png").exists():
            nuc_idx = self.nuclear_channel
            render_tissue_overlay(
                self._patch_backdrop()[0],
                mask,
                mpp=self._mpp(),
                nuclear_name=self.raws[nuc_idx]
                if nuc_idx is not None
                else None,
                save_to=out / "tissue_overlay.png",
            )
        return True

    def import_cell_mask(
        self, mask: np.ndarray, *, viz: bool = True
    ) -> xr.DataArray:
        """Ingest a user-provided cell instance mask → ``cells/cell_mask``.

        For users who bring their own segmentation instead of running
        :meth:`segment_cells`. The mask's pixel values **are** the cell
        ids (0 = background) and are **preserved as-is — no relabel** —
        so external per-cell labels join. Centroids are derived from the
        mask (``regionprops``). Unlike :meth:`segment_cells`, the mask is
        **not** tissue-restricted (it is authoritative) and tissue
        detection is not required. Because nothing is filtered to
        tissue, the cell-overlay PNG never draws a tissue contour for
        an imported mask — even if ``coral tissue`` ran earlier — since
        that would visually imply a restriction that isn't applied.

        Args:
            mask: A 2-D integer ``(y, x)`` instance mask at level 0
                (pixel = cell_id, 0 = background). Must match the slide's
                level-0 height/width.
            viz: Write the cell-overlay PNG (default ``True``).

        Returns:
            The level-0 cell mask (see :attr:`cells`).

        Raises:
            ValueError: If the mask shape doesn't match level 0, or the
                mask isn't a non-negative integer array.

        Example:
            >>> import shutil, tempfile
            >>> from pathlib import Path
            >>> import numpy as np
            >>> from coral import CoralSlide
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dst = Path(d) / "s.zarr"
            ...     _ = shutil.copytree("tests/data/tiny_slide.zarr", dst)
            ...     s = CoralSlide.open(dst)
            ...     m = np.zeros((32, 32), dtype="int32")
            ...     m[10:13, 10:13] = 7
            ...     _ = s.import_cell_mask(m, viz=False)
            ...     int(s.cells.values.max())
            7
        """
        from coral.cells.table import mask_to_centroids
        from coral.io.masks import validate_mask_shape

        arr = np.asarray(mask)
        _, height, width = self._store["0"].shape
        validate_mask_shape(arr, int(height), int(width))
        if not np.issubdtype(arr.dtype, np.integer):
            msg = (
                f"{self._path.name}: cell mask must be an integer instance "
                f"mask (pixel = cell_id); got dtype {arr.dtype}."
            )
            raise ValueError(msg)
        if int(arr.min()) < 0:
            msg = f"{self._path.name}: cell mask has negative labels."
            raise ValueError(msg)
        arr = arr.astype(np.int32)
        df = mask_to_centroids(arr, self._state.slide.name)
        mpp = self._mpp()
        # Imported cell masks are not tissue-restricted (they are
        # authoritative), so the overlay never draws a tissue contour —
        # it would imply a restriction that doesn't apply.
        self._write_cell_outputs(arr, df, mpp, tissue_mask=None, viz=viz)
        write_structure_map(self._path)
        return self.cells

    def import_cell_labels(
        self,
        df: Any,  # noqa: ANN401 — a labels DataFrame
        *,
        label_set: str,
    ) -> None:
        """Ingest optional per-cell phenotype labels → ``cells/cell_labels``.

        The labels are the **ground truth** for cell-phenotyping FM
        benchmarking; they are **optional** (the cell pipeline works
        without them). ``df`` is the canonical schema — required
        ``cell_id`` (must match the stored mask labels — no
        centroid-snapping) + ``label``, optional ``x, y`` to **override**
        the mask-derived centroid (validated: a provided centroid must
        land on its own cell, else it is rejected with a warning and the
        derived centroid kept). Cell ids not present in the mask are
        dropped with a warning. ``label_set`` names this annotation
        version so several coexist.

        Args:
            df: Canonical label table (``cell_id``, ``label``, optional
                ``x``/``y``).
            label_set: Name for this annotation version (e.g. ``"c14"``).

        Raises:
            ValueError: If a required column is missing, or no cell mask
                has been imported/segmented yet.

        Example:
            Attach phenotype labels after importing a cell mask::

                import pandas as pd
                from coral import CoralSlide

                slide = CoralSlide.open("slide.zarr")
                labels = pd.DataFrame({"cell_id": [1, 2], "label": ["T", "B"]})
                slide.import_cell_labels(labels, label_set="manual")
        """
        from coral.cells.labels import write_cell_labels

        missing = {"cell_id", "label"} - set(df.columns)
        if missing:
            msg = (
                f"{self._path.name}: label table missing required "
                f"column(s) {sorted(missing)} — canonical schema is "
                "cell_id, label (optional x, y)."
            )
            raise ValueError(msg)
        try:
            mask = np.asarray(self._store["cells/cell_mask"][:])
        except KeyError as exc:
            msg = (
                f"{self._path.name}: no cell mask — import a cell mask "
                "(or run segment_cells) before importing labels."
            )
            raise ValueError(msg) from exc

        mask_ids = {int(v) for v in np.unique(mask) if v}
        in_mask = df["cell_id"].astype(int).isin(mask_ids)
        if not in_mask.all():
            logger.warning(
                "      %d label cell_ids not in the cell mask — dropped.",
                int((~in_mask).sum()),
            )
        df = df[in_mask]

        if {"x", "y"} <= set(df.columns):
            self._apply_centroid_override(df, mask)

        write_cell_labels(
            df, self._path / "cells" / "cell_labels.csv", label_set
        )
        self._state.tasks.cells.outputs["labels"] = "cells/cell_labels.csv"
        self._save_state()
        bar_status(
            f"      wrote {len(df)} labels (set={label_set!r}) → "
            "cells/cell_labels.csv"
        )

    def _apply_centroid_override(
        self,
        df: Any,  # noqa: ANN401 — a labels DataFrame
        mask: np.ndarray,
    ) -> None:
        """Override mask-derived centroids with provided ``(x, y)``.

        Each override is validated against the mask (``mask[round(y),
        round(x)] == cell_id``); valid ones update
        ``cells/cell_centroids.csv``, invalid ones are warned and the
        derived centroid kept.
        """
        import pandas as pd

        sub = df.dropna(subset=["x", "y"])
        if sub.empty:
            return
        h, w = mask.shape
        cid = sub["cell_id"].astype(int).to_numpy()
        xi = np.clip(np.round(sub["x"].astype(float)).astype(int), 0, w - 1)
        yi = np.clip(np.round(sub["y"].astype(float)).astype(int), 0, h - 1)
        ok = mask[yi, xi] == cid
        if not ok.all():
            logger.warning(
                "      %d centroid override(s) don't land on their own cell "
                "— kept the mask-derived centroid for those.",
                int((~ok).sum()),
            )
        good = sub[ok]
        if good.empty:
            return
        path = self._path / "cells" / "cell_centroids.csv"
        cents = pd.read_csv(path).set_index("cell_id")
        g = good.set_index(good["cell_id"].astype(int))
        cents.loc[g.index, "x"] = g["x"].astype(float)
        cents.loc[g.index, "y"] = g["y"].astype(float)
        cents = cents.reset_index()
        cents.to_csv(path, index=False)

    def _cell_channels(
        self,
        membrane_markers: list[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Assemble the `(nuclear, composite-membrane, mpp)` for cells.

        The nuclear channel is the store's ingest-stored stain
        (``nuclear_channel``) — fixed at ingest, not overridable per run.
        The membrane channels are the structural family (via
        ``infer_structural_channels``) or ``membrane_markers``. Loads them
        at level 0 and returns the nuclear channel + the composite
        membrane = **sum** of the membrane channels (float32; Cellpose
        normalizes internally).

        Raises:
            ValueError: If the store has no nuclear channel, or no membrane
                channels are found/supplied.
        """
        from coral.tissue.infer import (
            infer_structural_channels,
            resolve_marker_indices,
        )

        nuclear_idx = self.nuclear_channel
        if nuclear_idx is None:
            msg = (
                f"{self._path.name}: no nuclear channel — re-ingest with a "
                "nuclear marker in the panel."
            )
            raise ValueError(msg)

        if membrane_markers is not None:
            membrane_idxs = resolve_marker_indices(self, membrane_markers)
        else:
            membrane_idxs = infer_structural_channels(self)
        kept = set(self.kept_indices)
        membrane_idxs = [i for i in membrane_idxs if i in kept]
        if not membrane_idxs:
            msg = (
                f"{self._path.name}: no membrane channels found — pass "
                "membrane_markers=..."
            )
            raise ValueError(msg)

        image = self.image
        nuclear = np.asarray(image[nuclear_idx], dtype=np.float32)
        membrane = np.zeros(nuclear.shape, dtype=np.float32)
        for idx in membrane_idxs:
            membrane += np.asarray(image[idx], dtype=np.float32)
        mpp = self._mpp()
        raws = self.raws
        source = "specified" if membrane_markers is not None else "default"
        bar_write(
            f"      nuclear={raws[nuclear_idx]} · "
            f"membrane={', '.join(raws[i] for i in membrane_idxs)} "
            f"({source})"
        )
        return nuclear, membrane, mpp

    def segment_cells(
        self,
        model: Any,  # noqa: ANN401 — any BaseCellSegmenter
        *,
        membrane_markers: list[str] | None = None,
        viz: bool = True,
        restrict_to_tissue: bool = True,
        tissue_method: str | None = None,
    ) -> xr.DataArray:
        """Segment cells and write the outputs.

        By default segmentation is **restricted to the tissue mask** from
        :meth:`detect_tissue` (cropped to the tissue bbox for speed; cells
        whose centroid falls outside tissue are dropped). Set
        ``restrict_to_tissue=False`` to segment the **whole image** with no
        tissue mask (every cell kept) — for a core or ROI that fills the
        frame, or when tissue detection is unreliable; this can be slow or
        memory-heavy on a large whole-slide image. Writes:

        - ``cells/cell_mask`` — int32 instance mask, level 0 (source of truth)
        - ``cells/cell_centroids.csv`` — canonical per-cell table
        - ``cells/cell_overlay.png`` — outline overlay (when ``viz``)

        The nuclear channel is the store's ingest-stored stain (fixed at
        ingest, not overridable here); ``membrane_markers`` picks the
        membrane channels (default: the structural family).

        Args:
            model: A cell segmenter (e.g. ``CellposeSegmenter``).
            membrane_markers: Force the membrane channels by marker name.
            viz: Write the overlay PNG (default ``True``).
            restrict_to_tissue: Restrict segmentation to the tissue mask
                (default ``True``); ``False`` segments the whole image.
            tissue_method: Which ``tissue/tissue_<method>/`` mask to use
                when ``restrict_to_tissue`` (``None`` auto-resolves).

        Returns:
            The level-0 cell mask (see :attr:`cells`).

        Raises:
            ValueError: If ``restrict_to_tissue`` and tissue detection has
                not run on this slide.

        Example:
            After tissue detection, segment with Cellpose::

                from coral import CoralSlide
                from coral.cells import CellposeSegmenter

                slide = CoralSlide.open("slide.zarr")
                mask = slide.segment_cells(CellposeSegmenter())
        """
        from coral.cells.table import mask_to_centroids

        tissue = (
            self._tissue_mask_np(tissue_method) if restrict_to_tissue else None
        )
        with self._task(self._state.tasks.cells):
            start = time.perf_counter()
            nuclear, membrane, mpp = self._cell_channels(membrane_markers)

            if tissue is not None:
                rows = np.flatnonzero(tissue.any(axis=1))
                cols = np.flatnonzero(tissue.any(axis=0))
                if rows.size == 0:
                    msg = (
                        f"{self._path.name}: empty tissue mask — "
                        f"nothing to segment."
                    )
                    raise ValueError(msg)
                y0, y1 = int(rows[0]), int(rows[-1]) + 1
                x0, x1 = int(cols[0]), int(cols[-1]) + 1
                raw = np.asarray(
                    model.segment(
                        nuclear[y0:y1, x0:x1], membrane[y0:y1, x0:x1], mpp=mpp
                    ),
                    dtype=np.int32,
                )
                full = np.zeros(tissue.shape, dtype=np.int32)
                full[y0:y1, x0:x1] = raw
                mask, df = self._restrict_to_tissue(full, tissue)
            else:
                mask = np.asarray(
                    model.segment(nuclear, membrane, mpp=mpp), dtype=np.int32
                )
                df = mask_to_centroids(mask, self._state.slide.name)

            self._write_cell_outputs(
                mask,
                df,
                mpp,
                tissue_mask=tissue,
                viz=viz,
                elapsed=time.perf_counter() - start,
            )
        return self.cells

    def _restrict_to_tissue(
        self, full: np.ndarray, tissue: np.ndarray
    ) -> tuple[np.ndarray, Any]:
        """Drop cells whose centroid is outside tissue + relabel ``1..N``.

        Vectorized (one ``regionprops`` via ``mask_to_centroids`` + a
        remap lookup), so it scales to whole-slide masks. Returns the
        relabelled ``(mask, table)`` — the table is ``cell_id, x, y,
        slide`` with consecutive ids matching the mask.
        """
        from coral.cells.table import mask_to_centroids

        df = mask_to_centroids(full, self._state.slide.name)
        height, width = tissue.shape
        if len(df):
            yi = np.clip(
                np.round(np.asarray(df["y"])).astype(int), 0, height - 1
            )
            xi = np.clip(
                np.round(np.asarray(df["x"])).astype(int), 0, width - 1
            )
            df = df[tissue[yi, xi]].reset_index(drop=True)

        old_ids = np.asarray(df["cell_id"], dtype=np.int32)
        new_ids = np.arange(1, len(df) + 1, dtype=np.int32)
        remap = np.zeros(int(full.max()) + 1, dtype=np.int32)
        remap[old_ids] = new_ids
        mask = remap[full]
        df = df.assign(cell_id=new_ids)
        return mask, df

    def _write_cell_outputs(
        self,
        mask: np.ndarray,
        df: Any,  # noqa: ANN401 — a cell-table DataFrame
        mpp: float,
        *,
        tissue_mask: np.ndarray | None = None,
        viz: bool,
        elapsed: float | None = None,
    ) -> None:
        """Write the flat ``cells/`` outputs (mask + table + overlay) + state.

        The int32 instance mask (pixel = cell_id, 0 = background) is the
        source of truth, stored as a single ``cells/cell_mask`` array; the
        centroid table and overlay are derived from it. ``tissue_mask``
        (when given) is drawn as a contour on the overlay — optional because
        an imported cell mask need not be tissue-restricted.
        """
        from coral.cells.table import write_cell_table

        height, width = int(mask.shape[0]), int(mask.shape[1])
        # rerun: clean-replace cells/. A re-segmentation reassigns cell
        # ids, so any imported cell_labels.csv (keyed on the old ids) is
        # stale — drop it, loudly, so the user re-imports against the new
        # mask.
        cells_dir = self._path / "cells"
        if (cells_dir / "cell_labels.csv").exists():
            logger.warning(
                "      removing stale cells/cell_labels.csv — cell ids are "
                "reassigned by re-segmentation; re-import labels afterwards."
            )
        _clear_output_dir(cells_dir)
        cells = self._store.require_group("cells")
        cells.create_dataset(
            "cell_mask",
            data=mask,
            chunks=(min(512, height), min(512, width)),
            overwrite=True,
        )

        out = self._path / "cells"
        out.mkdir(exist_ok=True)
        write_cell_table(df, out / "cell_centroids.csv")

        outputs = {
            "mask": "cells/cell_mask",
            "centroids": "cells/cell_centroids.csv",
        }
        if viz:
            from coral.cells.viz import render_cell_overlay

            render_cell_overlay(
                self._nuclear_image(),
                mask,
                mpp=mpp,
                save_to=out / "cell_overlay.png",
                tissue_mask=tissue_mask,
                n_cells=len(df),
            )
            outputs["overlay"] = "cells/cell_overlay.png"

        self._state.tasks.cells.status = "completed"
        self._state.tasks.cells.completed_at = now_iso()
        self._state.tasks.cells.outputs = outputs
        self._save_state()
        bar_status(
            f"      {len(df)} cells"
            f"{' · saved cell overlay' if viz else ''}"
            f"{f' · {fmt_duration(elapsed)}' if elapsed is not None else ''}"
        )

    def visualize_cells_inset(
        self,
        top_left: tuple[int, int],
        box_size: tuple[int, int],
        *,
        save_to: str | Path | None = None,
        show_cells: bool = True,
        cell_color: str = "cyan",
        labels_csv: str | Path | None = None,
        label_color: str = "yellow",
    ) -> Path:
        """Render a whole-slide + zoomed-inset figure of the cells.

        For inspecting cells on large slides where the full overlay is
        too dense: shows the whole slide with the inset box on the left
        and the box at full resolution (with cell outlines) on the
        right. Optionally labels cells from a CSV.

        Args:
            top_left: Inset ``(x, y)`` top-left in level-0 pixels.
            box_size: Inset ``(height, width)`` in level-0 pixels.
            save_to: Output PNG path. Defaults to
                ``cells/inset_<x>_<y>_<w>x<h>.png`` in the slide store.
            show_cells: Draw cell outlines in the inset.
            cell_color: Colour of the box + cell outlines.
            labels_csv: Optional CSV with ``cell_id,label`` columns;
                cells in the box are annotated with their label.
            label_color: Colour of the per-cell label text.

        Returns:
            The path the figure was written to.

        Raises:
            ValueError: If cells have not been segmented, or the box is
                out of bounds / non-positive.

        Example:
            >>> import shutil, tempfile
            >>> from pathlib import Path
            >>> from coral import CoralSlide
            >>> from coral.tissue import OtsuTissueSegmenter
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dst = Path(d) / "s.zarr"
            ...     _ = shutil.copytree("tests/data/tiny_slide.zarr", dst)
            ...     slide = CoralSlide.open(dst)
            ...     _ = slide.detect_tissue(OtsuTissueSegmenter(), viz=False)
            ...     try:
            ...         slide.visualize_cells_inset((0, 0), (16, 16))
            ...     except ValueError:
            ...         print("not segmented")
            not segmented
        """
        import pandas as pd

        from coral.cells.viz import render_cell_inset

        mask = np.asarray(self.cells)  # raises if cells not segmented
        nuclear = self._nuclear_image()
        mpp = self._mpp()
        labels = pd.read_csv(labels_csv) if labels_csv is not None else None

        x, y = int(top_left[0]), int(top_left[1])
        box_h, box_w = int(box_size[0]), int(box_size[1])
        out_path = Path(
            self._path / "cells" / f"inset_{x}_{y}_{box_w}x{box_h}.png"
            if save_to is None
            else save_to
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render_cell_inset(
            nuclear,
            mask,
            top_left=top_left,
            box_size=box_size,
            mpp=mpp,
            save_to=out_path,
            show_cells=show_cells,
            cell_color=cell_color,
            labels=labels,
            label_color=label_color,
        )
        return out_path

    def extract_patches(
        self,
        config: Any,  # noqa: ANN401 — a PatchConfig
        *,
        viz: bool = True,
        tissue_method: str | None = None,
    ) -> np.ndarray:
        """Generate + persist grid patch coordinates for the slide.

        Patches are produced at the slide's **base mpp** (level 0) — the
        resolution KRONOS encodes at. ``config.target_mpp`` is resolved
        against that base: ``None`` means base; a coarser value is a
        future feature (custom-mpp downsampling is deferred); a finer
        value is impossible. Requires tissue detection to have run.
        Grid mode writes ``patches/<slug>/{coords, tissue_prop,
        config.json}`` for the **whole** grid — every patch is kept and
        its tissue fraction stored beside it, so a tissue cut-off is a
        downstream choice rather than a re-patch;
        ``mode="cell_centered"`` writes ``{coords, cell_ids,
        config.json}`` (one patch per cell, boxed on its centroid).
        Unless ``viz`` is false a ``patch_overlay.png`` is written;
        ``state.tasks.patch[<slug>]`` is recorded. The tissue method used
        is stored in ``config.json`` for provenance — review tissue
        overlays, then pick one method before patching.

        Args:
            config: A ``PatchConfig`` (size, stride/overlap, target_mpp,
                mode).
            viz: Write the patch-grid overlay PNG (default ``True``).
            tissue_method: Which ``tissue/tissue_<method>/`` mask to use.
                ``None`` auto-resolves (see
                :func:`~coral.tissue.paths.resolve_tissue_method`).

        Returns:
            The ``(N, 2)`` int64 level-0 ``(x, y)`` coordinates of every
            patch in the grid.

        Raises:
            NotImplementedError: If ``config.target_mpp`` is coarser
                than the slide's base mpp (custom-mpp downsampling is a
                future feature).
            ValueError: If ``target_mpp`` is finer than base, tissue
                detection has not run, or (cell-centered) cells have not
                been segmented.

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
            ...     coords = slide.extract_patches(cfg, viz=False)
            ...     coords.shape[1]
            2
        """
        from coral.patch.core import Patcher
        from coral.tissue.paths import resolve_tissue_method

        base_mpp = self._mpp()
        resolved_mpp = self._resolve_patch_mpp(config.target_mpp, base_mpp)
        slug = config.resolved_slug(resolved_mpp)
        if config.mode == "cell_centered":
            # Cell-centered patches don't filter on tissue; resolve
            # only for provenance / optional overlay contour.
            try:
                resolved_tissue = resolve_tissue_method(
                    self._path, tissue_method
                )
            except ValueError:
                resolved_tissue = tissue_method or "none"
        else:
            resolved_tissue = resolve_tissue_method(self._path, tissue_method)
        slot = self._state.tasks.patch.setdefault(slug, TaskState())
        with self._task(slot):
            # rerun: replace patches/<slug>/ cleanly (e.g. drop a stale
            # patch_overlay.png left by a prior viz=True run).
            _clear_output_dir(self._path / "patches" / slug)
            start = time.perf_counter()
            if config.mode == "cell_centered":
                coords, cell_ids, _dropped = Patcher(
                    self, config, tissue_method=resolved_tissue
                ).cell_centered_coords()
                self._write_cell_patch_outputs(
                    slug,
                    config,
                    resolved_mpp,
                    coords,
                    cell_ids,
                    viz=viz,
                    tissue_method=resolved_tissue,
                    elapsed=time.perf_counter() - start,
                )
            else:
                coords, tissue_prop = Patcher(
                    self, config, tissue_method=resolved_tissue
                ).generate()
                self._write_patch_outputs(
                    slug,
                    config,
                    resolved_mpp,
                    coords,
                    tissue_prop,
                    viz=viz,
                    tissue_method=resolved_tissue,
                    elapsed=time.perf_counter() - start,
                )
        return coords

    def _resolve_patch_mpp(
        self, target_mpp: float | None, base_mpp: float
    ) -> float:
        """Resolve a patch ``target_mpp`` against the slide's base mpp.

        ``None`` → base. Within 0.1% of base → base. Coarser → future
        feature. Finer → impossible.
        """
        if target_mpp is None:
            return base_mpp
        ratio = target_mpp / base_mpp
        if ratio > 1.001:
            msg = (
                f"{self._path.name}: target_mpp={target_mpp:g} is coarser "
                f"than the slide's base mpp ({base_mpp:g}); custom-mpp "
                "downsampling is a future feature — patch at base mpp "
                "(target_mpp=None) for now."
            )
            raise NotImplementedError(msg)
        if ratio < 0.999:
            msg = (
                f"{self._path.name}: target_mpp={target_mpp:g} is finer than "
                f"the slide's base mpp ({base_mpp:g}); upsampling beyond the "
                "captured resolution is impossible."
            )
            raise ValueError(msg)
        return base_mpp

    def _write_patch_outputs(
        self,
        slug: str,
        config: Any,  # noqa: ANN401 — a PatchConfig
        resolved_mpp: float,
        coords: np.ndarray,
        tissue_prop: np.ndarray,
        *,
        viz: bool,
        tissue_method: str,
        elapsed: float | None = None,
    ) -> None:
        """Write coords/tissue_prop/config.json (+ viz) + update state."""
        from coral.patch.viz import render_patch_overlay

        n = int(coords.shape[0])
        grp = self._store.require_group("patches").require_group(slug)
        grp.create_dataset(
            "coords",
            data=coords.astype(np.int32),
            chunks=(max(1, min(n, 1 << 16)), 2),
            overwrite=True,
        )
        grp.create_dataset(
            "tissue_prop",
            data=tissue_prop.astype(np.float32),
            chunks=(max(1, min(n, 1 << 16)),),
            overwrite=True,
        )
        config_doc = {
            **config.model_dump(mode="json"),
            "resolved_mpp": resolved_mpp,
            "slug": slug,
            "tissue_method": tissue_method,
        }
        slug_dir = self._path / "patches" / slug
        (slug_dir / "config.json").write_text(json.dumps(config_doc, indent=2))

        outputs = {
            "coords": f"patches/{slug}/coords",
            "tissue_prop": f"patches/{slug}/tissue_prop",
            "config": f"patches/{slug}/config.json",
        }
        if viz:
            backdrop, left_title = self._patch_backdrop()
            render_patch_overlay(
                backdrop,
                coords,
                config.patch_size,
                mpp=resolved_mpp,
                save_to=slug_dir / "patch_overlay.png",
                tissue_prop=tissue_prop,
                tissue_mask=self._tissue_mask_np(tissue_method),
                effective_overlap=1.0
                - config.effective_stride / config.patch_size,
                left_title=left_title,
            )
            outputs["overlay"] = f"patches/{slug}/patch_overlay.png"

        task = self._state.tasks.patch.setdefault(slug, TaskState())
        task.status = "completed"
        task.completed_at = now_iso()
        task.outputs = outputs
        self._save_state()
        n_tissue = int((np.asarray(tissue_prop) > 0).sum())
        bar_status(
            f"      {n} patches ({n_tissue} with tissue)"
            f"{' · saved patch overlay' if viz else ''}"
            f"{f' · {fmt_duration(elapsed)}' if elapsed is not None else ''}"
        )

    def _write_cell_patch_outputs(
        self,
        slug: str,
        config: Any,  # noqa: ANN401 — a PatchConfig
        resolved_mpp: float,
        coords: np.ndarray,
        cell_ids: np.ndarray,
        *,
        viz: bool,
        tissue_method: str,
        elapsed: float | None = None,
    ) -> None:
        """Write coords/cell_ids/config.json (+ viz) + update state."""
        from coral.patch.viz import render_cell_patch_overlay

        n = int(coords.shape[0])
        grp = self._store.require_group("patches").require_group(slug)
        grp.create_dataset(
            "coords",
            data=coords.astype(np.int32),
            chunks=(max(1, min(n, 1 << 16)), 2),
            overwrite=True,
        )
        grp.create_dataset(
            "cell_ids",
            data=cell_ids.astype(np.int32),
            chunks=(max(1, min(n, 1 << 16)),),
            overwrite=True,
        )
        config_doc = {
            **config.model_dump(mode="json"),
            "resolved_mpp": resolved_mpp,
            "slug": slug,
            "tissue_method": tissue_method,
        }
        slug_dir = self._path / "patches" / slug
        (slug_dir / "config.json").write_text(json.dumps(config_doc, indent=2))

        outputs = {
            "coords": f"patches/{slug}/coords",
            "cell_ids": f"patches/{slug}/cell_ids",
            "config": f"patches/{slug}/config.json",
        }
        if viz:
            backdrop, left_title = self._patch_backdrop()
            cell_mask = np.asarray(self._store["cells/cell_mask"][:])
            try:
                tissue_mask: np.ndarray | None = self._tissue_mask_np(
                    tissue_method
                )
            except ValueError:
                tissue_mask = None
            render_cell_patch_overlay(
                backdrop,
                coords,
                config.patch_size,
                mpp=resolved_mpp,
                save_to=slug_dir / "patch_overlay.png",
                tissue_mask=tissue_mask,
                cell_ids=cell_ids,
                cell_mask=cell_mask,
                left_title=left_title,
            )
            outputs["overlay"] = f"patches/{slug}/patch_overlay.png"

        task = self._state.tasks.patch.setdefault(slug, TaskState())
        task.status = "completed"
        task.completed_at = now_iso()
        task.outputs = outputs
        self._save_state()
        bar_status(
            f"      {n} cell patches"
            f"{' · saved patch overlay' if viz else ''}"
            f"{f' · {fmt_duration(elapsed)}' if elapsed is not None else ''}"
        )

    def _patch_backdrop(self) -> tuple[np.ndarray, str]:
        """Level-0 backdrop image + its labelled panel title.

        Prefers the inferred nuclear channel (titled "Nuclear channel
        (<marker>)"); falls back to channel 0 ("Backdrop (<marker>)")
        when no nuclear marker is found.
        """
        markers = self.markers
        idx = self.nuclear_channel
        if idx is not None:
            name = markers[idx] if idx < len(markers) else f"channel {idx}"
            return np.asarray(self.image[idx]), f"Nuclear channel ({name})"
        name = markers[0] if markers else "channel 0"
        return np.asarray(self.image[0]), f"Backdrop ({name})"

    def _feature_location(
        self, encoder: str, slug: str, used: list[str], suffix: str | None
    ) -> tuple[str, str, Path]:
        """Resolve where an extractor's features for a selection live.

        The single source of truth for the ``features/<slug>/<encoder>/
        <variant>`` layout (the variant path is the feature array), so
        :meth:`encode_features` (write) and :meth:`features` (read) derive
        the same path and can never drift.
        The variant leaf is the custom ``suffix`` verbatim, else
        ``markers_<N>`` for the selected marker count.

        Args:
            encoder: The extractor's registry name (e.g. ``mean_marker``).
            slug: The patch-set slug the features belong to.
            used: The selected marker names — its length names the variant.
            suffix: A custom variant name, or ``None`` for ``markers_<N>``.

        Returns:
            ``(variant, task_key, folder)`` — the variant leaf, the
            ``state.json`` task key ``"<slug>/<encoder>/<variant>"``, and
            the on-disk path where the variant's feature array lives.

        Example:
            >>> import tempfile
            >>> from pathlib import Path
            >>> import numpy as np, zarr
            >>> with tempfile.TemporaryDirectory() as d:
            ...     p = Path(d) / "s.zarr"
            ...     r = zarr.open_group(str(p), mode="w")
            ...     _ = r.create_dataset("0", data=np.zeros((1, 2, 2)))
            ...     r.attrs["mpp"] = 0.5
            ...     s = CoralSlide.open(p)
            ...     variant, key, _ = s._feature_location(
            ...         "mean_marker", "0.5mpp_256px", ["DAPI", "CD3"], None
            ...     )
            ...     (variant, key)
            ('markers_2', '0.5mpp_256px/mean_marker/markers_2')
        """
        variant = suffix if suffix else f"markers_{len(used)}"
        task_key = f"{slug}/{encoder}/{variant}"
        folder = self._path / "features" / slug / encoder / variant
        return variant, task_key, folder

    def encode_features(
        self,
        extractor: Any,  # noqa: ANN401 — an CoralEncoder
        config: Any,  # noqa: ANN401 — a PatchConfig identifying the set
        *,
        channels: Any = None,  # noqa: ANN401 — a Selection, or None for all
        batch_size: int = 16,
        suffix: str | None = None,
        num_workers: int = 0,
    ) -> np.ndarray:
        """Encode a stored patch set into per-patch features.

        Reads the patch coords written by :meth:`extract_patches` for
        ``config`` (``patches/<slug>/``), materialises each patch's
        level-0 pixels, runs ``extractor``, and writes its single output
        array directly at ``features/<slug>/<extractor>/<variant>`` — a
        standalone zarr array whose ``.zattrs`` carry the marker names and
        provenance; read it back with :meth:`features`. **Grid** patches
        are read as the raw box; **cell-centered** patches are isolated to
        the target cell.

        Markers default to all of :attr:`markers`; a channel ``Selection``
        selects a subset by glob. Completed sets are skipped; re-encode by
        deleting the set or using a fresh store.

        Args:
            extractor: A ``CoralEncoder`` (e.g. ``mean_marker``).
            config: The ``PatchConfig`` whose patch set to encode.
            channels: Optional channel ``Selection`` (marker subset).
            batch_size: Patches per encode call.
            suffix: Variant-folder name nested under the encoder. Default
                ``markers_<N>`` (selected marker count) so marker variants
                of one encoder don't collide; a custom value is used as-is.
            num_workers: Loader subprocesses reading patches ahead of the
                forward pass. ``0`` (default) reads inline, so the GPU
                idles during reads; raising it overlaps the two. Gains
                plateau quickly — the ceiling is read bandwidth, not
                core count.

        Returns:
            ``(n_patches, n_markers)`` feature array.

        Raises:
            ValueError: If the patch set does not exist, or it was
                already encoded with a different marker set (pass
                ``suffix`` to keep both).

        Example:
            >>> import tempfile, json
            >>> from pathlib import Path
            >>> import numpy as np, zarr
            >>> from coral.config.patch import PatchConfig
            >>> from coral.features import MeanMarkerExtractor
            >>> with tempfile.TemporaryDirectory() as d:
            ...     p = Path(d) / "s.zarr"
            ...     r = zarr.open_group(str(p), mode="w")
            ...     _ = r.create_dataset(
            ...         "0", data=np.ones((2, 16, 16), dtype="uint16")
            ...     )
            ...     r.attrs["channels"] = [
            ...         {"marker": m.lower(), "raw": m, "match": "RESOLVED"}
            ...         for m in ["DAPI", "CD3"]
            ...     ]
            ...     r.attrs["mpp"] = 0.5
            ...     cfg = PatchConfig(mode="grid", patch_size=8)
            ...     slug = cfg.resolved_slug(0.5)
            ...     g = r.require_group("patches").require_group(slug)
            ...     _ = g.create_dataset(
            ...         "coords", data=np.array([[0, 0]], dtype="int32")
            ...     )
            ...     s = CoralSlide.open(p)
            ...     out = s.encode_features(MeanMarkerExtractor(), cfg)
            ...     out.shape
            (1, 2)
        """
        base_mpp = self._mpp()
        resolved_mpp = self._resolve_patch_mpp(config.target_mpp, base_mpp)
        slug = config.resolved_slug(resolved_mpp)
        if not (self._path / "patches" / slug / "coords").exists():
            msg = (
                f"{self._path.name}: no patch set {slug!r} — "
                f"run extract_patches first."
            )
            raise ValueError(msg)

        name = extractor.name
        idxs, used = self._resolve_markers(channels)
        variant, task_key, folder_path = self._feature_location(
            name, slug, used, suffix
        )

        if folder_path.exists():
            existing = set(
                zarr.open_array(str(folder_path), mode="r").attrs.get(
                    "markers_used", []
                )
            )
            if existing != set(used):
                msg = (
                    f"{self._path.name}: features/{slug}/{name}/{variant} "
                    f"already exists but was extracted with a different "
                    f"marker set. To keep both, rename the --subset file so "
                    f"its stem names a different folder, or use a separate "
                    f"--job-dir."
                )
                raise ValueError(msg)
            bar_status(f"skipped — already done ({task_key})")
            return self._feature_array(folder_path)

        slot = self._state.tasks.extract.setdefault(task_key, TaskState())
        with self._task(slot):
            self._check_required_markers(extractor, used)
            extractor.warn_for_patches(int(config.patch_size), config.mode)

            coords = np.asarray(
                zarr.open_array(
                    str(self._path / "patches" / slug / "coords"), mode="r"
                )
            )
            cell_ids = self._read_patch_cell_ids(slug)
            unit = "cell" if cell_ids is not None else "patch"
            nuclear_marker = (
                self.markers[self.nuclear_channel]
                if self.nuclear_channel is not None
                else None
            )
            start = time.perf_counter()

            def _finish(features: np.ndarray) -> None:
                self._write_feature_outputs(
                    slug,
                    name,
                    variant,
                    extractor,
                    features,
                    used,
                    cell_ids=cell_ids,
                )
                bar_status(
                    f"      wrote features/{slug}/{name}/{variant} · "
                    f"{features.shape} · "
                    f"{fmt_duration(time.perf_counter() - start)}"
                )

            features = self._extract_patch_features(
                extractor,
                coords,
                int(config.patch_size),
                idxs,
                used,
                batch_size,
                cell_ids=cell_ids,
                unit=unit,
                nuclear_marker=nuclear_marker,
                finish=_finish,
                num_workers=num_workers,
            )
        return features

    def features(
        self,
        extractor: Any,  # noqa: ANN401 — name str or CoralEncoder
        config: Any,  # noqa: ANN401 — a PatchConfig or its slug string
        *,
        channels: Any = None,  # noqa: ANN401 — a Selection, or None for all
        suffix: str | None = None,
    ) -> xr.Dataset:
        """Read stored features as a labeled :class:`xarray.Dataset`.

        Returns the extractor's output for ``config``'s patch set with
        named dims and coordinates: a ``marker`` coordinate when the
        output is per-marker, ``cell_id`` (joined from
        ``patches/<slug>/cell_ids``) for cell-centered patches, and grid
        ``x``/``y`` joined from ``patches/<slug>/coords`` (the one source
        of truth — positions are not duplicated in the feature store).
        The stored array carries ``_ARRAY_DIMENSIONS``, so it is
        self-describing; this accessor additionally labels the markers
        and wires the grid positions and provenance.

        Pass the **same** ``extractor`` and marker selection you extracted
        with (``channels``/``suffix``); this resolves the stored
        ``features/<slug>/<encoder>/<variant>`` array the way
        :meth:`encode_features` wrote it. If that variant is absent, the
        error lists the variants that do exist.

        Args:
            extractor: Extractor name (``"mean_marker"``) or instance — the
                clean name, not the on-disk variant folder.
            config: The ``PatchConfig`` whose features to read, or its
                already-resolved slug string.
            channels: The channel ``Selection`` used at extraction
                (``None`` = all markers), to resolve the ``markers_<N>``
                variant.
            suffix: A custom variant name if one was passed to
                ``encode_features`` (else the ``markers_<N>`` default).

        Returns:
            ``xarray.Dataset`` — each output as a data variable with dims
            ``(patch, ...)``, plus coordinates and provenance ``.attrs``.

        Raises:
            ValueError: If no features are stored for that (config,
                extractor, variant) — the message lists what is available.

        Example:
            >>> import tempfile
            >>> from pathlib import Path
            >>> import numpy as np, zarr
            >>> from coral.config.patch import PatchConfig
            >>> from coral.features import MeanMarkerExtractor
            >>> with tempfile.TemporaryDirectory() as d:
            ...     p = Path(d) / "s.zarr"
            ...     r = zarr.open_group(str(p), mode="w")
            ...     _ = r.create_dataset(
            ...         "0", data=np.ones((2, 16, 16), dtype="uint16")
            ...     )
            ...     r.attrs["channels"] = [
            ...         {"marker": m.lower(), "raw": m, "match": "RESOLVED"}
            ...         for m in ["DAPI", "CD3"]
            ...     ]
            ...     r.attrs["mpp"] = 0.5
            ...     cfg = PatchConfig(mode="grid", patch_size=8)
            ...     g = r.require_group("patches").require_group(
            ...         cfg.resolved_slug(0.5)
            ...     )
            ...     _ = g.create_dataset(
            ...         "coords", data=np.array([[0, 0]], dtype="int32")
            ...     )
            ...     s = CoralSlide.open(p)
            ...     _ = s.encode_features(MeanMarkerExtractor(), cfg)
            ...     ds = s.features(MeanMarkerExtractor(), cfg)
            ...     ds["features"].dims
            ('patch', 'marker')
        """
        name = extractor if isinstance(extractor, str) else extractor.name
        if isinstance(config, str):
            slug = config
        else:
            base_mpp = self._mpp()
            resolved_mpp = self._resolve_patch_mpp(config.target_mpp, base_mpp)
            slug = config.resolved_slug(resolved_mpp)
        _, used = self._resolve_markers(channels)
        variant, _, grp_path = self._feature_location(name, slug, used, suffix)
        if not grp_path.exists():
            enc_dir = self._path / "features" / slug / name
            available = (
                sorted(v.name for v in enc_dir.iterdir() if v.is_dir())
                if enc_dir.is_dir()
                else []
            )
            msg = f"{self._path.name}: no features {slug}/{name}/{variant}"
            if available:
                msg += f"; available variants: {available}"
            raise ValueError(msg)
        arr = zarr.open_array(str(grp_path), mode="r")
        outputs: dict[str, list[str]] = dict(arr.attrs["outputs"])
        out_key, dims = next(iter(outputs.items()))
        data_vars: dict[str, Any] = {out_key: (list(dims), np.asarray(arr))}
        coords: dict[str, Any] = {}
        if "marker" in dims:
            names = arr.attrs.get("markers_used", [])
            coords["marker"] = np.asarray(names).astype(str)
        if arr.attrs.get("patch_mode") == "cell":
            cell_ids = self._read_patch_cell_ids(slug)
            if cell_ids is not None:
                coords["cell_id"] = ("patch", cell_ids)
        else:
            cpath = self._path / "patches" / slug / "coords"
            if cpath.exists():
                xy = np.asarray(zarr.open_array(str(cpath), mode="r"))
                coords["x"] = ("patch", xy[:, 0])
                coords["y"] = ("patch", xy[:, 1])
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.attrs.update(
            {k: v for k, v in arr.attrs.items() if k != "_ARRAY_DIMENSIONS"}
        )
        return ds

    def _read_patch_cell_ids(self, slug: str) -> np.ndarray | None:
        """Cell ids for a cell-centered patch set, or ``None`` for grid."""
        path = self._path / "patches" / slug / "cell_ids"
        if not path.exists():
            return None
        return np.asarray(zarr.open_array(str(path), mode="r"))

    def _resolve_markers(self, channels: Any) -> tuple[list[int], list[str]]:  # noqa: ANN401
        """Resolve a channel ``Selection`` (or ``None``) to indices + names.

        ``None`` selects every **kept** marker in order. A selection keeps
        kept markers matching any ``include`` glob (default ``*``) and not
        matching any ``exclude`` glob, case-insensitively. Channels the
        user excluded (``keep="no"``) are never selected.
        """
        markers = list(self.markers)
        kept = self.kept_indices
        kept_markers = [markers[i] for i in kept]
        if channels is None:
            self._assert_nuclear_selected(kept_markers)
            return kept, kept_markers
        import fnmatch

        include_orig = channels.include or ["*"]
        include = [p.lower() for p in include_orig]
        exclude_orig = channels.exclude
        exclude = [p.lower() for p in exclude_orig]
        for orig, pat in zip(include_orig, include, strict=True):
            if not any(fnmatch.fnmatch(m.lower(), pat) for m in kept_markers):
                msg = (
                    f"{self._path.name}: --subset include pattern {orig!r} "
                    f"matches no kept marker in the image."
                )
                raise ValueError(msg)
        for orig, pat in zip(exclude_orig, exclude, strict=True):
            if not any(fnmatch.fnmatch(m.lower(), pat) for m in kept_markers):
                msg = (
                    f"{self._path.name}: --subset exclude pattern {orig!r} "
                    f"matches no kept marker in the image."
                )
                raise ValueError(msg)
        idxs = [
            kept[j]
            for j, m in enumerate(kept_markers)
            if any(fnmatch.fnmatch(m.lower(), p) for p in include)
            and not any(fnmatch.fnmatch(m.lower(), p) for p in exclude)
        ]
        if not idxs:
            msg = f"{self._path.name}: --subset matched no markers"
            raise ValueError(msg)
        used = [markers[i] for i in idxs]
        self._assert_nuclear_selected(used)
        return idxs, used

    def _assert_nuclear_selected(self, used: list[str]) -> None:
        """Hard-error if the selected markers carry no nuclear stain.

        Every extraction (like tissue + cell) needs a nuclear reference,
        so a panel or keep set that drops it is rejected up front.
        """
        from coral.tissue.infer import infer_dapi_index

        if infer_dapi_index(used) is None:
            msg = (
                f"{self._path.name}: the selected markers include no "
                f"nuclear stain — every extraction needs it; include the "
                f"nuclear marker in the panel."
            )
            raise ValueError(msg)

    def _check_required_markers(
        self,
        extractor: Any,  # noqa: ANN401
        used: list[str],
    ) -> None:
        """Warn on an extractor's missing required markers (never fatal).

        The ``required_markers()`` hook is dormant today — every shipped
        encoder returns ``None`` — but is kept as a warn-only contract
        point for a future marker-required encoder.
        """
        required = extractor.required_markers()
        if not required:
            return
        have = {m.lower() for m in used}
        missing = [m for m in required if m.lower() not in have]
        if missing:
            logger.warning(
                "%s: extractor %r needs markers not in the selection: "
                "%s — proceeding.",
                self._path.name,
                extractor.name,
                missing,
            )

    def _extract_patch_features(
        self,
        extractor: Any,  # noqa: ANN401
        coords: np.ndarray,
        patch_size: int,
        idxs: list[int],
        used: list[str],
        batch_size: int,
        *,
        cell_ids: np.ndarray | None = None,
        unit: str = "patch",
        nuclear_marker: str | None = None,
        finish: Any = None,  # noqa: ANN401 — Callable[[ndarray], None] | None
        num_workers: int = 0,
    ) -> np.ndarray:
        """Lazily read patches and run the encoder in batches.

        Builds an :class:`~coral.features.dataset.CoralDataset` (lazy level-0
        read; grid raw box, or cell-isolated when ``cell_ids`` given; the
        data-driven dtype-scale per ``extractor.scale``), computes the
        marker embedding **once**, then per batch applies the encoder's
        ``transform`` (the slide's ``nuclear_marker`` is its DAPI hint) and
        ``forward``. A tqdm bar advances per batch so long encodes show
        elapsed time / ETA; CI / non-TTY runs stay quiet.

        Optional ``finish(features)`` runs while the bar is still active
        so the stage result can land as the bar postfix. When a CLI
        :func:`~coral.utils.progress.per_image_bar` is active, its total is
        reset to the encode-batch count (same pattern as CARTA / Cellpose);
        otherwise a standalone leave=True bar is created.
        """
        from contextlib import nullcontext
        from math import ceil

        from tqdm.std import tqdm

        from coral.features.dataset import CoralDataset
        from coral.utils.progress import get_active_bar

        marker_emb = extractor.embed_markers(used)
        extractor.prepare_slide(self.image.isel(c=idxs))
        ds = CoralDataset(
            self._path,
            coords,
            patch_size,
            idxs,
            scale=extractor.scale,
            cell_ids=cell_ids,
        )
        label = "cells" if unit == "cell" else "patches"
        n = len(ds)
        n_batches = ceil(n / batch_size) if n else 0
        stem = Path(self._path).stem
        out: list[np.ndarray] = []

        active = get_active_bar()
        own_bar = active is None
        if own_bar:
            # Library / non-CLI path: own leave=True bar.
            pbar = tqdm(
                total=max(n_batches, 1),
                desc=f"{stem} encoding",
                unit="batch",
                leave=True,
                disable=None,
            )
            bar_cm = activate_bar(pbar)
        else:
            # CLI path: reuse the per-image bar; reset to encode batches.
            pbar = active
            if not pbar.disable:
                pbar.unit = "batch"
                pbar.reset(total=max(n_batches, 1))
            bar_cm = nullcontext(pbar)

        with bar_cm:
            for batch, _coords in _patch_batches(
                ds, batch_size, num_workers=num_workers
            ):
                x = extractor.transform(
                    batch, used, nuclear_marker=nuclear_marker
                )
                out.append(np.asarray(extractor.forward(x, used, marker_emb)))
                pbar.update(1)
            if not out:
                features = np.zeros((0, len(idxs)), dtype=np.float64)
            else:
                features = np.concatenate(out, axis=0)
            if finish is not None:
                finish_fn = finish  # Callable[[np.ndarray], None]
                finish_fn(features)
            else:
                bar_status(f"encoded {n} {label} (batch {batch_size})")
        return features

    def _feature_array(self, folder: Path) -> np.ndarray:
        """Read back a stored feature output array (for resume).

        The variant path is the array itself (no nested output child).
        """
        return np.asarray(zarr.open_array(str(folder), mode="r"))

    @staticmethod
    def _primary_output(
        extractor: Any,  # noqa: ANN401 — an CoralEncoder
    ) -> tuple[str, tuple[str, ...]]:
        """The extractor's single ``(output_name, trailing_dims)``.

        CORAL's encoders each declare one logical output, all named
        ``features`` (mean_marker → dim ``marker``; a CLS encoder → dim
        ``feature``). The leading ``patch`` dim is implicit and added by
        the writer.
        """
        schema = extractor.output_schema or {"features": ()}
        key, dims = next(iter(schema.items()))
        return key, tuple(dims)

    def _write_feature_outputs(
        self,
        slug: str,
        encoder: str,
        variant: str,
        extractor: Any,  # noqa: ANN401
        features: np.ndarray,
        used: list[str],
        *,
        cell_ids: np.ndarray | None = None,
    ) -> None:
        """Write the single self-describing feature array at the variant.

        The extractor's one ``output_schema`` entry names the array's
        trailing dims (leading ``patch`` implicit) — e.g. mean_marker →
        ``_ARRAY_DIMENSIONS`` ``("patch", "marker")`` — and it is stored
        directly at ``features/<slug>/<encoder>/<variant>`` with no nested
        child. Its ``.zattrs`` carry the marker names (``markers_used``)
        and provenance; there is no separate ``marker`` array. Cell rows
        record ``patch_mode="cell"`` so :meth:`features` joins the cell
        ids from ``patches/<slug>/cell_ids`` (the one source of truth,
        like grid x/y from ``patches/<slug>/coords``). The array keeps the
        encoder's native dtype (no force-to-float64).
        """
        features = np.asarray(features)
        n = int(features.shape[0])
        out_key, trailing = self._primary_output(extractor)
        dims = ["patch", *trailing]
        parent = (
            self._store.require_group("features")
            .require_group(slug)
            .require_group(encoder)
        )
        arr = parent.create_dataset(
            variant,
            data=features,
            chunks=(max(1, min(n, 1 << 14)), *features.shape[1:]),
            overwrite=True,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = dims
        arr.attrs["markers_used"] = list(used)
        arr.attrs["extractor_name"] = extractor.name
        arr.attrs["extractor_version"] = extractor.version
        arr.attrs["patch_mode"] = "cell" if cell_ids is not None else "grid"
        arr.attrs["patch_slug"] = slug
        arr.attrs["outputs"] = {out_key: dims}
        arr.attrs["coral_feature_schema_version"] = _FEATURE_SCHEMA_VERSION
        arr.attrs["extracted_at"] = now_iso()
        prov = extractor.marker_provenance()
        if prov is not None:
            for key, value in prov.items():
                arr.attrs[key] = list(value)
        base = f"features/{slug}/{encoder}/{variant}"
        task = self._state.tasks.extract.setdefault(
            f"{slug}/{encoder}/{variant}", TaskState()
        )
        task.status = "completed"
        task.completed_at = now_iso()
        task.outputs = {out_key: base}
        self._save_state()

    def visualize_patches(
        self,
        config: Any,  # noqa: ANN401 — a PatchConfig
        *,
        save_to: str | Path | None = None,
        **style: Any,  # noqa: ANN401 — render_patch_overlay style knobs
    ) -> Path:
        """Re-render the patch-grid overlay for ``config``.

        Recomputes the scored grid (cheap), then redraws the overlay —
        handy for iterating the visual style (colours, line width, fill)
        without re-extracting. For ``mode="cell_centered"`` it
        re-renders the cell-patch overlay instead (boxes + centroid
        dots). Requires tissue detection to have run.

        Args:
            config: The ``PatchConfig`` whose grid to draw.
            save_to: Output PNG path. Defaults to the config's
                ``patch_overlay.png`` in the slide store.
            **style: Passed through to
                :func:`coral.patch.viz.render_patch_overlay` (e.g.
                ``patch_color``, ``line_width``, ``fill_alpha``).

        Returns:
            The path the overlay was written to.

        Raises:
            ValueError: If tissue detection has not run on the slide.

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
            ...     out = slide.visualize_patches(cfg, patch_color="cyan")
            ...     out.exists()
            True
        """
        from coral.patch.core import Patcher
        from coral.patch.viz import (
            render_cell_patch_overlay,
            render_patch_overlay,
        )

        base_mpp = self._mpp()
        resolved_mpp = self._resolve_patch_mpp(config.target_mpp, base_mpp)
        slug = config.resolved_slug(resolved_mpp)
        out_path = Path(
            self._path / "patches" / slug / "patch_overlay.png"
            if save_to is None
            else save_to
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        backdrop, left_title = self._patch_backdrop()
        style.setdefault("left_title", left_title)
        style.setdefault("tissue_mask", self._tissue_mask_np())

        if config.mode == "cell_centered":
            coords, cell_ids, _dropped = Patcher(
                self, config
            ).cell_centered_coords()
            style.setdefault(
                "cell_mask",
                np.asarray(self._store["cells/cell_mask"][:]),
            )
            render_cell_patch_overlay(
                backdrop,
                coords,
                config.patch_size,
                mpp=resolved_mpp,
                save_to=out_path,
                cell_ids=cell_ids,
                **style,
            )
            return Path(out_path)

        coords, tissue_prop = Patcher(self, config).generate()
        style.setdefault("tissue_prop", tissue_prop)
        style.setdefault(
            "effective_overlap",
            1.0 - config.effective_stride / config.patch_size,
        )
        render_patch_overlay(
            backdrop,
            coords,
            config.patch_size,
            mpp=resolved_mpp,
            save_to=out_path,
            **style,
        )
        return Path(out_path)
