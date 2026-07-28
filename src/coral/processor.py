"""``CoralProcessor`` — run one encoder over a cohort of slides.

Mirrors TRIDENT's ``Processor``: discover slides, build the encoder **once**,
and loop with a per-slide ``.lock`` + skip-done, reusing the single-slide
:meth:`CoralSlide.encode_features` unchanged. N processors on N GPUs split a
cohort collision-free via the locks (``coral.utils.locks``); a bad slide is
recorded and the cohort continues. Intra-slide resume (skipping completed
``<slug>/<encoder>`` tasks) is ``encode_features``'s own job.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coral.features import EXTRACTOR_REGISTRY
from coral.slide import CoralSlide
from coral.utils.errors import CoralError
from coral.utils.locks import (
    create_lock,
    is_locked,
    is_stale_lock,
    remove_lock,
)


@dataclass
class CohortResult:
    """Outcome of a cohort run — which slides processed / locked / failed.

    Example:
        >>> CohortResult().summary()
        '0 processed, 0 skipped (locked), 0 failed'
    """

    processed: list[Path] = field(default_factory=list)
    skipped_locked: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human summary of the cohort run.

        Returns:
            Counts of processed, lock-skipped, and failed slides.

        Example:
            >>> CohortResult().summary()
            '0 processed, 0 skipped (locked), 0 failed'
        """
        return (
            f"{len(self.processed)} processed, "
            f"{len(self.skipped_locked)} skipped (locked), "
            f"{len(self.failed)} failed"
        )


class CoralProcessor:
    """Run one encoder over a dataset of slides, resumably + parallel-safely.

    ``source`` is a cohort directory (searched **recursively** for ``.zarr``
    stores), an explicit list of slide paths, or a single ``.zarr``.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "a.zarr").mkdir()
        ...     _ = (Path(d) / "sub").mkdir()
        ...     _ = (Path(d) / "sub" / "b.zarr").mkdir()
        ...     proc = CoralProcessor(d)
        ...     [p.name for p in proc.slides]  # recursive, sorted
        ['a.zarr', 'b.zarr']
    """

    def __init__(self, source: str | Path | list[Path]) -> None:
        """Discover the cohort's slides; raise if none are found.

        Args:
            source: Cohort directory, a single ``.zarr``, or an explicit
                list of slide paths.

        Raises:
            CoralError: If no ``.zarr`` slides are found under ``source``.
        """
        self.slides = self._discover(source)
        if not self.slides:
            raise CoralError(f"no .zarr slides found in {source!r}")

    @staticmethod
    def _discover(source: str | Path | list[Path]) -> list[Path]:
        if isinstance(source, (list, tuple)):
            return [Path(s) for s in source]
        p = Path(source)
        if p.suffix == ".zarr":
            return [p]
        if p.is_dir():
            return sorted(p.rglob("*.zarr"))
        return []

    @staticmethod
    def _resolve_encoder(encoder: Any) -> Any:  # noqa: ANN401 — CoralEncoder|str
        """A built encoder is used as-is; a registry name is built once."""
        if isinstance(encoder, str):
            if encoder not in EXTRACTOR_REGISTRY:
                raise CoralError(f"unknown extractor {encoder!r}")
            return EXTRACTOR_REGISTRY[encoder].build()
        return encoder

    def run(
        self,
        work: Callable[[Path], None],
    ) -> CohortResult:
        """Loop the cohort with a per-slide lock + skip-locked + tolerance.

        The generic engine behind :meth:`encode_features` — and what the CLI
        uses directly for its per-patch-set loop. ``work(slide_path)`` does
        the per-slide job; any exception is recorded and the cohort
        continues. A locked slide is skipped unless its lock is **stale** (a
        crashed run's — reclaimed and reprocessed).

        Args:
            work: Callable invoked once per unlocked slide path.

        Returns:
            :class:`CohortResult` with processed / skipped / failed lists.

        Example:
            >>> proc.run(lambda s: do_work(s)).summary()  # doctest: +SKIP
            '12 processed, 0 skipped (locked), 0 failed'
        """
        result = CohortResult()
        total = len(self.slides)
        from coral.utils.progress import bar_status, bar_write, per_image_bar

        for i, slide_path in enumerate(self.slides, start=1):
            bar_write(f"[{i}/{total}] {slide_path.name}")
            with per_image_bar(desc=slide_path.stem, total=1, unit="batch"):
                if is_locked(slide_path):
                    if not is_stale_lock(slide_path):
                        bar_status("skipped — locked by another run")
                        result.skipped_locked.append(slide_path)
                        continue
                    bar_write("      reclaiming a stale lock")
                create_lock(slide_path)
                try:
                    work(slide_path)
                    result.processed.append(slide_path)
                except Exception as e:  # noqa: BLE001 — one bad ≠ cohort fail
                    bar_status(f"FAILED — {e}")
                    result.failed.append((slide_path, str(e)))
                finally:
                    remove_lock(slide_path)
        return result

    def encode_features(
        self,
        encoder: Any,  # noqa: ANN401 — an CoralEncoder or a registry name
        config: Any,  # noqa: ANN401 — a PatchConfig
        *,
        subset: Any = None,  # noqa: ANN401 — a subset YAML path, or None
        channels: Any = None,  # noqa: ANN401
        batch_size: int = 16,
        suffix: str | None = None,
    ) -> CohortResult:
        """Encode one patch-set ``config`` across the cohort (locked loop).

        Builds the encoder once (or takes a built one) and runs
        :meth:`CoralSlide.encode_features` per slide via :meth:`run`.

        Args:
            encoder: Built :class:`~coral.features.CoralEncoder` or
                registry name.
            config: :class:`~coral.config.PatchConfig` naming the patch set.
            subset: Optional subset YAML path; its filename names the
                variant (``markers_<stem>``). ``None`` = all kept markers
                (``markers_all``).
            channels: Pre-built marker ``Selection``; requires ``suffix``.
            batch_size: Patches encoded per forward pass.
            suffix: Explicit variant-folder name, overriding ``subset``.

        Returns:
            :class:`CohortResult` for this encode pass.

        Example:
            >>> proc.encode_features(enc, cfg).summary()  # doctest: +SKIP
            '12 processed, 0 skipped (locked), 0 failed'
        """
        enc = self._resolve_encoder(encoder)

        def _work(slide_path: Path) -> None:
            CoralSlide.open(slide_path).encode_features(
                enc,
                config,
                subset=subset,
                channels=channels,
                batch_size=batch_size,
                suffix=suffix,
            )

        return self.run(_work)
