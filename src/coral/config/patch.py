"""Patch extraction configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PatchMode = Literal["grid", "cell_centered"]

# Keys the patch writer adds to config.json beyond the model fields —
# derived provenance (see ``CoralSlide._write_patch_outputs``). Stripped when
# reading a stored config so it round-trips; any other extra key still
# errors (e.g. a retired knob).
_STORED_PROVENANCE_KEYS = frozenset({"resolved_mpp", "slug", "tissue_method"})


class PatchConfig(BaseModel):
    """Configuration for a patch-extraction run on a single slide.

    CORAL is mpp-centric: there is no magnification anywhere. Patches are
    extracted at the slide's base resolution (level 0) — which is how
    KRONOS encodes, whatever the slide's mpp — so ``target_mpp`` is
    ``None`` by default and resolves to that base mpp at extract time.

    Attributes:
        patch_size: Patch side in pixels at the target mpp.
        stride: Stride in pixels. If ``None``, derived from ``overlap``
            as ``int(patch_size * (1 - overlap))``.
        overlap: Fractional overlap (0.0–1.0). Ignored if ``stride``
            is given.
        target_mpp: Target microns-per-pixel. ``None`` (default) means
            the slide's base mpp (level 0). A value coarser than the
            slide's base mpp is a future feature (custom-mpp
            downsampling is deferred); finer is impossible.
        mode: Patching mode. ``"grid"`` (the default) tiles the whole
            image and scores every patch by its tissue coverage;
            ``"cell_centered"`` emits one patch per segmented cell and
            requires cells to have been segmented first.

    Note:
        Grid patching keeps **every** patch and stores its tissue
        fraction alongside the coordinates — no patch is discarded. Pick
        a tissue cut-off downstream, where the analysis can see it (and
        change it) without re-patching.

    Example:
        >>> from coral.config import PatchConfig
        >>> cfg = PatchConfig(patch_size=256, target_mpp=0.5)
        >>> cfg.resolved_slug(0.5)
        '0.5mpp_256px'
    """

    # Unknown keys are an error, not a silent no-op: a config carrying a
    # retired knob (e.g. the old ``min_tissue_proportion``) must fail
    # loudly rather than quietly patch with different behaviour.
    model_config = ConfigDict(extra="forbid")

    patch_size: int = Field(gt=0)
    stride: int | None = Field(default=None, gt=0)
    overlap: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_mpp: float | None = Field(default=None, gt=0)
    mode: PatchMode = "grid"

    @model_validator(mode="after")
    def _check_stride_overlap_exclusive(self) -> PatchConfig:
        if self.stride is not None and self.overlap > 0.0:
            msg = "Provide either `stride` or non-zero `overlap`, not both."
            raise ValueError(msg)
        return self

    @classmethod
    def from_stored(cls, doc: dict[str, object]) -> PatchConfig:
        """Reconstruct a config from a stored ``patches/<slug>/config.json``.

        That file is a **superset** of the model — the patch inputs plus
        derived provenance (``resolved_mpp``, ``slug``, ``tissue_method``,
        written by :meth:`CoralSlide._write_patch_outputs`). The provenance
        is stripped, then the remaining keys are validated as usual, so a
        legitimate patch set round-trips while a **retired** knob (e.g. the
        old ``min_tissue_proportion``) still errors — a stale store is
        caught, not silently mis-read.

        Args:
            doc: The parsed ``config.json`` dict.

        Returns:
            The ``PatchConfig`` that produced the stored patch set.

        Raises:
            ValidationError: If a non-provenance extra key is present
                (e.g. a retired knob) or a model field is invalid.

        Example:
            >>> PatchConfig.from_stored(
            ...     {
            ...         "patch_size": 256,
            ...         "mode": "grid",
            ...         "resolved_mpp": 0.37,
            ...         "slug": "0.37mpp_256px",
            ...         "tissue_method": "otsu",
            ...     }
            ... ).patch_size
            256
        """
        return cls.model_validate(
            {k: v for k, v in doc.items() if k not in _STORED_PROVENANCE_KEYS}
        )

    @property
    def effective_stride(self) -> int:
        """Stride in pixels — explicit if given, else from overlap.

        Always returns at least 1 (a stride of 0 would mean every
        patch is at the same coordinate; we clamp instead of relying
        on integer floor of extreme overlaps).

        Returns:
            Effective stride in pixels (≥ 1).

        Example:
            >>> from coral.config import PatchConfig
            >>> PatchConfig(patch_size=256, overlap=0.5).effective_stride
            128
            >>> PatchConfig(patch_size=256, stride=64).effective_stride
            64
        """
        if self.stride is not None:
            return self.stride
        return max(1, int(self.patch_size * (1.0 - self.overlap)))

    def resolved_slug(self, mpp: float) -> str:
        """Authoritative store slug for this config at a resolved mpp.

        Used to name the patch sub-group inside the slide store once
        the physical mpp is known (``extract_patches`` resolves
        ``target_mpp is None`` against the slide's base mpp). The mpp
        is rounded to 4 dp then formatted with ``:g`` so the folder
        name reads as the true physical resolution, e.g.
        ``0.37mpp_256px``.

        Args:
            mpp: The resolved microns-per-pixel of the patches.

        Returns:
            Filesystem-safe slug, e.g. ``0.5mpp_256px`` or
            ``0.5mpp_256px_0.5overlap`` or ``cell_0.5mpp_64px``.

        Example:
            >>> PatchConfig(patch_size=256).resolved_slug(0.37)
            '0.37mpp_256px'
            >>> PatchConfig(patch_size=256, overlap=0.5).resolved_slug(0.5)
            '0.5mpp_256px_0.5overlap'
        """
        return self._slug(f"{round(mpp, 4):g}")

    @property
    def name(self) -> str:
        """Config-only preview slug (no slide bound).

        Mirrors :meth:`resolved_slug` but, since the physical mpp isn't
        known without a slide, ``target_mpp is None`` shows the literal
        token ``base``. The on-disk folder uses
        :meth:`resolved_slug` (the resolved base mpp) instead — so this
        is for human-facing previews / logging, not for store paths.

        Returns:
            Filesystem-safe preview slug.

        Example:
            >>> PatchConfig(patch_size=256).name
            'basempp_256px'
            >>> PatchConfig(patch_size=256, target_mpp=0.5).name
            '0.5mpp_256px'
            >>> PatchConfig(patch_size=256, overlap=0.5, target_mpp=0.5).name
            '0.5mpp_256px_0.5overlap'
            >>> PatchConfig(patch_size=256, stride=128, target_mpp=0.5).name
            '0.5mpp_256px_0.5overlap'
            >>> PatchConfig(
            ...     patch_size=64, target_mpp=0.5, mode="cell_centered"
            ... ).name
            'cell_0.5mpp_64px'
        """
        token = "base" if self.target_mpp is None else f"{self.target_mpp:g}"
        return self._slug(token)

    def _slug(self, mpp_token: str) -> str:
        """Build a slug from a formatted mpp token + this config.

        ``{mpp_token}mpp_{patch_size}px`` with a ``cell_`` prefix for
        cell-centered mode (which ignores overlap) and a
        ``_{frac:g}overlap`` suffix for grid modes with non-zero
        effective overlap. The overlap fraction is computed from
        ``effective_stride / patch_size`` so it's stable whether the
        user passed ``overlap`` or ``stride``.
        """
        body = f"{mpp_token}mpp_{self.patch_size}px"
        if self.mode == "cell_centered":
            return f"cell_{body}"
        effective_overlap = 1.0 - (self.effective_stride / self.patch_size)
        if effective_overlap > 0.0:
            return f"{body}_{effective_overlap:g}overlap"
        return body
