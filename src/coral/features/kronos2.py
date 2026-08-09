"""``Kronos2Extractor`` — KRONOS2 patch CLS features (HF, marker-aware).

Loads ``MahmoodLab/KRONOS2`` via ``AutoModel(trust_remote_code=True)`` and
returns the 768-d CLS embedding per patch. The bit-exact marker-aware
normalisation lives **in the model** (``model.preprocess``); this extractor
only does the dtype ``/scaling_factor`` and the batched forward — so CORAL
stays model-agnostic and the normalisation has a single home.

KRONOS2 is an **opt-in** dependency (the ``kronos2`` extra: torch +
transformers + xformers + timm). Imports are lazy, so this module loads and
type-checks without the extra, and ``EXTRACTOR_REGISTRY["KRONOS2"]`` is
always present; loading a model without the extra raises a clear error.

Reproducibility: fp32, deterministic, and on GPU the **batch size can
matter**. Because KRONOS2 runs in fp32, any batch-size effect is bounded by
cuBLAS picking a batch-dependent kernel and is at most ~1e-4 (earlier runs
saw ~5e-5 at batch 8 vs 16, ~2e-4 at batch 4); on some GPU/cuBLAS versions
it is exactly 0 (a 1→64 sweep on real data was bit-identical). So treat
batch invariance as GPU-dependent, not guaranteed either way. The
gold-standard features were
produced at batch 16, and :meth:`forward` runs the whole input in one model
call, so the batch size is whatever the caller passes via
``encode_features(batch_size=...)``; reproduce the gold standard with
``batch_size=16``.
"""

from __future__ import annotations

import importlib.util
import logging
import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from coral.dtypes import scaling_factor
from coral.features import register
from coral.features.base import CoralEncoder
from coral.markers.additional import SlideStat, region_key
from coral.utils import resolve_device

logger = logging.getLogger(__name__)

# The heavy stack lives in the opt-in `kronos2` extra; probe without
# importing it (keeps this module importable + type-checkable without it).
KRONOS2_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)

_DEFAULT_REF = "hf_hub:MahmoodLab/KRONOS2"
_INSTALL_MSG = (
    "KRONOS2 needs the optional `kronos2` extra (torch + transformers + "
    "xformers + timm) — install it: `uv sync --extra kronos2` "
    "(or `pip install coral[kronos2]`)."
)


@register("KRONOS2")
class Kronos2Extractor(CoralEncoder):
    """KRONOS2 marker-aware ViT — per-patch CLS embedding ``(n_patches, 768)``.

    Build with :meth:`from_pretrained` (loads the model once); the encoder
    holds the model and is reused across slides. The marker z-score lives
    **in the model** (``model.preprocess``), so :meth:`transform` just calls
    it (``scale=True`` → the patcher already handed it float32 ``[0, 1]``
    patches), and :meth:`forward` runs the fixed-batch CLS pass. The slide's
    nuclear marker is the ``nuclear_marker`` hint on :meth:`transform` (wired
    by ``CoralSlide.encode_features``) for the model's DAPI special-case.

    Example:
        >>> Kronos2Extractor().required_markers() is None
        True
        >>> Kronos2Extractor.name
        'KRONOS2'
    """

    name: ClassVar[str] = "KRONOS2"
    embed_dim: ClassVar[int] = 768
    precision: ClassVar[str] = "float32"  # fp32, no autocast — bit-exact
    # KRONOS2 can z-score novel markers from prepare-pass stats.
    supports_novel_markers: ClassVar[bool] = True
    # CLS feature vector -> features/<slug>/KRONOS2/<variant>/features,
    # dims (patch, feature). The leading "patch" dim is implicit.
    output_schema: ClassVar[dict[str, tuple[str, ...]]] = {
        "features": ("feature",),
    }

    def __init__(self, model: Any = None) -> None:  # noqa: ANN401 — HF model
        """Hold a loaded model (or ``None`` until :meth:`from_pretrained`)."""
        self._model = model
        # Panels already warned about (markers, dapi) — warn once per panel.
        self._preflighted: set[tuple[tuple[str, ...], str | None]] = set()
        # Prepare-pass state: novel markers to stat, and the
        # most recent slide's partials + their tissue-mask reuse signature.
        self._stat_targets: list[str] = []
        self._last_slide_stats: dict[str, SlideStat] = {}
        self._last_region_key: str | None = None

    def configure_stats(self, targets: list[str]) -> None:
        """Set the novel markers the next :meth:`prepare_slide` stats.

        The prepare pass calls this once with the cohort's
        novel markers (those needing data-driven ``(mean, std)``); a later
        :meth:`prepare_slide` computes partials for exactly these names.

        Example:
            >>> ext = Kronos2Extractor()
            >>> ext.configure_stats(["FoxA1"])
            >>> ext.last_slide_stats
            {}
        """
        self._stat_targets = list(targets)

    @property
    def last_slide_stats(self) -> dict[str, SlideStat]:
        """Per-marker ``(n, μ, s²)`` from the last :meth:`prepare_slide`.

        Empty until a prepare-pass call (one with a ``tissue_mask`` and
        configured targets) runs; the orchestration reads it and persists
        the partials in the slide store.
        """
        return self._last_slide_stats

    @property
    def last_region_key(self) -> str | None:
        """Reuse signature of the most recent :meth:`prepare_slide` region.

        ``None`` until a prepare-pass call runs; see
        :func:`coral.markers.additional.region_key`.
        """
        return self._last_region_key

    def required_markers(self) -> list[str] | None:
        """Marker-agnostic — encodes whatever markers the patches carry."""
        return None

    @classmethod
    def build(cls, device: str | None = None) -> Kronos2Extractor:
        """Load the default KRONOS2 checkpoint (the CLI build-once hook).

        Args:
            device: Torch device; defaults to GPU when available.

        Example:
            >>> ext = Kronos2Extractor.build()  # doctest: +SKIP
        """
        return cls.from_pretrained(_DEFAULT_REF, device=device)

    @classmethod
    def from_pretrained(
        cls, ref: str = _DEFAULT_REF, *, device: str | None = None
    ) -> Kronos2Extractor:
        """Load KRONOS2 from the Hub (or a local dir) into the extractor.

        Args:
            ref: ``"hf_hub:<repo>"``, a bare repo id, or a local path. The
                ``hf_hub:`` prefix is stripped (TRIDENT convention).
            device: Torch device; defaults to CUDA when available.

        Raises:
            ImportError: If the ``kronos2`` extra is not installed.

        Example:
            >>> Kronos2Extractor.from_pretrained()  # doctest: +SKIP
        """
        if not KRONOS2_AVAILABLE:
            raise ImportError(_INSTALL_MSG)
        # KRONOS2's vendored DINOv2 + xFormers emit noisy import-time
        # diagnostics that read as failures though nothing is wrong: a
        # caught Triton-probe traceback (xFormers logs it because a bare
        # venv lacks the setuptools Triton needs) and "xFormers is
        # available" UserWarnings. Silence both before the model loads —
        # output only, no effect on the model or its embeddings. The
        # xFormers logger is raised to ERROR so any genuine error still
        # surfaces.
        logging.getLogger("xformers").setLevel(logging.ERROR)

        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModel,
        )

        repo = ref.removeprefix("hf_hub:")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is available")
            model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        dev = device or resolve_device()
        model = model.to(dev)
        return cls(model=model)

    def prepare_slide(
        self,
        image: Any,  # noqa: ANN401 — (c, y, x) array
        *,
        markers: list[str] | None = None,
        tissue_mask: Any = None,  # noqa: ANN401 — (y, x) bool array
    ) -> None:
        """Compute per-target ``(n, μ, s²)`` over the tissue-masked region.

        The prepare-pass call: for each :meth:`configure_stats`
        target present in ``markers``, take that channel's tissue-masked
        pixels, scale to ``[0, 1]`` by the dtype divisor (the same
        ``scaling_factor`` the patcher uses), and store the sample stats
        (``ddof=1``) in :attr:`last_slide_stats` keyed by marker, plus the
        :attr:`last_region_key` reuse signature.

        **No-op** when ``tissue_mask`` is ``None`` (the extract-time call),
        when no targets are configured, or when ``markers`` is ``None`` — so
        it never perturbs the patch loop. Pure numpy: the frozen model is
        untouched, so features stay bit-exact.

        Example:
            >>> import numpy as np
            >>> ext = Kronos2Extractor()
            >>> ext.configure_stats(["NOVEL"])
            >>> img = np.full((2, 2, 2), 6553, dtype=np.uint16)
            >>> ext.prepare_slide(
            ...     img,
            ...     markers=["DAPI", "NOVEL"],
            ...     tissue_mask=np.ones((2, 2), dtype=bool),
            ... )
            >>> round(ext.last_slide_stats["NOVEL"].mean, 2)
            0.1
        """
        if tissue_mask is None or not self._stat_targets or markers is None:
            return None
        mask = np.asarray(tissue_mask, dtype=bool)
        scaling = scaling_factor(np.dtype(image.dtype))
        name_to_idx = {m: i for i, m in enumerate(markers)}
        stats: dict[str, SlideStat] = {}
        for target in self._stat_targets:
            idx = name_to_idx.get(target)
            if idx is None:  # target not on this slide's selection — skip
                continue
            pixels = np.asarray(image[idx])[mask].astype(np.float64) / scaling
            n = int(pixels.size)
            mean = float(pixels.mean()) if n else 0.0
            var = float(pixels.var(ddof=1)) if n > 1 else 0.0
            stats[target] = SlideStat(n, mean, var)
        self._last_slide_stats = stats
        self._last_region_key = region_key(mask, scaling)
        return None

    def transform(
        self,
        patches: Any,  # noqa: ANN401 — float32 [0, 1] patch batch
        markers: list[str],
        *,
        nuclear_marker: str | None = None,
    ) -> np.ndarray:
        """Apply the model's per-marker z-score (the bit-exact normalisation).

        ``scale=True`` means the patcher already divided by the dtype
        ``scaling_factor`` → float32 ``[0, 1]``; this delegates the
        per-marker z-score to the model (``model.preprocess``, DAPI =
        ``nuclear_marker``) so the normalisation has a single home.

        Example:
            >>> ext = Kronos2Extractor.from_pretrained()  # doctest: +SKIP
            >>> ext.transform(p, mk, nuclear_marker="DRAQ5")  # doctest: +SKIP
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        self._warn_default_stats(markers, nuclear_marker)
        scaled = np.asarray(patches, dtype=np.float32)
        return self._model.preprocess(
            scaled, markers, preferred_dapi=nuclear_marker
        )

    def novel_markers(
        self, markers: list[str], nuclear_marker: str | None = None
    ) -> list[str]:
        """Markers absent from the model's stats vocabulary (would default).

        Reproduces the model's **own** resolution — ``clean_marker_name`` +
        ``marker_match_key`` against ``model._marker_index`` (built from
        ``model._marker_stats``), with the DAPI special-case — so CORAL and
        the model agree on exactly which markers are novel. The slide's
        nuclear channel (``nuclear_marker``) maps to ``dapi`` and is never
        novel. Returns the offending names in input order; empty once the
        novel markers have been registered via ``additional_markers.csv``.

        Example:
            >>> Kronos2Extractor().novel_markers(["CD3"])  # doctest: +SKIP
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import sys

        stats = getattr(self._model, "_marker_stats", None) or {}
        if not stats:
            return []
        mod: Any = sys.modules[type(self._model).__module__]
        clean = mod.clean_marker_name
        match_key = mod.marker_match_key
        index = getattr(self._model, "_marker_index", None) or (
            mod.build_marker_index(stats)
        )
        dapi = clean(nuclear_marker) if nuclear_marker else None
        out: list[str] = []
        for m in markers:
            if dapi is not None and clean(m) == dapi:
                continue
            if match_key(m) not in index:
                out.append(m)
        return out

    def register_additional_markers(self, csv_path: str | Path) -> None:
        """Register novel markers from a completed ``additional_markers.csv``.

        The CORAL-side handoff to the KRONOS2 half: delegates to the model's
        own ``register_additional_markers``, which loads the Journey-1
        ``(mean, std)`` stats and builds the Journey-2 BioLinkBERT text
        embeddings from the CSV rows (the text columns the user supplied +
        the ``mean``/``std`` CORAL filled). Afterwards the registered markers
        are no longer reported by :meth:`novel_markers`.

        Example:
            >>> ext = Kronos2Extractor.from_pretrained()  # doctest: +SKIP
            >>> ext.register_additional_markers("add.csv")  # doctest: +SKIP
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        self._model.register_additional_markers(str(csv_path))

    def _warn_default_stats(
        self, markers: list[str], nuclear_marker: str | None
    ) -> None:
        """Warn once per panel for markers that z-score with default stats.

        A marker that :meth:`novel_markers` flags is absent from the model's
        stats table and silently falls back to the default ``(mean, std)``
        inside ``model.preprocess`` — so it may be mis-normalised unless it
        was registered via ``additional_markers.csv``. Pure
        read + log; the frozen model is untouched, so features stay
        bit-exact. Warns once per ``(markers, dapi)`` signature to avoid
        per-batch spam.
        """
        sig = (tuple(markers), nuclear_marker)
        if sig in self._preflighted:
            return
        self._preflighted.add(sig)
        defaulted = self.novel_markers(markers, nuclear_marker)
        if defaulted:
            names = ", ".join(repr(m) for m in defaulted)
            logger.warning(
                "KRONOS2 z-score: %d of %d markers not found in the "
                "marker-stats table; these channels fall back to the "
                "DEFAULT (mean, std) and may be mis-normalised: %s",
                len(defaulted),
                len(markers),
                names,
            )

    def forward(
        self,
        x: Any,  # noqa: ANN401 — normalised patch batch
        markers: list[str],
        marker_emb: Any = None,  # noqa: ANN401 — unused (model embeds names)
    ) -> np.ndarray:
        """CLS forward over the whole input ``x`` → ``(n, 768)``.

        The batch size matters for bit-exactness — the gold standard ran at
        batch 16, and a different size shifts the attention kernel's
        reductions (see the module note). The whole input is run in one
        model call, so the caller sets the batch size via
        ``encode_features(batch_size=...)``; pass ``batch_size=16`` to
        reproduce the gold standard. ``markers`` (names) drive the model;
        ``marker_emb`` is unused (KRONOS2 embeds marker names internally).
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]

        normed = np.ascontiguousarray(np.asarray(x))
        dev = next(self._model.parameters()).device
        with torch.inference_mode():
            t = torch.from_numpy(normed).to(dev)  # pyright: ignore
            cls = self._model(t, markers).float().cpu().numpy()
        return cls.astype(np.float32)
