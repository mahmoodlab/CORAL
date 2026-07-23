"""``EvaExtractor`` — Eva patch CLS features (marker-aware, GenePT).

Loads Meta-style **Eva** (a multimodal self-supervised FM for spatial
proteomics; two-stage ViT) from the public HF repo ``yandrewl/Eva`` and
returns the 768-d CLS embedding per patch. Unlike the RGB encoders Eva
is **marker-aware**: it ingests the multiplex channels directly
(channels-last ``[0, 1]``, resized to 224) **plus a list of marker
names** (``bms``) and maps each to a GenePT embedding internally. So Eva
is the **first** encoder whose :meth:`embed_markers` slot does real work
— it maps the panel's KRONOS names to Eva ``bms`` names
(:func:`coral.markers.get_mappable_markers`), dropping markers absent from
Eva's vocabulary + nuclear stains. CLS only (the avg-token path is never
shipped, like every encoder).

Loading is public (no HF token): :meth:`from_pretrained` pulls the
checkpoint from ``yandrewl/Eva`` via the **vendored** Eva model
(:mod:`coral.features._eva`). The model loads its GenePT marker embeddings
from the ``assets/Eva/``-resolved pickle path (~909 MB, Zenodo), injected
into the model conf — no CWD symlink. The model config is built inline
(matching the kronos pipeline that produced the gold), so there is no
``config.yaml`` dependency.

Eva is an **opt-in** dependency (the ``eva`` extra: torch + timm + …).
The Eva model itself is vendored into CORAL (no external package), but the
heavy ML stack lives in the extra, so this module imports the vendored
model **lazily** — it loads + type-checks without the extra, and
``EXTRACTOR_REGISTRY["Eva"]`` is always present; loading without the extra
raises a clear ``ImportError``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from coral.assets import require_asset
from coral.features import register
from coral.features.base import CoralEncoder
from coral.markers import get_mappable_markers
from coral.utils import resolve_device

logger = logging.getLogger(__name__)

_DEFAULT_REPO = "yandrewl/Eva"
_INSTALL_MSG = (
    "Eva needs the optional `eva` extra (torch + timm + …) — install it: "
    "`uv sync --extra eva` (or `pip install coral[eva]`)."
)

# Inline model config, matching the kronos pipeline that produced the
# gold (mask_ratio 0.0 → no masking at inference). Avoids a config.yaml
# dependency (training used mask_ratio 0.75).
_EVA_CONF: dict[str, dict[str, Any]] = {
    "ds": {
        "patch_size": 224,
        "token_size": 8,
        "marker_dim": 3072,
        "mask_strategy": "random",
        "mask_ratio": 0.0,
    },
    "cm": {
        "dim": 512,
        "n_layers": 2,
        "n_heads": 4,
        "mlp_ratio": 4.0,
        "patch_embed_channel": "agnostic",
    },
    "pm": {
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "mlp_ratio": 4.0,
        "out_dim": 512,
        "channel_recon": "repeat",
    },
    "de": {
        "dim": 512,
        "n_layers": 8,
        "n_heads": 16,
        "mlp_ratio": 4.0,
        "flatten": "patch",
        "marker_dim": 512,
    },
}


@register("Eva")
class EvaExtractor(CoralEncoder):
    """Eva marker-aware ViT — per-patch 768-d CLS embedding.

    Build with :meth:`from_pretrained` (loads the model once, reused
    across slides). ``scale=True`` → the patcher hands float32 ``[0, 1]``
    patches; :meth:`embed_markers` maps the panel to Eva ``bms`` names
    (once), :meth:`transform` filters to the mappable channels + resizes
    to 224 channels-last, and :meth:`forward` runs Eva's
    ``extract_features`` (CLS).

    Example:
        >>> EvaExtractor().required_markers() is None
        True
        >>> (EvaExtractor.name, EvaExtractor.embed_dim)
        ('Eva', 768)
    """

    name: ClassVar[str] = "Eva"
    embed_dim: ClassVar[int] = 768
    precision: ClassVar[str] = "float32"  # fp32 for a clean tie-out
    output_schema: ClassVar[dict[str, tuple[str, ...]]] = {
        "features": ("feature",),
    }

    def __init__(self, model: Any = None) -> None:  # noqa: ANN401 — Eva model
        """Hold a loaded model (or ``None`` until :meth:`from_pretrained`)."""
        self._model = model
        self._kept_idx: list[int] = []
        self._provenance: dict[str, list[str]] | None = None

    def required_markers(self) -> list[str] | None:
        """No hard requirement — Eva maps any panel, dropping the rest."""
        return None

    def warn_for_patches(self, patch_size: int, mode: str) -> None:
        """Warn when a small cell patch will be upsampled to 224 (footgun).

        Eva ingests a fixed 224px input, so a cell-centered patch smaller
        than 128px is bilinearly upsampled (e.g. 64 → 224, 4×), which can
        alter the embeddings considerably. Emits one ``logger.warning``
        recommending cell-centered patching at 224px; never fatal. Grid
        patches and patches ≥ 128px stay silent.

        Args:
            patch_size: Patch side length in pixels.
            mode: The patch set's mode (``"grid"`` or ``"cell_centered"``).

        Example:
            >>> EvaExtractor().warn_for_patches(224, "cell_centered")
        """
        if mode == "cell_centered" and patch_size < 128:
            logger.warning(
                "Eva: %dpx cell-centered patches will be bilinearly "
                "resized to 224px, which may alter results considerably. "
                "For faithful results, use cell-centered patching at 224px.",
                patch_size,
            )

    @classmethod
    def build(cls, device: str | None = None) -> EvaExtractor:
        """The CLI build-once hook — resolves the pickle from ``assets/Eva``.

        Reads ``assets/Eva/GenePT_embedding.pkl`` and loads via
        :meth:`from_pretrained`, so ``coral extract --extractor Eva`` works
        once the pickle is in place.

        Args:
            device: Torch device; defaults to GPU when available.

        Raises:
            FileNotFoundError: If the GenePT pickle is not under
                ``assets/Eva/``.

        Example:
            >>> EvaExtractor.build()  # doctest: +SKIP
        """
        pkl = require_asset(
            cls.name,
            "GenePT_embedding.pkl",
            what=(
                "a GenePT embeddings pickle "
                "(GenePT_gene_protein_embedding_model_3_text.pickle, "
                "~909 MB)"
            ),
            source="Zenodo at https://zenodo.org/records/10833191",
        )
        return cls.from_pretrained(str(pkl), device=device)

    @classmethod
    def from_pretrained(
        cls,
        genept_pkl: str,
        *,
        repo: str = _DEFAULT_REPO,
        device: str | None = None,
    ) -> EvaExtractor:
        """Load Eva from the Hub via ``load_from_hf`` into the extractor.

        Args:
            genept_pkl: Path to the GenePT embeddings pickle.
            repo: The HF repo id (default ``yandrewl/Eva``, public).
            device: Torch device; defaults to CUDA when available.

        Raises:
            ImportError: If the ``eva`` extra is not installed.
            FileNotFoundError: If ``genept_pkl`` does not exist.

        Example:
            >>> EvaExtractor.from_pretrained("genept.pkl")  # doctest: +SKIP
        """
        try:
            from omegaconf import OmegaConf  # pyright: ignore

            from coral.features._eva import load_from_hf  # pyright: ignore
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e

        if not Path(genept_pkl).is_file():
            raise FileNotFoundError(f"GenePT pickle not found: {genept_pkl}")
        conf = OmegaConf.create(_EVA_CONF)
        # The vendored model reads the GenePT pickle from conf.ds.genept_pkl
        # (CORAL edit; see coral.features._eva) — no CWD symlink.
        conf.ds.genept_pkl = genept_pkl
        dev = device or resolve_device()
        model = load_from_hf(repo_id=repo, conf=conf, device=dev)
        return cls(model=model)

    def embed_markers(self, markers: list[str]) -> list[str]:
        """Map the panel's KRONOS names to Eva ``bms`` names (once/panel).

        The first non-None :meth:`embed_markers` override: resolves the
        panel via :func:`coral.markers.get_mappable_markers`, caches the
        kept channel indices for :meth:`transform`, and returns the Eva
        ``bms`` names for :meth:`forward`. Markers absent from Eva's
        vocabulary (incl. the 9 viral) + nuclear stains are dropped.

        Args:
            markers: The panel's marker names in channel order.

        Returns:
            The Eva ``bms`` names (aligned to the kept channels).

        Raises:
            ValueError: If no marker in the panel maps to Eva's vocab.

        Example:
            >>> EvaExtractor().embed_markers(["Hoechst1", "CD3", "CD20"])
            ['CD3e', 'CD20']
        """
        kept_idx, bms, dropped = get_mappable_markers(markers)
        if not kept_idx:
            msg = "Eva: no panel marker maps to Eva's vocabulary."
            raise ValueError(msg)
        self._kept_idx = kept_idx
        self._record_provenance(markers, kept_idx, bms, dropped)
        return bms

    def _record_provenance(
        self,
        markers: list[str],
        kept_idx: list[int],
        bms: list[str],
        dropped: list[str],
    ) -> None:
        """Cache the effective/dropped/no-GenePT marker sets + warn.

        Names are the original panel names (not Eva ``bms`` names).
        ``markers_no_genept`` are kept markers whose Eva name lacks a GenePT
        gene — the model gives them a random embedding, so their features
        are not meaningful; that (otherwise silent) case is logged. The
        authoritative no-GenePT source is the loaded model's module dict of
        vocab markers lacking a pickle gene (empty without a model, so a
        model-free panel mapping reports none).
        """
        if self._model is None:
            unknown: Any = set()
        else:
            unknown = self._model.model.marker_embed.unknown_marker_embeddings
        no_genept = [
            markers[i]
            for i, name in zip(kept_idx, bms, strict=True)
            if name in unknown
        ]
        self._provenance = {
            "markers_effective": [markers[i] for i in kept_idx],
            "markers_dropped": dropped,
            "markers_no_genept": no_genept,
        }
        if no_genept:
            logger.warning(
                "Eva: %d kept marker(s) have no GenePT embedding and get a "
                "random vector — their features are not meaningful: %s.",
                len(no_genept),
                no_genept,
            )

    def marker_provenance(self) -> dict[str, list[str]] | None:
        """The Eva marker sets recorded by :meth:`embed_markers`.

        Returns the ``markers_effective`` / ``markers_dropped`` /
        ``markers_no_genept`` dict (original panel names), or ``None`` before
        :meth:`embed_markers` has run.

        Example:
            >>> EvaExtractor().marker_provenance() is None
            True
        """
        return self._provenance

    def transform(
        self,
        patches: Any,  # noqa: ANN401 — float32 [0, 1] patch batch
        markers: list[str],
        *,
        nuclear_marker: str | None = None,
    ) -> Any:  # noqa: ANN401 — torch tensor
        """Filter to mappable channels → resize 224 → channels-last.

        Uses the kept indices cached by :meth:`embed_markers` (called
        once before the patch loop) to drop the unmappable channels, then
        bilinear-resizes to 224 and permutes to Eva's channels-last
        ``(B, 224, 224, C)``. No RGB, no ImageNet, no marker z-score.
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]
        import torch.nn.functional as f  # pyright: ignore

        dev = next(self._model.parameters()).device
        arr = np.ascontiguousarray(patches, dtype=np.float32)
        t = torch.from_numpy(arr).to(dev)[:, self._kept_idx]  # pyright: ignore
        t = f.interpolate(
            t, size=(224, 224), mode="bilinear", align_corners=False
        )
        return t.permute(0, 2, 3, 1)  # channels-last (B, 224, 224, C)

    def forward(
        self,
        x: Any,  # noqa: ANN401 — channels-last torch tensor
        markers: list[str],
        marker_emb: Any,  # noqa: ANN401 — the Eva bms names
    ) -> np.ndarray:
        """Batched Eva forward → ``(N, 768)`` CLS features.

        Runs the whole input ``x`` in one model call; fp32, per-image.
        The caller controls how many patches arrive per call via
        ``encode_features(batch_size=...)`` — Eva is heavy (3072-d marker
        embeddings × all channels), so a small ``batch_size`` is usually
        needed to fit GPU memory. Each patch is encoded independently, but
        cuBLAS picks batch-dependent kernels on GPU, so the fp32 result
        still shifts slightly (~5e-5) with ``batch_size`` (CPU is
        bit-identical). ``marker_emb`` is the Eva ``bms`` names from
        :meth:`embed_markers`, replicated per batch item; ``cls=True``
        takes the CLS token (``channel_mode="full"``). CLS only.
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]

        from coral.features._eva import extract_features  # pyright: ignore

        bms = marker_emb
        dev = next(self._model.parameters()).device
        with torch.inference_mode():
            feat = extract_features(
                patch=x,
                bms=[bms] * len(x),
                model=self._model,
                device=dev,
                cls=True,
                channel_mode="full",
            )
        return feat.float().cpu().numpy().astype(np.float32)
