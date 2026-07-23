"""``Kronos1Extractor`` — KRONOS1 patch CLS features (marker-aware).

Loads the **original published KRONOS** (``MahmoodLab/KRONOS``) — a custom
DINOv2-style **ViT-S/16, 384-d**, the foundation model CORAL is built
around — and returns the per-patch CLS embedding. KRONOS1 is the **first**
encoder to use the scaffold's per-marker z-score: unlike KRONOS2 (whose
z-score lives *in* the model), KRONOS1 is a raw ViT, so **CORAL applies the
z-score itself** (in :meth:`transform`) and passes integer ``marker_ids``
(in :meth:`forward`). KRONOS1 ≠ KRONOS2 normalisation.

Both the marker stats and the ids come from the vendored 177-marker
vocabulary (:mod:`coral.markers._kronos_marker_meta`), resolved by
:func:`coral.markers.resolve_markers` (kronos-exact matching; out-of-vocab →
a deterministic odd id + default stats + a warning). Like KRONOS2 the
:meth:`embed_markers` slot stays ``None`` — the marker work happens in
:meth:`transform` (which has the DAPI hint) and the resolution is cached
so :meth:`forward` reuses the same ids. CLS only, ever.

KRONOS1 is an **opt-in** dependency: the ``kronos1`` extra (torch +
xformers). The model itself is **vendored** into
:mod:`coral.features._kronos1` (no external ``kronos`` package),
imported lazily, so this module loads + type-checks without the extra, and
``EXTRACTOR_REGISTRY["KRONOS1"]`` is always present; loading without the
extra raises a clear ``ImportError``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from coral.features import register
from coral.features.base import CoralEncoder
from coral.markers.kronos1_markers import MarkerRecord, resolve_markers
from coral.utils import resolve_device

# The vendored loader wants the ``hf_hub:`` prefix (it branches on it), so —
# unlike KRONOS2 — we keep it.
_DEFAULT_REF = "hf_hub:MahmoodLab/KRONOS"
_INSTALL_MSG = (
    "KRONOS1 needs the optional `kronos1` extra (torch + xformers) — "
    "install it: `uv sync --extra kronos1` (or `pip install coral[kronos1]`). "
    "The weights repo `MahmoodLab/KRONOS` is gated — request access first."
)


@register("KRONOS1")
class Kronos1Extractor(CoralEncoder):
    """KRONOS1 marker-aware ViT — per-patch CLS embedding ``(n, 384)``.

    Build with :meth:`from_pretrained` (loads the model once, reused
    across slides). ``scale=True`` → the patcher hands float32 ``[0, 1]``
    patches; :meth:`transform` applies the per-marker z-score
    ``(x − mean) / std`` and :meth:`forward` runs the model with the
    resolved integer ``marker_ids`` and keeps the CLS token.

    Example:
        >>> Kronos1Extractor().required_markers() is None
        True
        >>> (Kronos1Extractor.name, Kronos1Extractor.embed_dim)
        ('KRONOS1', 384)
    """

    name: ClassVar[str] = "KRONOS1"
    embed_dim: ClassVar[int] = 384
    precision: ClassVar[str] = "float32"  # fp32; deterministic
    output_schema: ClassVar[dict[str, tuple[str, ...]]] = {
        "features": ("feature",),
    }

    def __init__(self, model: Any = None) -> None:  # noqa: ANN401 — model
        """Hold a loaded model (or ``None`` until :meth:`from_pretrained`)."""
        self._model = model
        # Per-panel resolution cache (markers -> records); the marker
        # warning fires once per panel via this memoisation.
        self._cache: dict[tuple[str, ...], list[MarkerRecord]] = {}

    def required_markers(self) -> list[str] | None:
        """Marker-agnostic — encodes whatever markers the patches carry."""
        return None

    @classmethod
    def build(cls, device: str | None = None) -> Kronos1Extractor:
        """Load the default KRONOS1 checkpoint (the CLI build-once hook).

        Args:
            device: Torch device; defaults to GPU when available.

        Example:
            >>> ext = Kronos1Extractor.build()  # doctest: +SKIP
        """
        return cls.from_pretrained(_DEFAULT_REF, device=device)

    @classmethod
    def from_pretrained(
        cls, ref: str = _DEFAULT_REF, *, device: str | None = None
    ) -> Kronos1Extractor:
        """Load KRONOS1 from the Hub (or a local path) into the extractor.

        Args:
            ref: ``"hf_hub:<repo>"`` (the prefix is kept — the kronos
                loader branches on it) or a local checkpoint path.
            device: Torch device; defaults to GPU when available.

        Raises:
            ImportError: If the ``kronos1`` extra is not installed.

        Example:
            >>> Kronos1Extractor.from_pretrained()  # doctest: +SKIP
        """
        try:
            from coral.features._kronos1 import create_model_from_pretrained
        except ImportError as exc:
            raise ImportError(_INSTALL_MSG) from exc

        # The default config is already vits16; passing it is explicit.
        model, _precision, _dim = create_model_from_pretrained(
            checkpoint_path=ref,
            cfg={"model_type": "vits16", "token_overlap": False},
        )
        model.eval()
        dev = device or resolve_device()
        model = model.to(dev)
        return cls(model=model)

    def _resolve(
        self, markers: list[str], nuclear_marker: str | None = None
    ) -> list[MarkerRecord]:
        """Resolve + cache a panel's marker records (warns once/panel)."""
        key = tuple(markers)
        recs = self._cache.get(key)
        if recs is None:
            recs = resolve_markers(markers, nuclear_marker=nuclear_marker)
            self._cache[key] = recs
        return recs

    def transform(
        self,
        patches: Any,  # noqa: ANN401 — float32 [0, 1] patch batch
        markers: list[str],
        *,
        nuclear_marker: str | None = None,
    ) -> np.ndarray:
        """Apply the per-marker z-score ``(x − mean) / std``.

        The patcher already divided by the dtype ``scaling_factor``
        (``scale=True`` → float32 ``[0, 1]``); this z-scores each channel
        with its KRONOS1 vocabulary stats (out-of-vocab → default stats).
        Pure numpy — the resolution is cached so :meth:`forward` reuses the
        same ids. ``nuclear_marker`` is the DAPI hint.

        Example:
            >>> import numpy as np
            >>> ext = Kronos1Extractor()
            >>> ext.transform(np.ones((2, 1, 4, 4)), ["DAPI"]).shape
            (2, 1, 4, 4)
        """
        recs = self._resolve(markers, nuclear_marker)
        means = np.array([r.mean for r in recs], dtype=np.float32)
        stds = np.array([r.std for r in recs], dtype=np.float32)
        x = np.asarray(patches, dtype=np.float32)
        return (x - means[None, :, None, None]) / stds[None, :, None, None]

    def forward(
        self,
        x: Any,  # noqa: ANN401 — z-scored patch batch
        markers: list[str],
        marker_emb: Any = None,  # noqa: ANN401 — unused (ids via _resolve)
    ) -> np.ndarray:
        """CLS forward with the resolved ``marker_ids`` → ``(n, 384)``.

        Runs the whole input ``x`` in one model call (fp32), passing the
        panel's resolved integer ``marker_ids`` (cached by
        :meth:`transform`). The caller controls how many patches arrive per
        call via ``encode_features(batch_size=...)``; cuBLAS picks
        batch-dependent kernels on GPU, so the fp32 result shifts slightly
        (~5e-5) with ``batch_size`` (CPU is bit-identical). Keeps the CLS
        token; the marker + token outputs are discarded (CORAL ships CLS
        only). ``marker_emb`` is unused.
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]

        ids = [r.marker_id for r in self._resolve(markers)]
        normed = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        dev = next(self._model.parameters()).device
        id_row = torch.tensor(ids, dtype=torch.long, device=dev)
        with torch.inference_mode():
            t = torch.from_numpy(normed).to(dev)  # pyright: ignore
            mids = id_row.unsqueeze(0).expand(t.shape[0], -1)
            cls = self._model(t, marker_ids=mids)[0]
        return cls.float().cpu().numpy().astype(np.float32)
