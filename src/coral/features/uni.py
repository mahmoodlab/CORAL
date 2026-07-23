"""``UNIExtractor`` — UNI patch CLS features (RGB pathology ViT-L/16).

Loads Mahmood Lab's **UNI** (``hf-hub:MahmoodLab/UNI``) via ``timm`` and
returns the 1024-d CLS embedding per patch. UNI is an **RGB** pathology
model, so the multiplex patch is collapsed to a pseudo-RGB image (the shared
60-colour blend in :mod:`coral.features.transforms`) and ImageNet-normalised —
canonical UNI takes **no** marker z-score (it's marker-agnostic;
``embed_markers → None``). See the matrix doc for why the kronos
``rgb_norm`` gold's extra z-score is *not* reproduced here.

UNI is an **opt-in** dependency (the ``uni`` extra: torch + timm). Imports
are lazy, so this module loads and type-checks without the extra, and
``EXTRACTOR_REGISTRY["UNI"]`` is always present; loading without the extra
raises a clear error.
"""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar

import numpy as np

from coral.features import register
from coral.features.base import CoralEncoder
from coral.features.transforms import imagenet_normalize, multiplex_to_rgb
from coral.utils import resolve_device

# The heavy stack lives in the opt-in `uni` extra; probe without importing.
UNI_AVAILABLE = (
    importlib.util.find_spec("timm") is not None
    and importlib.util.find_spec("torch") is not None
)

_DEFAULT_REF = "hf-hub:MahmoodLab/UNI"
_INSTALL_MSG = (
    "UNI needs the optional `uni` extra (torch + timm) — install it: "
    "`uv sync --extra uni` (or `pip install coral[uni]`)."
)


@register("UNI")
class UNIExtractor(CoralEncoder):
    """UNI RGB pathology ViT-L/16 — per-patch 1024-d CLS embedding.

    Build with :meth:`from_pretrained` (loads the model once, reused across
    slides). ``scale=True`` → the patcher hands it float32 ``[0, 1]``
    patches; :meth:`transform` collapses them to pseudo-RGB + ImageNet
    (channel-order-dependent), and :meth:`forward` runs the batched UNI pass.

    Example:
        >>> UNIExtractor().required_markers() is None
        True
        >>> (UNIExtractor.name, UNIExtractor.embed_dim)
        ('UNI', 1024)
    """

    name: ClassVar[str] = "UNI"
    embed_dim: ClassVar[int] = 1024
    precision: ClassVar[str] = "float16"  # RGB ViT, TRIDENT-canonical
    output_schema: ClassVar[dict[str, tuple[str, ...]]] = {
        "features": ("feature",),
    }

    def __init__(self, model: Any = None) -> None:  # noqa: ANN401 — timm model
        """Hold a loaded model (or ``None`` until :meth:`from_pretrained`)."""
        self._model = model

    def required_markers(self) -> list[str] | None:
        """Marker-agnostic — UNI is RGB (markers only set channel order)."""
        return None

    @classmethod
    def build(cls, device: str | None = None) -> UNIExtractor:
        """Load the default UNI checkpoint (the CLI build-once hook).

        Args:
            device: Torch device; defaults to GPU when available.

        Example:
            >>> ext = UNIExtractor.build()  # doctest: +SKIP
        """
        return cls.from_pretrained(_DEFAULT_REF, device=device)

    @classmethod
    def from_pretrained(
        cls, ref: str = _DEFAULT_REF, *, device: str | None = None
    ) -> UNIExtractor:
        """Load UNI from the Hub via ``timm`` into the encoder.

        Args:
            ref: A ``timm`` model ref (default ``hf-hub:MahmoodLab/UNI``).
            device: Torch device; defaults to CUDA when available.

        Raises:
            ImportError: If the ``uni`` extra is not installed.

        Example:
            >>> UNIExtractor.from_pretrained()  # doctest: +SKIP
        """
        if not UNI_AVAILABLE:
            raise ImportError(_INSTALL_MSG)
        import timm  # pyright: ignore[reportMissingImports]

        model = timm.create_model(
            ref, pretrained=True, init_values=1e-5, dynamic_img_size=True
        )
        dev = device or resolve_device()
        return cls(model=model.to(dev).eval())

    def transform(
        self,
        patches: Any,  # noqa: ANN401 — float32 [0, 1] patch batch
        markers: list[str],
        *,
        nuclear_marker: str | None = None,
    ) -> Any:  # noqa: ANN401 — torch tensor
        """Multiplex patch → pseudo-RGB → ImageNet (the UNI input).

        ``scale=True`` means the patcher already produced float32 ``[0, 1]``
        patches; this lifts them to the model's device, blends the channels
        to pseudo-RGB and ImageNet-normalises. No marker z-score.
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]

        dev = next(self._model.parameters()).device
        arr = np.ascontiguousarray(patches, dtype=np.float32)
        t = torch.from_numpy(arr).to(dev)  # pyright: ignore
        return imagenet_normalize(multiplex_to_rgb(t))

    def forward(
        self,
        x: Any,  # noqa: ANN401 — pseudo-RGB torch tensor
        markers: list[str],
        marker_emb: Any = None,  # noqa: ANN401 — unused (marker-agnostic)
    ) -> np.ndarray:
        """Batched UNI forward → ``(N, 1024)`` CLS features.

        Runs the whole input ``x`` in one model call under
        ``autocast(precision)`` on CUDA; the caller controls how many
        patches arrive per call via ``encode_features(batch_size=...)``.
        On CUDA, UNI runs under fp16 autocast, so cuBLAS picks
        batch-dependent kernels and the CLS features shift slightly
        (fp16-scale) with ``batch_size``; on CPU they are bit-identical.
        ``markers``/``marker_emb`` are unused (UNI is RGB / marker-agnostic).
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]

        dev = next(self._model.parameters()).device
        dtype = getattr(torch, self.precision)
        with (
            torch.inference_mode(),
            torch.autocast(
                dev.type, dtype=dtype, enabled=(dev.type == "cuda")
            ),
        ):
            cls = self._model(x).float().cpu().numpy()
        return cls.astype(np.float32)
