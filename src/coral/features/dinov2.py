"""``DINOv2Extractor`` — DINOv2 patch CLS features (RGB ViT-L/14).

Loads Meta AI's **DINOv2** (``vit_large_patch14_reg4_dinov2.lvd142m``, a
ViT-L/14 with register tokens) via ``timm`` and returns the 1024-d CLS
embedding per patch. DINOv2 is a self-supervised **RGB** model — the
same family as :mod:`coral.features.uni` — so the multiplex patch is
collapsed to the shared pseudo-RGB blend
(:mod:`coral.features.transforms`) and ImageNet-normalised, with **no**
marker z-score (it's marker-agnostic; ``embed_markers → None``).

The one preprocessing step DINOv2 adds over UNI is a **÷14 center-crop**:
ViT-L/14's patch stride does not divide a 256-px patch, so each patch is
cropped to the largest multiple of 14 (256 → 252) before the pseudo-RGB
blend. The crop is kept local to this module — UNI (patch-16) never
needs it, so it is not hoisted into the shared transforms.

Unlike UNI/KRONOS2, the DINOv2 weights are **public** — ``timm``
downloads them from the Hub automatically, so there is no gated repo and
no HF token. DINOv2 is still an **opt-in** dependency (the ``dinov2``
extra: torch + timm). Imports are lazy, so this module loads and
type-checks without the extra, and ``EXTRACTOR_REGISTRY["DINOv2"]`` is
always present; loading without the extra raises a clear error.
"""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar

import numpy as np

from coral.features import register
from coral.features.base import CoralEncoder
from coral.features.transforms import imagenet_normalize, multiplex_to_rgb
from coral.utils import resolve_device

# The heavy stack lives in the opt-in `dinov2` extra; probe without import.
DINOV2_AVAILABLE = (
    importlib.util.find_spec("timm") is not None
    and importlib.util.find_spec("torch") is not None
)

# Public timm ref (ViT-L/14 + 4 register tokens, LVD-142M); ungated.
_DEFAULT_REF = "vit_large_patch14_reg4_dinov2.lvd142m"
_PATCH_SIZE = 14  # ViT-L/14 — input H, W must be multiples of this.
_INSTALL_MSG = (
    "DINOv2 needs the optional `dinov2` extra (torch + timm) — install "
    "it: `uv sync --extra dinov2` (or `pip install coral[dinov2]`)."
)


def _crop_to_multiple(patches: Any, multiple: int) -> Any:  # noqa: ANN401
    """Center-crop the last two dims to a multiple of ``multiple``.

    ViT-L/14's patch stride (14) does not divide a 256-px patch, so the
    spatial dims are center-cropped to the largest multiple that fits
    (256 → 252) before the model. Pure index slicing, so it behaves
    identically on a numpy array or a torch tensor.

    Args:
        patches: ``(..., H, W)`` array or tensor.
        multiple: The divisor the cropped H and W must be a multiple of
            (DINOv2: 14).

    Returns:
        The center-cropped ``(..., H', W')`` view, where ``H'``/``W'`` are
        the largest multiples of ``multiple`` not exceeding ``H``/``W``.

    Example:
        >>> import numpy as np
        >>> _crop_to_multiple(np.zeros((1, 3, 256, 256)), 14).shape
        (1, 3, 252, 252)
    """
    h, w = patches.shape[-2], patches.shape[-1]
    hc, wc = (h // multiple) * multiple, (w // multiple) * multiple
    top, left = (h - hc) // 2, (w - wc) // 2
    return patches[..., top : top + hc, left : left + wc]


@register("DINOv2")
class DINOv2Extractor(CoralEncoder):
    """DINOv2 RGB ViT-L/14 — per-patch 1024-d CLS embedding.

    Build with :meth:`from_pretrained` (loads the model once, reused
    across slides). ``scale=True`` → the patcher hands it float32
    ``[0, 1]`` patches; :meth:`transform` center-crops them to ÷14,
    collapses to pseudo-RGB + ImageNet (channel-order-dependent), and
    :meth:`forward` runs the batched DINOv2 pass.

    Example:
        >>> DINOv2Extractor().required_markers() is None
        True
        >>> (DINOv2Extractor.name, DINOv2Extractor.embed_dim)
        ('DINOv2', 1024)
    """

    name: ClassVar[str] = "DINOv2"
    embed_dim: ClassVar[int] = 1024
    precision: ClassVar[str] = "float16"  # RGB ViT, TRIDENT-canonical
    output_schema: ClassVar[dict[str, tuple[str, ...]]] = {
        "features": ("feature",),
    }

    def __init__(self, model: Any = None) -> None:  # noqa: ANN401 — timm model
        """Hold a loaded model (or ``None`` until :meth:`from_pretrained`)."""
        self._model = model

    def required_markers(self) -> list[str] | None:
        """Marker-agnostic — DINOv2 is RGB (markers set channel order)."""
        return None

    @classmethod
    def build(cls, device: str | None = None) -> DINOv2Extractor:
        """Load the default DINOv2 checkpoint (the CLI build-once hook).

        Args:
            device: Torch device; defaults to GPU when available.

        Example:
            >>> ext = DINOv2Extractor.build()  # doctest: +SKIP
        """
        return cls.from_pretrained(_DEFAULT_REF, device=device)

    @classmethod
    def from_pretrained(
        cls, ref: str = _DEFAULT_REF, *, device: str | None = None
    ) -> DINOv2Extractor:
        """Load DINOv2 from the Hub via ``timm`` into the encoder.

        Args:
            ref: A ``timm`` model ref (default
                ``vit_large_patch14_reg4_dinov2.lvd142m`` — public/ungated).
            device: Torch device; defaults to CUDA when available.

        Raises:
            ImportError: If the ``dinov2`` extra is not installed.

        Example:
            >>> DINOv2Extractor.from_pretrained()  # doctest: +SKIP
        """
        if not DINOV2_AVAILABLE:
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
        """Multiplex patch → ÷14 crop → pseudo-RGB → ImageNet (the input).

        ``scale=True`` means the patcher already produced float32
        ``[0, 1]`` patches; this lifts them to the model's device,
        center-crops to a multiple of 14 (ViT-L/14), blends the channels
        to pseudo-RGB and ImageNet-normalises. No marker z-score.
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]

        dev = next(self._model.parameters()).device
        arr = np.ascontiguousarray(patches, dtype=np.float32)
        t = torch.from_numpy(arr).to(dev)  # pyright: ignore
        cropped = _crop_to_multiple(t, _PATCH_SIZE)
        return imagenet_normalize(multiplex_to_rgb(cropped))

    def forward(
        self,
        x: Any,  # noqa: ANN401 — pseudo-RGB torch tensor
        markers: list[str],
        marker_emb: Any = None,  # noqa: ANN401 — unused (marker-agnostic)
    ) -> np.ndarray:
        """Batched DINOv2 forward → ``(N, 1024)`` CLS features.

        Runs the whole input ``x`` in one model call under
        ``autocast(precision)`` on CUDA; the caller controls how many
        patches arrive per call via ``encode_features(batch_size=...)``.
        On CUDA, DINOv2 runs under fp16 autocast, so cuBLAS picks
        batch-dependent kernels and the CLS features shift slightly
        (fp16-scale) with ``batch_size``; on CPU they are bit-identical.
        ``markers``/``marker_emb`` are unused (DINOv2 is RGB /
        marker-agnostic).
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
