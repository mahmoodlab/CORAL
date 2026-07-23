"""``UNIPostExtractor`` — post-average UNI (colorize each channel, average).

Canonical :class:`~coral.features.uni.UNIExtractor` blends **all** channels
into one pseudo-RGB image and runs UNI **once** (average-then-extract)::

    cls = UNI( mean_c  channel_c · color_c )

``UNIPostExtractor`` swaps the order: it colorizes **each channel
independently** into its own pseudo-RGB image (same colormap), runs UNI on
each, and averages the per-channel embeddings (extract-then-average)::

    cls = mean_c  UNI( channel_c · color_c )

Every marker's signal survives the backbone intact and only mixes in embedding
space. Same ViT-L/16 weights (``hf-hub:MahmoodLab/UNI``), same ImageNet
normalization, same 1024-d CLS surface — a drop-in comparison against canonical
UNI. Only :meth:`transform` and :meth:`forward` differ, so everything else
(model loading, marker-agnosticism, dims) is inherited from ``UNIExtractor``.

Shares the ``uni`` extra with canonical UNI; loading without it raises the same
clear error, and ``EXTRACTOR_REGISTRY["UNIpost"]`` is always present.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from coral.features import register
from coral.features.transforms import colorize_per_channel, imagenet_normalize
from coral.features.uni import _INSTALL_MSG, UNIExtractor


@register("UNIpost")
class UNIPostExtractor(UNIExtractor):
    """Post-average UNI — per-channel UNI pass, then mean over channels.

    Built exactly like :class:`~coral.features.uni.UNIExtractor` (same weights,
    via :meth:`from_pretrained`); the difference is the preprocessing order.
    ``transform`` colorizes each channel into its own pseudo-RGB image
    (``(B, C, 3, H, W)``) and ``forward`` runs UNI on all ``B·C`` images in one
    batched call, then averages the per-channel embeddings → ``(B, 1024)``.

    Example:
        >>> UNIPostExtractor().required_markers() is None
        True
        >>> (UNIPostExtractor.name, UNIPostExtractor.embed_dim)
        ('UNIpost', 1024)
    """

    name: ClassVar[str] = "UNIpost"

    def transform(
        self,
        patches: Any,  # noqa: ANN401 — float32 [0, 1] patch batch
        markers: list[str],
        *,
        nuclear_marker: str | None = None,
    ) -> Any:  # noqa: ANN401 — torch tensor
        """Multiplex patch → per-channel pseudo-RGB → ImageNet.

        Unlike canonical UNI, channels are **not** blended: each becomes its
        own pseudo-RGB image, yielding ``(B, C, 3, H, W)``, ImageNet-normalised
        (the RGB stats broadcast over the extra channel axis). No z-score.
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]

        dev = next(self._model.parameters()).device
        arr = np.ascontiguousarray(patches, dtype=np.float32)
        t = torch.from_numpy(arr).to(dev)  # pyright: ignore
        return imagenet_normalize(colorize_per_channel(t))

    def forward(
        self,
        x: Any,  # noqa: ANN401 — (B, C, 3, H, W) pseudo-RGB torch tensor
        markers: list[str],
        marker_emb: Any = None,  # noqa: ANN401 — unused (marker-agnostic)
    ) -> np.ndarray:
        """Batched per-channel UNI forward, averaged → ``(B, 1024)`` CLS.

        Flattens the ``(B, C, 3, H, W)`` input to a single ``(B·C, 3, H, W)``
        batch, runs UNI once (one kernel, fp16 autocast on CUDA — same path as
        :meth:`UNIExtractor.forward`), then reshapes and means over the ``C``
        axis. The mean is order-independent, so this matches the per-channel
        loop up to fp16 kernel scale; on CPU it is bit-identical.
        ``markers``/``marker_emb`` are unused (UNI is RGB / marker-agnostic).
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]

        dev = next(self._model.parameters()).device
        dtype = getattr(torch, self.precision)
        b, c, _, h, w = x.shape
        flat = x.reshape(b * c, 3, h, w)
        with (
            torch.inference_mode(),
            torch.autocast(
                dev.type, dtype=dtype, enabled=(dev.type == "cuda")
            ),
        ):
            emb = self._model(flat).float()  # (B·C, 1024)
        cls = emb.reshape(b, c, -1).mean(dim=1).cpu().numpy()  # (B, 1024)
        return cls.astype(np.float32)
