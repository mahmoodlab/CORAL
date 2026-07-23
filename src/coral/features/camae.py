"""``CAMAEExtractor`` — CA-MAE patch CLS features (channel-agnostic MAE).

Loads the Mahmood Lab **CA-MAE** (Channel-Agnostic Masked Autoencoder,
ViT-S/16) from a local HF-format checkpoint and returns the 384-d CLS
embedding per patch. CA-MAE ingests the multiplex channels **directly** (no
pseudo-RGB, no ImageNet) and **self-standardises inside the model** (``/255``
then a per-image ``InstanceNorm``), so :meth:`transform` is the scaffold
identity default and ``scale=False``: CORAL hands it the **raw** patch pixels
and the model normalises. (Feeding raw keeps the InstanceNorm out of its
``eps`` regime — InstanceNorm is scale-invariant for healthy variance, so the
absolute scale is washed; no marker z-score, per the matrix.) The per-channel
marker embedding (``return_channelwise_embeddings=True`` → ``(N, C*384)``) is
**never** shipped; CORAL serves CLS only, like KRONOS/KRONOS2.

Two checkpoint quirks are handled at load/forward:

- ``AutoModel(trust_remote_code=True)`` fails on this checkpoint (transformers
  copies only the entry module's *direct* relative imports, missing the
  transitive ``masking.py``), so the checkpoint **package** is imported
  directly (it ships ``__init__.py`` + its modules).
- The checkpoint's sincos ``pos_embed`` is sized for ``max_in_chans=11``;
  multiplex panels have more channels, so it is regenerated for the panel's
  channel count with the model's own (deterministic, non-learned) sincos
  function — the weights are channel-agnostic and pos_embed-length-agnostic.

CA-MAE is an **opt-in** dependency (the ``camae`` extra: torch + timm +
transformers). Imports are lazy, so this module loads and type-checks
without the extra, and ``EXTRACTOR_REGISTRY["CA-MAE"]`` is always present;
loading a model without the extra raises a clear error. CA-MAE is **not** on
a public Hub, so :meth:`from_pretrained` takes an explicit local checkpoint
path; the CLI build-once (:meth:`build`) discovers that checkpoint under
``assets/CA-MAE/`` by convention (see ``assets/README.md``).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from coral.assets import model_assets_dir
from coral.features import register
from coral.features.base import CoralEncoder
from coral.utils import resolve_device

# The heavy stack lives in the opt-in `camae` extra; probe without importing
# it (keeps this module importable + type-checkable without it).
CAMAE_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("timm") is not None
    and importlib.util.find_spec("transformers") is not None
)

_INSTALL_MSG = (
    "CA-MAE needs the optional `camae` extra (torch + timm + transformers) "
    "— install it: `uv sync --extra camae` (or `pip install coral[camae]`)."
)


def _is_camae_checkpoint(path: Path) -> bool:
    """True if ``path`` is a CA-MAE checkpoint package dir.

    The marker is the entry module ``huggingface_mae.py`` that
    :func:`_load_local_camae` imports.

    Example:
        >>> _is_camae_checkpoint(Path("/no/such/dir"))
        False
    """
    return path.is_dir() and (path / "huggingface_mae.py").is_file()


def _load_local_camae(checkpoint: str) -> tuple[Any, Any]:
    """Import a local CA-MAE checkpoint package; return (model, sincos_fn).

    ``AutoModel(trust_remote_code=True)`` fails on this checkpoint because
    transformers' dynamic-module copy misses the transitive ``masking.py``;
    the checkpoint dir is a Python package (``__init__.py`` + modules), so we
    import it directly and use its own ``MAEModel`` +
    ``generate_2d_sincos_pos_embeddings``.

    Args:
        checkpoint: Path to the local CA-MAE checkpoint dir. Its basename
            must be a valid Python package name.

    Returns:
        ``(model, sincos_fn)`` — the loaded (CPU) ``MAEModel`` and the
        module's sincos pos-embed generator (for the channel-count resize).
    """
    ckpt = Path(checkpoint)
    parent = str(ckpt.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    pkg = ckpt.name
    mae = importlib.import_module(f"{pkg}.huggingface_mae")
    vit = importlib.import_module(f"{pkg}.vit")
    model = mae.MAEModel.from_pretrained(str(ckpt))
    return model, vit.generate_2d_sincos_pos_embeddings


@register("CA-MAE")
class CAMAEExtractor(CoralEncoder):
    """CA-MAE channel-agnostic ViT-S/16 — per-patch 384-d CLS embedding.

    Build with :meth:`from_pretrained` (loads the model once, reused across
    slides). ``scale=False`` → the patcher hands it the raw patch pixels;
    :meth:`transform` is the identity default (the model self-standardises),
    and :meth:`forward` runs the batched CA-MAE pass and returns the CLS
    vector ``(n_patches, 384)``.

    Example:
        >>> CAMAEExtractor().required_markers() is None
        True
        >>> (
        ...     CAMAEExtractor.name,
        ...     CAMAEExtractor.embed_dim,
        ...     CAMAEExtractor.scale,
        ... )
        ('CA-MAE', 384, False)
    """

    name: ClassVar[str] = "CA-MAE"
    embed_dim: ClassVar[int] = 384  # ViT-S/16 CLS
    precision: ClassVar[str] = "float32"  # fp32, no autocast
    # Feed raw pixels; the model self-standardises (/255 + InstanceNorm), and
    # InstanceNorm washes the absolute scale (no marker z-score, per matrix).
    scale: ClassVar[bool] = False
    # CLS feature vector -> features/<slug>/CA-MAE/<variant>/features,
    # dims (patch, feature). The leading "patch" dim is implicit.
    output_schema: ClassVar[dict[str, tuple[str, ...]]] = {
        "features": ("feature",),
    }

    def __init__(
        self,
        model: Any = None,  # noqa: ANN401 — HF model
        pos_embed_fn: Any = None,  # noqa: ANN401 — sincos generator
    ) -> None:
        """Hold a loaded model (or ``None`` until :meth:`from_pretrained`)."""
        self._model = model
        self._pos_embed_fn = pos_embed_fn

    def required_markers(self) -> list[str] | None:
        """Marker-agnostic — CA-MAE encodes whatever channels it's given.

        Example:
            >>> CAMAEExtractor().required_markers() is None
            True
        """
        return None

    @classmethod
    def build(cls, device: str | None = None) -> CAMAEExtractor:
        """The CLI build-once hook — finds the checkpoint in ``assets/CA-MAE``.

        CA-MAE is not on a public Hub, so the user drops the local HF
        checkpoint package (a dir containing ``huggingface_mae.py``) under
        ``assets/CA-MAE/``. ``build()`` discovers it and loads via
        :meth:`from_pretrained`, so ``coral extract --extractor CA-MAE``
        works once it is in place. The package dir's name must be a valid
        Python identifier (no hyphens) — it is imported by name.

        Args:
            device: Torch device; defaults to GPU when available.

        Raises:
            FileNotFoundError: If no checkpoint package is found under
                ``assets/CA-MAE/``.

        Example:
            >>> CAMAEExtractor.build()  # doctest: +SKIP
        """
        base = model_assets_dir(cls.name)
        ckpts = (
            [p for p in sorted(base.glob("*")) if _is_camae_checkpoint(p)]
            if base.is_dir()
            else []
        )
        if not ckpts:
            raise FileNotFoundError(
                f"CA-MAE needs a local checkpoint package under {base}, "
                f"none found. Download the OpenPhenom model files from "
                f"https://huggingface.co/recursionpharma/OpenPhenom "
                f"(easiest to grab the whole repo folder; the key files "
                f"are huggingface_mae.py, config.json, and "
                f"model.safetensors) and place them in a subdirectory "
                f"whose name has no hyphens or spaces (it is imported as a "
                f"Python package), e.g. assets/CA-MAE/openphenom/ in the "
                f"CORAL repo root."
            )
        return cls.from_pretrained(str(ckpts[0]), device=device)

    @classmethod
    def from_pretrained(
        cls, checkpoint: str, *, device: str | None = None
    ) -> CAMAEExtractor:
        """Load CA-MAE from a local checkpoint dir into the extractor.

        Args:
            checkpoint: Path to a local CA-MAE HF checkpoint dir (package
                with ``huggingface_mae.py`` + ``vit.py``).
            device: Torch device; defaults to GPU when available.

        Returns:
            A ``CAMAEExtractor`` holding the loaded model, pinned to the CLS
            path (``return_channelwise_embeddings = False``) and ``.eval()``.

        Raises:
            ImportError: If the ``camae`` extra is not installed.

        Example:
            >>> CAMAEExtractor.from_pretrained("ckpt")  # doctest: +SKIP
        """
        if not CAMAE_AVAILABLE:
            raise ImportError(_INSTALL_MSG)
        model, pos_embed_fn = _load_local_camae(checkpoint)
        # CLS only — never the per-channel (N, C*384) marker path.
        model.return_channelwise_embeddings = False
        model = model.eval()
        dev = device or resolve_device()
        model = model.to(dev)
        return cls(model=model, pos_embed_fn=pos_embed_fn)

    def _ensure_pos_embed(self, n_channels: int) -> None:
        """Resize the sincos ``pos_embed`` to cover ``n_channels`` channels.

        The HF checkpoint ships a pos_embed for ``max_in_chans=11``;
        multiplex panels have more, so regenerate it with the model's own
        (deterministic, non-learned) sincos function. Monotonic: only grows,
        so a later smaller panel reuses the existing (longer) buffer.
        """
        vitb = self._model.encoder.vit_backbone
        grid = int(vitb.patch_embed.grid_size[0])
        cls_n = 1 if vitb.cls_token is not None else 0
        needed = cls_n + n_channels * grid * grid
        if int(vitb.pos_embed.shape[1]) >= needed:
            return
        import torch  # pyright: ignore[reportMissingImports]

        pe = self._pos_embed_fn(
            vitb.embed_dim,
            length=grid,
            use_class_token=vitb.cls_token is not None,
            num_modality=n_channels,
        )
        # pos_embed is a (non-learned) Parameter; .to() yields a Tensor, so
        # re-wrap to keep the module's parameter contract.
        vitb.pos_embed = torch.nn.Parameter(
            pe.to(vitb.pos_embed.device), requires_grad=False
        )

    def forward(
        self,
        x: Any,  # noqa: ANN401 — raw patch batch
        markers: list[str],
        marker_emb: Any = None,  # noqa: ANN401 — unused (marker-agnostic)
    ) -> np.ndarray:
        """Batched CA-MAE forward → ``(N, 384)`` CLS features.

        Runs the whole input ``x`` in one model call; fp32, no autocast.
        CA-MAE self-standardises per image, so each patch is encoded
        independently of the others in the call — but cuBLAS picks
        batch-dependent kernels on GPU, so the fp32 result still shifts
        slightly (~5e-5) with ``batch_size`` (CPU is bit-identical). The
        caller controls how many patches arrive per call via
        ``encode_features(batch_size=...)``. ``markers`` / ``marker_emb``
        are unused (CA-MAE is channel-agnostic). The model's own
        ``/255 + InstanceNorm`` runs inside ``predict`` — ``x`` is the
        identity-passed raw patch batch; the pos_embed is resized to the
        panel's channel count first.
        """
        if self._model is None:
            raise RuntimeError(_INSTALL_MSG)
        import torch  # pyright: ignore[reportMissingImports]

        arr = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
        self._ensure_pos_embed(arr.shape[1])
        dev = next(self._model.parameters()).device
        with torch.inference_mode():
            t = torch.from_numpy(arr).to(dev)  # pyright: ignore
            cls = self._model.predict(t).float().cpu().numpy()
        return cls.astype(np.float32)
