"""``CartaTissueSegmenter`` — CARTA DeepLabV3-ResNet50 DAPI tissue segmenter.

Wraps the **CARTA** tissue segmenter (a torchvision ``deeplabv3_resnet50`` with
a single-class head, trained on multiplex-IF DAPI) as a CORAL
:class:`~coral.tissue.base.BaseTissueSegmenter`, registered as ``"carta"``.
It is the DL alternative to
:class:`~coral.tissue.otsu.OtsuTissueSegmenter` — a
learned, nuclear-only detector that needs no per-dataset channel tuning.

The model code is **vendored** into :mod:`coral.tissue._carta` (no dependency
on the private CARTA repo), imported lazily, so this module loads + type-checks
and ``SEGMENTER_REGISTRY["carta"]`` is always present even without the optional
``carta`` extra installed; loading the model without it raises a clear
:class:`ImportError`.

CARTA is **nuclear-only**: it uses channel 0 (DAPI) and ignores structural
channels. :meth:`forward` mirrors CARTA's per-crop path — resample the nuclear
plane to the 1.0 µm/px inference scale (the model's training scale), run tiled
DeepLab, apply CARTA's µm²-based mask cleanup, then upsample back to native.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np

from coral.tissue._carta.core_mask_cleanup import (
    DEFAULT_CORE_MASK_CLEANUP,
    CoreMaskCleanup,
    apply_core_mask_cleanup,
)
from coral.tissue._carta.seg_scale import (
    SEG_INFERENCE_UM_PER_PX,
    mask_seg_to_native,
    resample_to_seg_scale,
)
from coral.tissue.base import BaseTissueSegmenter
from coral.tissue.registry import register

#: Black border added around the nuclear plane before segmentation, in the
#: plane's own pixels, and cropped off afterwards.
#:
#: A core is a tight crop, so its tissue can sit hard against the array edge.
#: CARTA tiles at 512 px with `overlap: 0`, and when a dimension is not a
#: multiple of 512 the final tile is pulled flush against the edge and lands
#: partly on top of its neighbour. In that strip two predictions are averaged;
#: where they disagree the mean sits near the 0.5 threshold and the boundary
#: cuts a straight line along the seam. That is the reported symptom: a
#: boundary that suddenly breaks and overflows.
#:
#: Padding moves those seams off the tissue. Measured on two affected cores,
#: running the real chain: the first lost the break where its contour ran off
#: the bottom-left edge and went from two components to one (5.5% of pixels
#: changed); the second lost its straight right-hand edge and its island chain
#: became one shape (7.0% changed, coverage 0.722 -> 0.698).
#:
#: 300 px because that is what was tested. It is not a tuned optimum, and at
#: 0.5 um/px it costs about 150 um of margin and a few percent of inference
#: area. Coverage barely moves either way; the pixels that move are the ones
#: at the boundary, which is the whole point.
SEG_BORDER_PX: int = 300

#: CORAL's minimum tissue component, overriding CARTA's 17,000 um2.
#:
#: This is a CORAL choice about a knob the vendored code exposes, not an edit
#: to vendored behaviour: the dataclass default in
#: `_carta/core_mask_cleanup.py` still reads 17,000, and CARTA run on its own
#: is unchanged.
#:
#: Worth knowing: on the two cores above this changes nothing. Neither has a
#: component between 17,000 and 50,000 um2, so 30k and 50k gave masks identical
#: to 17k. It is here for the cores that do.
MIN_COMPONENT_AREA_UM2: float = 40_000.0

#: CARTA's cleanup with CORAL's minimum component. Built by `replace` so every
#: other field still tracks upstream if CARTA changes it.
CORAL_CORE_MASK_CLEANUP = replace(
    DEFAULT_CORE_MASK_CLEANUP, min_component_area_um2=MIN_COMPONENT_AREA_UM2
)

_DEFAULT_REF = "hf_hub:MahmoodLab/CARTA"
_WEIGHTS_FILE = "carta_tissue.pt"
_INSTALL_MSG = (
    "CARTA tissue segmentation needs the optional `carta` extra (torch + "
    "torchvision) — install it: `uv sync --extra carta` (or "
    "`pip install coral[carta]`). Weights download automatically from the "
    "public Hugging Face repo MahmoodLab/CARTA, or point at a local "
    "checkpoint via CARTA_TISSUE_WEIGHTS / from_pretrained(<path>)."
)
_LOAD_MSG = (
    "CartaTissueSegmenter has no loaded model — build it with "
    "CartaTissueSegmenter.from_pretrained(...) (or .build()) first."
)


def _download_hf_weights(repo_id: str) -> str:
    """Download the CARTA checkpoint from HF, or raise an actionable hint."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=_WEIGHTS_FILE, token=token
        )
    except Exception as exc:  # noqa: BLE001 — surface an actionable hint
        raise RuntimeError(
            f"Could not fetch CARTA weights from hf_hub:{repo_id} "
            f"({_WEIGHTS_FILE}). The repo is public — check your network "
            f"connection, or set CARTA_TISSUE_WEIGHTS to a local .pt."
        ) from exc


@register("carta")
class CartaTissueSegmenter(BaseTissueSegmenter):
    """CARTA DeepLabV3 DAPI tissue segmenter (nuclear-only, vendored).

    Build with :meth:`from_pretrained` / :meth:`build` (loads the model once,
    reused across slides). Post-processing knobs tune only the mask cleanup
    (not the fixed, checkpoint-locked model):

    Args:
        model: A loaded vendored ``DeepLabV3Segmenter`` (or ``None`` until
            :meth:`from_pretrained`).
        preserve_holes: Keep interior holes in the mask (like CARTA's
            ``--preserve-tissue-holes``) — sets the cleanup's bounded hole-fill
            to zero. Default ``False`` (small holes are filled).
        mask_cleanup: CARTA's µm²-based cleanup (small-component removal +
            bounded hole-fill + light closing), or ``None`` for the raw mask.
        border_px: Black margin added around the plane before segmentation and
            cropped off after, in the plane's own pixels. Keeps CARTA's tile
            seams off the tissue; see :data:`SEG_BORDER_PX`. ``0`` disables it.
        region_hull: Also wrap the mask in CORAL's alpha-shape tissue region
            (the Otsu post-processing) — off by default; useful for A/B
            comparison.
        max_bridge_distance: The alpha-shape knob (microns), used only when
            ``region_hull`` is set.

    Example:
        >>> CartaTissueSegmenter().name
        'carta'
        >>> CartaTissueSegmenter().required_channels() is None
        True
        >>> CartaTissueSegmenter(preserve_holes=True).params["preserve_holes"]
        True
    """

    name: ClassVar[str] = "carta"
    # Informational: CARTA infers at the 1.0 µm/px training scale (~10x). The
    # working mpp is passed to ``forward`` and drives the resample, so this
    # attribute is provenance only — it does not select a pyramid level.
    target_mag: ClassVar[float] = 10.0

    def __init__(
        self,
        model: Any = None,  # noqa: ANN401 — loaded vendored DeepLabV3Segmenter
        *,
        preserve_holes: bool = False,
        mask_cleanup: CoreMaskCleanup | None = CORAL_CORE_MASK_CLEANUP,
        region_hull: bool = False,
        max_bridge_distance: float = 200.0,
        border_px: int = SEG_BORDER_PX,
    ) -> None:
        """Store the model (if any) and the post-processing knobs."""
        self._model = model
        self._ref: str | None = None
        self.preserve_holes = preserve_holes
        self.mask_cleanup = mask_cleanup
        self.region_hull = region_hull
        self.border_px = border_px
        self.max_bridge_distance = max_bridge_distance

    @classmethod
    def build(
        cls,
        device: str | None = None,
        *,
        preserve_holes: bool = False,
        region_hull: bool = False,
        border_px: int = SEG_BORDER_PX,
    ) -> CartaTissueSegmenter:
        """Load the default CARTA checkpoint (the CLI build-once hook).

        Args:
            device: Torch device; ``None`` → CARTA ``auto`` (GPU if present).
            preserve_holes: See :class:`CartaTissueSegmenter`.
            region_hull: See :class:`CartaTissueSegmenter`.
            border_px: See :class:`CartaTissueSegmenter`.

        Example:
            >>> seg = CartaTissueSegmenter.build()  # doctest: +SKIP
        """
        return cls.from_pretrained(
            device=device,
            preserve_holes=preserve_holes,
            region_hull=region_hull,
            border_px=border_px,
        )

    @classmethod
    def from_pretrained(
        cls,
        ref: str = _DEFAULT_REF,
        *,
        device: str | None = None,
        preserve_holes: bool = False,
        mask_cleanup: CoreMaskCleanup | None = CORAL_CORE_MASK_CLEANUP,
        region_hull: bool = False,
        max_bridge_distance: float = 200.0,
        border_px: int = SEG_BORDER_PX,
    ) -> CartaTissueSegmenter:
        """Load the vendored CARTA DeepLab model into the segmenter.

        Weights resolve in this order: the ``CARTA_TISSUE_WEIGHTS`` env var (a
        local ``.pt``) → a ``"hf_hub:<repo>"`` ref (downloaded from the public
        Hub) → otherwise ``ref`` is treated as a local path.

        Args:
            ref: ``"hf_hub:<repo>"`` or a local ``.pt`` path.
            device: Torch device; ``None`` → CARTA ``auto``.
            preserve_holes: See :class:`CartaTissueSegmenter`.
            mask_cleanup: See :class:`CartaTissueSegmenter`.
            region_hull: See :class:`CartaTissueSegmenter`.
            border_px: See :class:`CartaTissueSegmenter`.
            max_bridge_distance: See :class:`CartaTissueSegmenter`.

        Returns:
            A ready-to-segment ``CartaTissueSegmenter``.

        Raises:
            ImportError: If the ``carta`` extra (torch+torchvision) is absent.

        Example:
            >>> CartaTissueSegmenter.from_pretrained()  # doctest: +SKIP
        """
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("torchvision") is None
        ):
            raise ImportError(_INSTALL_MSG)

        local = os.environ.get("CARTA_TISSUE_WEIGHTS")
        if local:
            weights_path = resolved_ref = local
            from_hub = False
        elif ref.startswith("hf_hub:"):
            weights_path = _download_hf_weights(ref[len("hf_hub:") :])
            resolved_ref = ref
            from_hub = True
        else:
            weights_path = resolved_ref = ref
            from_hub = False

        # A local path must exist — otherwise the vendored loader would
        # silently keep the randomly-initialized head and emit garbage masks.
        if not from_hub and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"CARTA tissue weights not found at {weights_path!r}. Point "
                f"CARTA_TISSUE_WEIGHTS at a valid .pt, pass a local path to "
                f"from_pretrained(), or use the default public hf_hub ref "
                f"(MahmoodLab/CARTA)."
            )

        from coral.tissue._carta.config import load_config
        from coral.tissue._carta.deeplabv3 import DeepLabV3Segmenter

        cfg = load_config()
        if device is not None:
            cfg.training.device = device
        model = DeepLabV3Segmenter(cfg=cfg, weights_path=weights_path)
        obj = cls(
            model=model,
            preserve_holes=preserve_holes,
            mask_cleanup=mask_cleanup,
            region_hull=region_hull,
            border_px=border_px,
            max_bridge_distance=max_bridge_distance,
        )
        obj._ref = resolved_ref
        return obj

    @property
    def params(self) -> dict[str, Any]:
        """JSON-safe tuning parameters, recorded into ``tissue/config.json``.

        Example:
            >>> CartaTissueSegmenter().params["seg_um_per_px"]
            1.0
        """
        inf = getattr(getattr(self._model, "cfg", None), "inference", None)
        cleanup = None
        if self.mask_cleanup is not None:
            fill = (
                0.0
                if self.preserve_holes
                else self.mask_cleanup.fill_holes_max_area_um2
            )
            cleanup = {
                "min_component_area_um2": (
                    self.mask_cleanup.min_component_area_um2
                ),
                "fill_holes_max_area_um2": fill,
                "closing_radius_um": self.mask_cleanup.closing_radius_um,
            }
        return {
            "weights_ref": self._ref,
            "tile_size": getattr(inf, "tile_size", 512),
            "confidence": getattr(inf, "confidence", 0.5),
            "dapi_rgb_mode": getattr(inf, "dapi_rgb_mode", "replicate"),
            "seg_um_per_px": SEG_INFERENCE_UM_PER_PX,
            "preserve_holes": self.preserve_holes,
            "border_px": self.border_px,
            "region_hull": self.region_hull,
            "max_bridge_distance": (
                self.max_bridge_distance if self.region_hull else None
            ),
            "mask_cleanup": cleanup,
        }

    def required_channels(self) -> list[str] | None:
        """``None`` — the nuclear channel is inferred by the slide driver.

        CARTA is nuclear-only; structural channels in the stack are ignored.

        Returns:
            Always ``None`` (nuclear channel inferred at call time).

        Example:
            >>> from coral.tissue import CartaTissueSegmenter
            >>> CartaTissueSegmenter().required_channels() is None
            True
        """
        return None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Identity — CARTA normalizes the DAPI plane internally.

        Args:
            image: ``(c, y, x)`` stack (channel 0 nuclear) or ``(y, x)`` plane.

        Returns:
            The image unchanged (as a numpy array); the model does its own
            percentile normalization + DAPI→RGB, matching CARTA exactly.

        Example:
            >>> import numpy as np
            >>> from coral.tissue import CartaTissueSegmenter
            >>> img = np.zeros((1, 4, 4), dtype="uint16")
            >>> CartaTissueSegmenter().preprocess(img).shape
            (1, 4, 4)
        """
        return np.asarray(image)

    def forward(self, image: np.ndarray, mpp: float = 1.0) -> np.ndarray:
        """Segment the nuclear plane at CARTA's locked scale, then upsample.

        Args:
            image: ``(c, y, x)`` stack (channel 0 nuclear) or ``(y, x)`` plane.
            mpp: Working microns-per-pixel of ``image`` (drives the resample to
                the 1.0 µm/px inference scale).

        Returns:
            Boolean ``(y, x)`` tissue mask at the input resolution.

        Raises:
            RuntimeError: If no loaded model is present.

        Example:
            Load weights then run the nuclear plane (needs the carta
            extra)::

                from coral.tissue import CartaTissueSegmenter

                seg = CartaTissueSegmenter.from_pretrained()
                mask = seg.forward(nuclear_plane, mpp=0.5)
        """
        if self._model is None:
            raise RuntimeError(_LOAD_MSG)
        arr = np.asarray(image)
        nuclear = arr[0] if arr.ndim == 3 else arr
        # Padded with black before the resample, cropped after. See
        # `border_px` for what this fixes and how it was measured.
        pad = max(0, int(self.border_px))
        padded = np.pad(nuclear, pad, constant_values=0) if pad else nuclear
        crop_seg = resample_to_seg_scale(padded, mpp)
        mask_seg = np.asarray(self._model.segment(crop_seg), dtype=bool)
        cleaned, _stats = apply_core_mask_cleanup(
            mask_seg, self._effective_cleanup()
        )
        if self.region_hull:
            from coral.tissue.hull import alpha_shape_region

            cleaned = alpha_shape_region(
                cleaned,
                SEG_INFERENCE_UM_PER_PX,
                max_bridge_um=self.max_bridge_distance,
            )
        native = np.asarray(
            mask_seg_to_native(cleaned, padded.shape[:2]), dtype=bool
        )
        return native[pad:-pad, pad:-pad] if pad else native

    def segment(self, image: np.ndarray, mpp: float = 1.0) -> np.ndarray:
        """Run :meth:`preprocess` then :meth:`forward` at ``mpp``.

        Args:
            image: ``(c, y, x)`` stack (channel 0 nuclear) or ``(y, x)`` plane.
            mpp: Working microns-per-pixel of ``image``.

        Returns:
            Boolean ``(y, x)`` tissue mask.

        Example:
            >>> CartaTissueSegmenter.build().segment(  # doctest: +SKIP
            ...     nuclear_plane, mpp=0.5
            ... )
        """
        return self.forward(self.preprocess(image), mpp)

    def _effective_cleanup(self) -> CoreMaskCleanup | None:
        """The cleanup with ``preserve_holes`` applied (hole-fill → 0)."""
        if self.mask_cleanup is None:
            return None
        if self.preserve_holes:
            return replace(self.mask_cleanup, fill_holes_max_area_um2=0.0)
        return self.mask_cleanup
