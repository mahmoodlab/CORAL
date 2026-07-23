"""``BaseTissueSegmenter`` — abstract base for tissue segmentation models.

Concrete segmenters (e.g. :class:`OtsuTissueSegmenter`) mirror
:class:`coral.features.CoralEncoder`: pure ``preprocess`` + ``forward`` with
a composed ``segment`` driver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class BaseTissueSegmenter(ABC):
    """Abstract base for CORAL tissue-mask segmenters.

    Subclasses declare:

    - ``name`` — registry key (e.g. ``"otsu"``, ``"dl_v1"``).
    - ``target_mag`` — magnification level the model expects, e.g.
      ``1.25`` for low-res Otsu, ``10`` for DL nets.
    - ``uses_structural`` — whether the segmenter thresholds on
      structural channels in addition to the nuclear stain. Otsu does
      (``True``); nuclear-only models (e.g. CARTA) leave it ``False``.

    Subclasses implement:

    - :meth:`required_channels` — which markers (or aliases) the
      segmenter needs. ``None`` means "decide via smart channel
      inference at call time".
    - :meth:`preprocess` — per-method preprocessing.
    - :meth:`forward` — segmentation forward pass returning a mask.

    :meth:`segment` composes the two; that's the standard call.

    Example:
        Call a concrete segmenter through the slide driver::

            from coral import CoralSlide
            from coral.tissue import OtsuTissueSegmenter

            slide = CoralSlide.open("slide.zarr")
            mask = slide.detect_tissue(OtsuTissueSegmenter())
    """

    name: ClassVar[str] = "base"
    target_mag: ClassVar[float] = 1.0
    #: Whether the slide driver loads structural channels (beyond the
    #: nuclear stain) for this segmenter — resolving + logging them and
    #: saving their max-projection review image. Otsu unions a structural
    #: max-projection into its threshold, so it sets this ``True``; a
    #: nuclear-only segmenter (e.g. CARTA) leaves it ``False`` and the
    #: driver never touches structural markers.
    uses_structural: ClassVar[bool] = False

    @property
    def params(self) -> dict[str, Any]:
        """The segmenter's tuning parameters, as a JSON-safe dict.

        Recorded verbatim into ``tissue/config.json`` for provenance, so a
        run is reproducible and a future segmenter's parameters slot in
        without changing the writer. The base returns ``{}``; a segmenter
        with knobs overrides this to expose them.

        Example:
            ``OtsuTissueSegmenter(max_bridge_distance=150).params``
            returns ``{"nuclear_threshold_factor": 0.7,
            "min_object_area_um2": 68.0, "max_bridge_distance": 150.0}``.
        """
        return {}

    @abstractmethod
    def required_channels(self) -> list[str] | None:
        """Return the channel names this segmenter needs, or ``None``.

        ``None`` means "infer at call time" — the storage adapter
        will use the slide's marker metadata + CORAL's marker
        registry to find a suitable channel (e.g. DAPI / Hoechst).

        Returns:
            Required marker names, or ``None`` to infer at call time.

        Example:
            >>> from coral.tissue import OtsuTissueSegmenter
            >>> OtsuTissueSegmenter().required_channels() is None
            True
        """

    @abstractmethod
    def preprocess(
        self,
        image: Any,  # noqa: ANN401 — typed by concrete segmenters
    ) -> Any:  # noqa: ANN401
        """Per-method preprocessing. Image in, image out.

        Args:
            image: Input image in the segmenter's expected layout.

        Returns:
            Preprocessed image ready for :meth:`forward`.

        Example:
            Concrete segmenters implement this; prefer :meth:`segment`.
        """

    @abstractmethod
    def forward(
        self,
        image: Any,  # noqa: ANN401 — typed by concrete segmenters
    ) -> Any:  # noqa: ANN401
        """Return a binary tissue mask for the input image.

        Args:
            image: Preprocessed image from :meth:`preprocess`.

        Returns:
            Binary tissue mask.

        Example:
            Concrete segmenters implement this; prefer :meth:`segment`.
        """

    def segment(
        self,
        image: Any,  # noqa: ANN401 — typed by concrete segmenters
    ) -> Any:  # noqa: ANN401
        """Run :meth:`preprocess` then :meth:`forward`.

        Args:
            image: Input image (numpy / torch / xarray) of the
                expected magnification + channel layout.

        Returns:
            Binary tissue mask.

        Example:
            Concrete segmenters may take extra kwargs (e.g. ``mpp``)::

                from coral.tissue import OtsuTissueSegmenter
                import numpy as np

                img = np.zeros((1, 16, 16), dtype="uint8")
                mask = OtsuTissueSegmenter().segment(img, mpp=1.0)
        """
        return self.forward(self.preprocess(image))
