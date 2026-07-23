"""``CoralEncoder`` — the four-slot patch-encoder scaffold.

Every CORAL encoder folds into ``CoralEncoder`` by filling four atomic slots
(``from_pretrained`` / ``embed_markers`` / ``transform`` / ``forward``); see
the class docstring for the per-encoder preprocessing contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class CoralEncoder(ABC):
    """Patch encoder scaffold — four atomic slots.

    The rock-solid contract every CORAL encoder folds into. A multiplex
    image has one channel per marker, so preprocessing splits cleanly: the
    **patcher** (:class:`~coral.features.dataset.CoralDataset`) hands out
    dtype-scaled ``float32`` patches (data-driven), and the encoder fills
    four slots that map 1-to-1 to its preprocessing recipe:

    - :meth:`from_pretrained` — load the model (+ embedding machinery).
    - :meth:`embed_markers` — marker **names/sequences** → an embedding,
      once per panel (``None`` for marker-agnostic encoders).
    - :meth:`transform` — image **pixels** → model input, every patch.
    - :meth:`forward` — the model call → ``(B, embed_dim)``.

    ``embed_markers`` (names, once) and ``transform`` (pixels, every patch)
    are never combined. :meth:`encode` composes the three and is the
    standalone entry point — an encoder is a real object you call directly.

    Class-level declarations (the encoder's matrix row): ``name`` (registry
    key), ``embed_dim`` (CLS width), ``precision`` (compute dtype for
    autocast, a **string** so this module stays torch-free), ``scale`` (does
    the patcher pre-scale to ``float32 [0, 1]``? ``False`` for mean_marker,
    which scales raw pixels in its own ``float64`` reduction), and
    ``output_schema``.

    Note: ``precision`` is the model's compute dtype; it is **independent**
    of the patch ``scale`` (the dtype-scaling divisor is set by the image
    data dtype, not the encoder).

    Example:
        >>> import numpy as np
        >>> class CopyEncoder(CoralEncoder):
        ...     name = "copy"
        ...     embed_dim = 4
        ...
        ...     def required_markers(self):
        ...         return None
        ...
        ...     def forward(self, x, markers, marker_emb):
        ...         return np.asarray(x).reshape(len(x), -1)
        >>> enc = CopyEncoder()
        >>> enc.embed_markers(["CD3"]) is None  # default: no embedding
        True
        >>> enc.encode(np.ones((2, 1, 2, 2)), ["CD3"]).shape
        (2, 4)
    """

    name: ClassVar[str] = "base"
    version: ClassVar[str] = "1"  # bumped when an encoder's output changes
    embed_dim: ClassVar[int] = 0
    # Compute precision for autocast; a string keeps this module torch-free
    # (``encode_features`` maps it to a ``torch.dtype``).
    precision: ClassVar[str] = "float32"
    # Does the patcher pre-scale patches to float32 [0, 1]? mean_marker sets
    # False (it divides raw pixels in its own float64 reduction).
    scale: ClassVar[bool] = True
    # Can this encoder z-score novel markers from data-driven stats computed
    # by the prepare pass? Only KRONOS2 sets True; the CLI gates
    # the ``--additional-markers`` prepare phase on it.
    supports_novel_markers: ClassVar[bool] = False
    output_schema: ClassVar[dict[str, tuple[str, ...]]] = {
        "features": ("feature",),
    }

    @abstractmethod
    def required_markers(self) -> list[str] | None:
        """Markers this encoder needs, or ``None`` if marker-agnostic."""

    def embed_markers(self, markers: list[str]) -> Any:  # noqa: ANN401
        """Marker names/sequences → an embedding, once per panel.

        Default ``None`` (marker-agnostic and marker-name-driven encoders;
        KRONOS2 passes names straight to its model). Eva (GenePT)
        overrides this. Runs once per slide, outside the patch loop —
        never combine it with :meth:`transform`.

        Example:
            >>> class E(CoralEncoder):
            ...     def required_markers(self):
            ...         return None
            ...
            ...     def forward(self, x, markers, marker_emb):
            ...         return x
            >>> E().embed_markers(["CD3", "CD8"]) is None
            True
        """
        return None

    def prepare_slide(
        self,
        image: Any,  # noqa: ANN401 — array
        *,
        markers: list[str] | None = None,
        tissue_mask: Any = None,  # noqa: ANN401 — (y, x) bool array
    ) -> None:
        """Receive the slide's level-0 selected channels, once per slide.

        Default **no-op**. ``CoralSlide.encode_features`` calls this once
        before the patch loop with the lazy level-0 image (the selected
        channels, ``(c, y, x)``). An encoder needing **whole-image**
        context — e.g. per-image preprocessing stats — overrides
        it; the patch-level encoders (all the others) ignore it, so it
        never perturbs them.

        The optional ``markers`` (the selected channel names, aligned to
        ``image``) and ``tissue_mask`` (a ``(y, x)`` bool array) are the
        prepare-pass extras: KRONOS2 uses them to compute
        novel-marker stats over the masked region. The extract-time call
        passes neither, so the stats path is a no-op there.

        Example:
            >>> class E(CoralEncoder):
            ...     def required_markers(self):
            ...         return None
            ...
            ...     def forward(self, x, markers, marker_emb):
            ...         return x
            >>> E().prepare_slide(object()) is None
            True
        """
        return None

    def warn_for_patches(self, patch_size: int, mode: str) -> None:
        """Warn if this encoder handles a patch geometry poorly.

        Default **no-op**. ``CoralSlide.encode_features`` calls this once,
        before the patch loop, with the patch set's ``patch_size`` (pixels)
        and ``mode`` (``"grid"`` or ``"cell_centered"``). Fixed-input
        encoders that resize small patches (e.g. Eva upsamples them to 224)
        override it to flag a quality footgun; every other encoder inherits
        the no-op and stays silent.

        Args:
            patch_size: Patch side length in pixels.
            mode: The patch set's mode (``"grid"`` or ``"cell_centered"``).

        Example:
            >>> class E(CoralEncoder):
            ...     def required_markers(self):
            ...         return None
            ...
            ...     def forward(self, x, markers, marker_emb):
            ...         return x
            >>> E().warn_for_patches(64, "cell_centered") is None
            True
        """
        return None

    def marker_provenance(self) -> dict[str, list[str]] | None:
        """Per-marker provenance to persist in feature zattrs, or ``None``.

        Default ``None`` — channel-agnostic encoders write nothing extra.
        Marker-aware encoders that drop or degrade markers (e.g. Eva, which
        drops markers outside its vocabulary and gives GenePT-less markers a
        random embedding) override this to return named marker sets, which
        ``CoralSlide.encode_features`` writes into the feature array's
        ``.zattrs`` as a durable record. Populated as a side effect of
        :meth:`embed_markers`, so it is meaningful only after that has run.

        Example:
            >>> class E(CoralEncoder):
            ...     def required_markers(self):
            ...         return None
            ...
            ...     def forward(self, x, markers, marker_emb):
            ...         return x
            >>> E().marker_provenance() is None
            True
        """
        return None

    def transform(
        self,
        patches: Any,  # noqa: ANN401 — array/tensor batch
        markers: list[str],
        *,
        nuclear_marker: str | None = None,
    ) -> Any:  # noqa: ANN401
        """Image pixels → model input, applied to every patch batch.

        Default identity (the patcher already handed out dtype-scaled
        patches). RGB encoders override with pseudo-RGB + ImageNet; KRONOS2
        with its in-model marker z-score (``nuclear_marker`` is its DAPI
        hint, ignored by everyone else).

        Example:
            >>> class E(CoralEncoder):
            ...     def required_markers(self):
            ...         return None
            ...
            ...     def forward(self, x, markers, marker_emb):
            ...         return x
            >>> p = object()
            >>> E().transform(p, ["CD3"]) is p
            True
        """
        return patches

    @abstractmethod
    def forward(
        self,
        x: Any,  # noqa: ANN401
        markers: list[str],
        marker_emb: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Run the model → ``(B, embed_dim)``.

        Receives both ``markers`` (names) and ``marker_emb`` (the
        :meth:`embed_markers` output) so a model uses whichever it needs.
        """

    def encode(
        self,
        patches: Any,  # noqa: ANN401
        markers: list[str],
        *,
        nuclear_marker: str | None = None,
    ) -> Any:  # noqa: ANN401
        """Compose :meth:`embed_markers` + :meth:`transform` + :meth:`forward`.

        The standalone entry point + interop seam — call an encoder on
        patches + markers without CORAL's slide machinery.

        Example:
            >>> import numpy as np
            >>> class E(CoralEncoder):
            ...     embed_dim = 8
            ...
            ...     def required_markers(self):
            ...         return None
            ...
            ...     def forward(self, x, markers, marker_emb):
            ...         return np.asarray(x).reshape(len(x), -1)
            >>> E().encode(np.ones((2, 2, 2, 2)), ["a", "b"]).shape
            (2, 8)
        """
        marker_emb = self.embed_markers(markers)
        x = self.transform(patches, markers, nuclear_marker=nuclear_marker)
        return self.forward(x, markers, marker_emb)

    @classmethod
    def build(cls, device: str | None = None) -> CoralEncoder:
        """Construct for a run (default no-arg ``cls()``); the CLI hook.

        Model-loading encoders override via :meth:`from_pretrained` so the
        model is loaded once and reused across slides.

        Args:
            device: Accepted and ignored by the base (the CPU-baseline
                path mean_marker inherits); model encoders forward it.

        Example:
            >>> class E(CoralEncoder):
            ...     def required_markers(self):
            ...         return None
            ...
            ...     def forward(self, x, markers, marker_emb):
            ...         return x
            >>> isinstance(E.build(), E)
            True
        """
        return cls()
