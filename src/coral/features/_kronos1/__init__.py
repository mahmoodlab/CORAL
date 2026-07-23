"""Vendored KRONOS1 model (github.com/MahmoodLab/KRONOS).

A verbatim copy of the ``kronos.model`` subpackage — the published KRONOS
ViT (``inference``, ``vision_transformer``, ``block``, ``attention``,
``pos_embed``, ``patch_embed``, ``swiglu_ffn``, ``dino_head``, ``mlp``,
``layer_scale``, ``drop_path``) — so CORAL no longer depends on the external
``kronos`` package (not on PyPI; needed a source install). The model code
is DINOv2-derived and carries Meta's Apache-2.0 headers; see ``LICENSE`` +
``NOTICE`` for attribution.

Only the package ``__init__`` differs from upstream: the original
``kronos/__init__.py`` eagerly imported an unused image stack
(``KRONOSImage``, ``ImagePatcher``, ``load_qc_model``); this one exposes
just the model loaders. The 11 module files are byte-for-byte copies.

This subpackage imports ``torch`` (and ``xformers`` when available) at
module load, so it is the opt-in ``kronos1`` extra;
:class:`coral.features.kronos1.Kronos1Extractor` imports it lazily and it is
excluded from CORAL's lint/type/doctest passes.
"""

from coral.features._kronos1.inference import (
    create_model,
    create_model_from_pretrained,
)

__all__ = ["create_model", "create_model_from_pretrained"]
