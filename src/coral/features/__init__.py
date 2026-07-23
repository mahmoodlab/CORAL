"""Feature encoders — the ``CoralEncoder`` scaffold + global registry.

Concrete encoders register themselves via the :func:`register` decorator
and fill the four ``CoralEncoder`` slots (see :mod:`coral.features.base`).

Example:
    >>> import numpy as np
    >>> from coral.features import CoralEncoder, register
    >>> @register("demo")
    ... class DemoEncoder(CoralEncoder):
    ...     name = "demo"
    ...
    ...     def required_markers(self):
    ...         return None
    ...
    ...     def forward(self, x, markers, marker_emb):
    ...         return np.asarray(x)
    >>> "demo" in EXTRACTOR_REGISTRY
    True
    >>> del EXTRACTOR_REGISTRY["demo"]  # cleanup for doctest determinism
"""

from __future__ import annotations

from coral.features.base import CoralEncoder
from coral.utils.registry import make_register_decorator

EXTRACTOR_REGISTRY: dict[str, type[CoralEncoder]] = {}

register = make_register_decorator(
    EXTRACTOR_REGISTRY,
    kind="extractor",
    base_class=CoralEncoder,
)

# Import concrete encoders for their @register side effect (after
# ``register`` is defined to avoid a circular import).
from coral.features.camae import CAMAEExtractor  # noqa: E402
from coral.features.dinov2 import DINOv2Extractor  # noqa: E402
from coral.features.eva import EvaExtractor  # noqa: E402
from coral.features.kronos1 import Kronos1Extractor  # noqa: E402
from coral.features.kronos2 import Kronos2Extractor  # noqa: E402
from coral.features.mean_marker import MeanMarkerExtractor  # noqa: E402
from coral.features.uni import UNIExtractor  # noqa: E402
from coral.features.uni_post import UNIPostExtractor  # noqa: E402


def listed_extractors() -> list[str]:
    """Extractor names suggested when an unknown one is requested.

    Every registered name resolves when passed to ``--extractor`` and
    ``--help`` lists them all; this returns the same set, sorted, for
    "did you mean" suggestions.

    Returns:
        Sorted extractor names, suitable for a "did you mean" list.

    Example:
        >>> names = listed_extractors()
        >>> "mean_marker" in names
        True
    """
    return sorted(EXTRACTOR_REGISTRY)


def extractor_cli_help(*, default: str = "mean_marker") -> str:
    """``--extractor`` help text listing every registered name.

    Args:
        default: Default extractor name shown in the help string.

    Returns:
        Help text for Typer's ``--extractor`` option.

    Example:
        >>> from coral.features import EXTRACTOR_REGISTRY, extractor_cli_help
        >>> help_text = extractor_cli_help()
        >>> "mean_marker" in help_text and "KRONOS2" in help_text
        True
    """
    names = ", ".join(f"'{n}'" for n in sorted(EXTRACTOR_REGISTRY))
    return (
        "Feature extractor to run, by registered name "
        f"(default {default}). Built-ins: {names}. "
        "mean_marker returns per-marker mean intensities; "
        "foundation-model encoders produce CLS embeddings."
    )


__all__ = [
    "EXTRACTOR_REGISTRY",
    "CAMAEExtractor",
    "DINOv2Extractor",
    "CoralEncoder",
    "EvaExtractor",
    "Kronos1Extractor",
    "Kronos2Extractor",
    "MeanMarkerExtractor",
    "UNIExtractor",
    "UNIPostExtractor",
    "extractor_cli_help",
    "listed_extractors",
    "register",
]
