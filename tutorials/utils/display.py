"""Compact inline previews of the review figures CORAL writes to a store.

The overlays under a store (``tissue_overlay.png``, ``patch_overlay.png``)
are saved at print resolution, so handing one straight to ``display()``
embeds several megabytes of lossless PNG in the notebook. These helpers
downscale to screen size and re-encode as JPEG first — the same trade the
notebooks already make for matplotlib figures via
``%config InlineBackend.figure_formats = ["jpeg"]``.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from IPython.display import Image as InlineImage
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def show_overlay(
    path: str | Path, max_width: int = 1200, quality: int = 85
) -> InlineImage:
    """Load a review figure and return a screen-sized JPEG preview.

    The image is scaled down to ``max_width`` (never up — a figure already
    narrower than that is left alone) and re-encoded as JPEG.

    Args:
        path: Path to the figure, e.g. a store's ``tissue_overlay.png``.
        max_width: Width in pixels above which the preview is downscaled.
        quality: JPEG quality, ``1``-``95``.

    Returns:
        An :class:`IPython.display.Image` holding the encoded JPEG, ready
        to be displayed as a cell's final expression.

    Example:
        Preview the overlay ``coral tissue`` wrote for a slide::

            tissue = slide.path / "tissue" / "tissue_otsu"
            show_overlay(tissue / "tissue_overlay.png")
    """
    with Image.open(path) as figure:
        preview = figure.convert("RGB")
    # Bounding the height by its own value leaves width the only constraint.
    preview.thumbnail((max_width, preview.height), Image.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, "JPEG", quality=quality, optimize=True)
    return InlineImage(data=buffer.getvalue(), format="jpeg")
