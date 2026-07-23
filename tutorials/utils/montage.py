"""False-colour marker montages for the CORAL tutorials.

Blends several single-channel markers into one RGB composite for a quick
visual check of an ingested slide. Each channel is min-max stretched,
tinted by a colour from the palette, and summed — the recipe the KRONOS
tutorials use. Kept out of the notebooks so the montage cells stay
focused on the markers and palette, not the plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from coral import CoralSlide


def get_rgb_image(
    multiplex_image: np.ndarray, colors: np.ndarray
) -> np.ndarray:
    """Blend a stack of single-channel planes into one RGB image.

    Each channel is independently min-max stretched to ``[0, 1]``, weighted
    by its colour, and the weighted channels are summed and clipped to an
    8-bit RGB image.

    Args:
        multiplex_image: Array of shape ``(y, x, n_channels)`` — the
            per-marker planes to blend.
        colors: Array of shape ``(n_channels, 3)`` — the RGB tint for each
            channel, values in ``[0, 255]``.

    Returns:
        An ``(y, x, 3)`` ``uint8`` RGB image.

    Example:
        >>> import numpy as np
        >>> planes = np.zeros((4, 4, 2))
        >>> colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.float32)
        >>> rgb = get_rgb_image(planes, colors)
        >>> rgb.shape, rgb.dtype
        ((4, 4, 3), dtype('uint8'))
    """
    multiplex_image = multiplex_image.astype(np.float32)
    for i in range(multiplex_image.shape[-1]):
        chan = multiplex_image[..., i]
        lo, hi = chan.min(), chan.max()
        if hi > lo:
            multiplex_image[..., i] = (chan - lo) / (hi - lo)
    rgb = np.tensordot(multiplex_image * 2, colors, axes=([2], [0]))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def composite(
    slide: CoralSlide,
    marker_set: list[str],
    colors: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Read a set of markers from a slide and blend them to RGB.

    Reads a strided, downsampled view of each named marker (a whole slide
    is large), stacks them, and blends them with :func:`get_rgb_image`.

    Args:
        slide: An ingested CORAL slide.
        marker_set: Canonical marker names to blend, one per colour.
        colors: Array of shape ``(len(marker_set), 3)`` — the RGB tint for
            each marker.
        factor: Stride for the downsampled view — every ``factor``-th pixel
            is read in ``y`` and ``x``.

    Returns:
        An ``(y, x, 3)`` ``uint8`` RGB composite.

    Example:
        For an ingested slide, blend the nuclear stain and a T-cell marker
        at 1/8 resolution::

            rgb = composite(slide, ["dapi", "cd8"], colors[:2], factor=8)
    """
    planes = [
        np.asarray(slide.image.sel(c=m)[::factor, ::factor], dtype=np.float32)
        for m in marker_set
    ]
    return get_rgb_image(np.stack(planes, axis=-1), colors)
