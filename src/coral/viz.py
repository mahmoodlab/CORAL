"""Shared figure helpers for CORAL overlays (tissue, patches).

Small matplotlib/numpy utilities used by more than one overlay
renderer (``coral.tissue.mask`` and ``coral.patch.viz``): percentile
contrast stretch to ``uint8`` and a level-0 scalebar. Kept here so the
two renderers share one implementation rather than each carrying a
private copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["add_scalebar", "normalize_uint8"]


def normalize_uint8(
    channel: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
) -> np.ndarray:
    """Percentile-stretch a 2-D channel to ``uint8`` grayscale.

    Args:
        channel: 2-D intensity array.
        p_low: Lower percentile mapped to 0.
        p_high: Upper percentile mapped to 255.

    Returns:
        ``uint8`` array the same shape as ``channel`` (all-zero when the
        two percentiles coincide, e.g. a flat channel).

    Example:
        >>> import numpy as np
        >>> normalize_uint8(np.arange(4, dtype="uint16").reshape(2, 2)).max()
        np.uint8(255)
    """
    lo = float(np.percentile(channel, p_low))
    hi = float(np.percentile(channel, p_high))
    if hi <= lo:
        return np.zeros(channel.shape, dtype=np.uint8)
    stretched = np.clip((channel.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def _nice_bar_um(target_um: float) -> int:
    """Round a target length to a nice round number of microns.

    Args:
        target_um: Rough desired bar length in microns.

    Returns:
        The nearest "nice" value from a fixed ladder of round lengths.

    Example:
        >>> _nice_bar_um(170)
        200
    """
    candidates = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
    return min(candidates, key=lambda c: abs(c - target_um))


def add_scalebar(
    ax: Axes, width: int, height: int, mpp: float, color: str = "white"
) -> None:
    """Draw a level-0-scaled scalebar in the lower-left of ``ax``.

    Args:
        ax: Matplotlib axes to draw on.
        width: Display width of the panel, in pixels.
        height: Display height of the panel, in pixels.
        mpp: Microns-per-pixel of the displayed image.
        color: Bar + label colour.

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> _fig, ax = plt.subplots()
        >>> add_scalebar(ax, 100, 100, 0.5)
        >>> plt.close(_fig)
    """
    bar_um = _nice_bar_um(width * mpp * 0.18)
    bar_px = bar_um / mpp
    x0, y0 = 0.04 * width, 0.93 * height
    ax.plot(
        [x0, x0 + bar_px], [y0, y0], color=color, lw=3, solid_capstyle="butt"
    )
    ax.text(
        x0,
        y0 - 0.015 * height,
        f"{bar_um:g} µm",
        color=color,
        fontsize=8,
        va="bottom",
    )
