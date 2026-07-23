"""Cell-segmentation overlay — the first-class cell-seg visualization.

A two-panel review figure (like the tissue / patch overlays): the
nuclear channel on the left, the cell-instance outlines over the same
backdrop on the right, with a cell count and a level-0 scalebar
(matplotlib + scikit-image boundaries; no OpenCV).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from coral.viz import add_scalebar, normalize_uint8

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["render_cell_inset", "render_cell_overlay"]


def render_cell_overlay(
    nuclear: np.ndarray,
    mask: np.ndarray,
    *,
    mpp: float,
    save_to: str | Path,
    tissue_mask: np.ndarray | None = None,
    n_cells: int | None = None,
    max_display: int = 1600,
    outline_color: str = "cyan",
    tissue_color: str = "yellow",
) -> None:
    """Render + save the two-panel cell-segmentation review figure.

    Left: the nuclear backdrop. Right: cell-instance outlines over the
    backdrop, with an optional tissue contour so the viewer can see which
    cells fall inside vs outside the detected tissue region. Both panels
    carry a level-0 scalebar; the header states the cell count.

    Args:
        nuclear: 2-D nuclear channel at level 0.
        mask: 2-D int instance-label mask (same shape as ``nuclear``).
        mpp: Microns-per-pixel of ``nuclear`` (the level-0 scale).
        save_to: Destination PNG path.
        tissue_mask: Optional 2-D boolean tissue mask (same shape as
            ``nuclear``). When provided, the tissue contour is drawn on
            the right panel underneath the cell outlines.
        n_cells: Cell count for the header; computed from ``mask`` if
            ``None``.
        max_display: Longest side of each rendered panel, in pixels.
        outline_color: Colour of the cell outlines.
        tissue_color: Colour of the tissue contour (used only when
            ``tissue_mask`` is provided).

    Example:
        >>> import tempfile
        >>> import numpy as np
        >>> from pathlib import Path
        >>> nuclear = np.zeros((48, 48), dtype="uint8")
        >>> mask = np.zeros((48, 48), dtype="int32")
        >>> mask[8:16, 8:16] = 1
        >>> mask[24:32, 24:32] = 2
        >>> tissue = np.zeros((48, 48), dtype=bool)
        >>> tissue[4:44, 4:44] = True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     out = Path(d) / "cells.png"
        ...     render_cell_overlay(
        ...         nuclear, mask, mpp=0.5, save_to=out, tissue_mask=tissue
        ...     )
        ...     out.exists()
        True
    """
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    if n_cells is None:
        n_cells = int(np.asarray(mask).max())

    factor = max(1, int(np.ceil(max(nuclear.shape) / max_display)))
    gray = normalize_uint8(nuclear[::factor, ::factor])
    small = np.asarray(mask[::factor, ::factor])
    disp_h, disp_w = gray.shape

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in (ax_left, ax_right):
        ax.imshow(gray, cmap="gray", interpolation="nearest")
        ax.set_axis_off()
        add_scalebar(ax, disp_w, disp_h, mpp * factor)
    ax_left.set_title("Nuclear channel", fontsize=12)

    # Cell outlines via RGBA overlay.
    rgba = np.zeros((*small.shape, 4))
    cell_boundary = find_boundaries(small, mode="inner")
    rgba[cell_boundary] = (*mcolors.to_rgba(outline_color)[:3], 1.0)
    ax_right.imshow(rgba, interpolation="nearest")

    # Tissue contour via matplotlib contour (thicker + smoother than
    # find_boundaries pixel-marking); lock limits so contour can't expand.
    legend_handles = [
        mpatches.Patch(color=outline_color, label="Cell mask"),
    ]
    if tissue_mask is not None:
        t_disp = np.asarray(tissue_mask[::factor, ::factor], dtype=float)
        ax_right.contour(
            t_disp, levels=[0.5], colors=[tissue_color], linewidths=1.5
        )
        ax_right.set_xlim(ax_left.get_xlim())
        ax_right.set_ylim(ax_left.get_ylim())
        legend_handles.append(
            mpatches.Patch(color=tissue_color, label="Tissue contour")
        )

    ax_right.legend(
        handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.6
    )
    ax_right.set_title("Cell outlines + tissue contour", fontsize=12)

    fig.suptitle(f"{n_cells:,} cells", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_cell_inset(
    nuclear: np.ndarray,
    mask: np.ndarray,
    *,
    top_left: tuple[int, int],
    box_size: tuple[int, int],
    mpp: float,
    save_to: str | Path,
    show_cells: bool = True,
    cell_color: str = "cyan",
    labels: pd.DataFrame | None = None,
    label_color: str = "yellow",
    max_display: int = 1400,
) -> None:
    """Render a slide-with-box + zoomed-inset figure for cell inspection.

    Left panel: the whole slide (downsampled nuclear) with the inset box
    outlined. Right panel: the box region at full resolution, optionally
    with cell outlines and per-cell text labels.

    Args:
        nuclear: 2-D nuclear channel at level 0.
        mask: 2-D int instance-label mask (same shape as ``nuclear``).
        top_left: Inset ``(x, y)`` top-left corner, in level-0 pixels.
        box_size: Inset ``(height, width)`` in level-0 pixels.
        mpp: Microns-per-pixel of ``nuclear`` (the level-0 scale).
        save_to: Destination PNG path.
        show_cells: Draw cell outlines in the inset (default ``True``).
        cell_color: Colour of the box + cell outlines.
        labels: Optional ``DataFrame`` with ``cell_id`` + ``label``
            columns; cells in the box are annotated with their label.
        label_color: Colour of the per-cell label text.
        max_display: Longest side of the left (whole-slide) panel, px.

    Raises:
        ValueError: If the box is non-positive or falls outside the
            image bounds.

    Example:
        >>> import tempfile
        >>> import numpy as np
        >>> from pathlib import Path
        >>> nuclear = np.zeros((64, 64), dtype="uint8")
        >>> mask = np.zeros((64, 64), dtype="int32")
        >>> mask[20:28, 20:28] = 1
        >>> with tempfile.TemporaryDirectory() as d:
        ...     out = Path(d) / "inset.png"
        ...     render_cell_inset(
        ...         nuclear,
        ...         mask,
        ...         top_left=(16, 16),
        ...         box_size=(32, 32),
        ...         mpp=0.5,
        ...         save_to=out,
        ...     )
        ...     out.exists()
        True
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as patheffects
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.segmentation import find_boundaries

    x, y = int(top_left[0]), int(top_left[1])
    box_h, box_w = int(box_size[0]), int(box_size[1])
    height, width = nuclear.shape
    if box_h <= 0 or box_w <= 0:
        msg = f"box size must be positive, got (h={box_h}, w={box_w})"
        raise ValueError(msg)
    if x < 0 or y < 0 or x + box_w > width or y + box_h > height:
        msg = (
            f"inset box (x={x}, y={y}, {box_w}x{box_h}) is out of bounds "
            f"for a {width}x{height} image"
        )
        raise ValueError(msg)

    fig, (ax_full, ax_inset) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: whole slide downsampled, with the inset box.
    factor = max(1, int(np.ceil(max(height, width) / max_display)))
    gray = normalize_uint8(nuclear[::factor, ::factor])
    ax_full.imshow(gray, cmap="gray", interpolation="nearest")
    ax_full.add_patch(
        Rectangle(
            (x / factor, y / factor),
            box_w / factor,
            box_h / factor,
            fill=False,
            edgecolor=cell_color,
            linewidth=1.5,
        )
    )
    ax_full.set_axis_off()
    add_scalebar(ax_full, gray.shape[1], gray.shape[0], mpp * factor)
    ax_full.set_title("Whole slide", fontsize=12)

    # Right: the box at full resolution.
    crop = nuclear[y : y + box_h, x : x + box_w]
    ax_inset.imshow(
        normalize_uint8(crop), cmap="gray", interpolation="nearest"
    )
    ax_inset.set_axis_off()
    add_scalebar(ax_inset, box_w, box_h, mpp)
    ax_inset.set_title(f"Inset {box_w}x{box_h} px @ ({x}, {y})", fontsize=12)

    if show_cells:
        mask_crop = np.asarray(mask[y : y + box_h, x : x + box_w])
        outlines = find_boundaries(mask_crop, mode="inner")
        rgba = np.zeros((*mask_crop.shape, 4))
        rgba[outlines] = (*mcolors.to_rgba(cell_color)[:3], 1.0)
        ax_inset.imshow(rgba, interpolation="nearest")
        if labels is not None:
            lut = dict(zip(labels["cell_id"], labels["label"], strict=False))
            stroke = [
                patheffects.withStroke(linewidth=0.7, foreground="black")
            ]
            for cell_id in np.unique(mask_crop):
                if cell_id == 0 or cell_id not in lut:
                    continue
                ys, xs = np.where(mask_crop == cell_id)
                ax_inset.text(
                    xs.mean(),
                    ys.mean(),
                    str(lut[cell_id]),
                    color=label_color,
                    fontsize=6,
                    ha="center",
                    va="center",
                    path_effects=stroke,
                )

    fig.tight_layout()
    fig.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.close(fig)
