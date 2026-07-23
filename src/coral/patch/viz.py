"""Patch-grid overlay — the first-class patching visualization.

A two-panel review figure (like the tissue overlay): the nuclear
channel on the left, the patch grid over the same backdrop on the
right. Every patch is drawn — patching keeps the whole grid — with the
tissue boundary contoured and each box labelled with its tissue
fraction, so how much tissue each patch actually covers reads at a
glance. A crisp header states what was extracted: how many patches, how
many touch tissue, the patch size in pixels and microns, and the
resolution (µm/px) (inspired by TRIDENT's patch-viz annotations). All
styling is parameterized so the design can keep iterating without
touching the patching path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coral.viz import add_scalebar, normalize_uint8

__all__ = ["render_cell_patch_overlay", "render_patch_overlay"]


def _boxes(coords: np.ndarray, side: float) -> list:
    """Build display-space ``Rectangle`` patches for ``coords``."""
    from matplotlib.patches import Rectangle

    return [
        Rectangle((x, y), side, side)
        for x, y in np.asarray(coords, dtype=float)
    ]


def _pad_window(arr: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """``(size, size)`` window of ``arr`` at top-left ``(x, y)``, 0-padded.

    Coords are in-bounds by construction, so the pad is only a safety net
    for a box that grazes the image edge.
    """
    win = np.asarray(arr[y : y + size, x : x + size])
    if win.shape == (size, size):
        return win
    out = np.zeros((size, size), dtype=win.dtype)
    out[: win.shape[0], : win.shape[1]] = win
    return out


def render_patch_overlay(
    nuclear: np.ndarray,
    coords: np.ndarray,
    size: int,
    *,
    mpp: float,
    save_to: str | Path,
    tissue_prop: np.ndarray | None = None,
    tissue_mask: np.ndarray | None = None,
    effective_overlap: float = 0.0,
    left_title: str = "Nuclear channel",
    max_display: int = 1400,
    patch_color: str = "lime",
    contour_color: str = "yellow",
    label_color: str = "deepskyblue",
    line_width: float = 0.5,
    fill_alpha: float = 0.15,
) -> None:
    """Render + save the two-panel patch-grid review figure.

    Left panel: the nuclear backdrop (titled ``left_title``). Right
    panel: the same backdrop with every patch filled ``patch_color``
    and, when ``tissue_mask`` is provided, the tissue boundary drawn in
    ``contour_color``. Both carry a true level-0 scalebar. A header
    states the patch count (and how many touch tissue), a labelled
    parameter line (patch size in px and µm, resolution), and the
    Patches/Tissue legend between that title band and the panel headings
    (kept off the image so the overlay stays readable).

    Args:
        nuclear: 2-D backdrop channel at level 0.
        coords: ``(N, 2)`` int ``(x, y)`` top-left coords of the patches,
            in level-0 pixels. Patching keeps the whole grid, so this is
            every patch — including those with no tissue under them.
        size: Patch side in level-0 pixels.
        mpp: Microns-per-pixel of ``nuclear`` (the level-0 scale).
        save_to: Destination PNG path.
        tissue_prop: Optional ``(N,)`` tissue fractions aligned to
            ``coords``; when given, each patch is labelled with its
            ``%`` in ``label_color`` and the header reports how many
            patches touch tissue.
        tissue_mask: Optional boolean ``(y, x)`` tissue mask at level 0.
            When given, the tissue boundary is drawn as a contour in
            ``contour_color`` so the detected tissue region is explicit.
        effective_overlap: Fractional overlap between adjacent patches
            (0.0–1.0). Per-patch ``%`` labels are suppressed when this
            exceeds 0.5 — the patches are too dense for labels to be
            legible.
        left_title: Title for the left (backdrop) panel.
        max_display: Longest side of each rendered panel, in pixels.
        patch_color: Fill + edge colour of the patch boxes.
        contour_color: Colour of the tissue-boundary contour line.
        label_color: Colour of the per-patch ``%`` labels.
        line_width: Patch-box edge width.
        fill_alpha: Opacity of the kept-patch fill in ``[0, 1]``.

    Example:
        >>> import tempfile
        >>> import numpy as np
        >>> from pathlib import Path
        >>> nuclear = np.zeros((64, 64), dtype="uint8")
        >>> coords = np.array([[0, 0], [32, 0], [0, 32], [32, 32]])
        >>> prop = np.array([0.9, 0.0, 0.4, 0.7])
        >>> mask = np.zeros((64, 64), dtype=bool)
        >>> mask[8:56, 8:56] = True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     out = Path(d) / "patches.png"
        ...     render_patch_overlay(
        ...         nuclear,
        ...         coords,
        ...         32,
        ...         mpp=0.5,
        ...         save_to=out,
        ...         tissue_prop=prop,
        ...         tissue_mask=mask,
        ...     )
        ...     out.exists()
        True
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as patheffects
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.gridspec import GridSpec

    factor = max(1, int(np.ceil(max(nuclear.shape) / max_display)))
    gray = normalize_uint8(nuclear[::factor, ::factor])
    disp_h, disp_w = gray.shape
    side = size / factor
    boxes = np.asarray(coords).reshape(-1, 2)
    prop = None if tissue_prop is None else np.asarray(tissue_prop)

    # Compact header band (title / patch-size / legend) above the two
    # image panels — legend stays off the overlay; images keep ~82% height.
    fig = plt.figure(figsize=(12, 6.8))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.18, 0.82],
        hspace=0.18,
        wspace=0.10,
        left=0.04,
        right=0.96,
        top=0.98,
        bottom=0.03,
    )
    ax_header = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_header.set_axis_off()

    for ax in (ax_left, ax_right):
        ax.imshow(gray, cmap="gray", interpolation="nearest")
        ax.set_axis_off()
        add_scalebar(ax, disp_w, disp_h, mpp * factor)
    ax_left.set_title(left_title, fontsize=12)

    ax_right.add_collection(
        PatchCollection(
            _boxes(boxes / factor, side),
            facecolor=(*mcolors.to_rgba(patch_color)[:3], fill_alpha),
            edgecolor=patch_color,
            linewidths=line_width,
        )
    )
    if tissue_mask is not None:
        mask_disp = np.asarray(tissue_mask, dtype=float)[::factor, ::factor]
        ax_right.contour(
            mask_disp, levels=[0.5], colors=[contour_color], linewidths=1.2
        )
        # contour can auto-expand axis limits — lock back to the imshow extent
        ax_right.set_xlim(ax_left.get_xlim())
        ax_right.set_ylim(ax_left.get_ylim())

    if prop is not None and len(boxes) and effective_overlap < 0.5:
        font = float(np.clip(side * 0.09, 4.0, 8.0))
        stroke = [patheffects.withStroke(linewidth=0.7, foreground="black")]
        labels = zip(boxes / factor, prop, strict=False)
        for (px, py), frac in labels:
            ax_right.text(
                px + side - 1,
                py + side - 1,
                f"{frac * 100:.0f}%",
                color=label_color,
                fontsize=font,
                ha="right",
                va="bottom",
                path_effects=stroke,
            )
    ax_right.set_title("Patches overlay", fontsize=12)

    legend = [
        mpatches.Patch(facecolor=patch_color, alpha=0.6, label="Patches")
    ]
    if tissue_mask is not None:
        legend.append(
            mpatches.Patch(
                facecolor="none", edgecolor=contour_color, label="Tissue"
            )
        )

    head = f"Patches: {len(boxes):,} (all kept)"
    if prop is not None:
        head += f", {int((prop > 0).sum()):,} touching tissue"
    geometry = f"Patch size: {size} px ({size * mpp:.0f} µm) at {mpp:g} µm/px"
    overlap_px = size - int(round(size * (1.0 - effective_overlap)))
    tissue = "Tissue coverage labelled per patch; filter downstream"
    # Patch-size line gets its own row; overlap (when non-zero) and the
    # tissue note on the next, so the header never reads as one cramped
    # strip.
    param_lines = [geometry]
    if effective_overlap > 0:
        overlap_str = f"Overlap: {overlap_px} px ({effective_overlap:g})"
        param_lines.append(f"{overlap_str}        {tissue}")
    else:
        param_lines.append(tissue)
    ax_header.text(
        0.5,
        0.92,
        head,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        transform=ax_header.transAxes,
    )
    ax_header.text(
        0.5,
        0.52,
        "\n".join(param_lines),
        ha="center",
        va="center",
        fontsize=11,
        color="0.35",
        linespacing=1.25,
        transform=ax_header.transAxes,
    )
    ax_header.legend(
        handles=legend,
        loc="lower center",
        ncol=max(1, len(legend)),
        fontsize=10,
        frameon=False,
        handlelength=1.8,
        columnspacing=2.0,
        borderpad=0.2,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.savefig(save_to, dpi=300)
    plt.close(fig)


def render_cell_patch_overlay(
    nuclear: np.ndarray,
    coords: np.ndarray,
    patch_size: int,
    *,
    mpp: float,
    save_to: str | Path,
    tissue_mask: np.ndarray | None = None,
    cell_ids: np.ndarray | None = None,
    cell_mask: np.ndarray | None = None,
    left_title: str = "Cell coverage",
    max_display: int = 1400,
    n_montage: int = 12,
    kept_color: str = "lime",
    contour_color: str = "yellow",
    sample_color: str = "red",
) -> None:
    """Render + save the cell-centered patch review figure.

    Two views answer "did cell patching work" at single-cell density
    (where drawing every patch box is illegible):

    - **Coverage map** (left): every cell over the nuclear backdrop —
      cell centroids as ``kept_color`` dots and the tissue boundary in
      ``contour_color`` — so coverage reads at a glance. A handful of
      patches are outlined and **numbered** to show where the montage
      crops sit. (Keep-edge: no cell is dropped at the border.)
    - **Patch montage** (right): those numbered ``patch_size`` crops —
      the actual pixels a cell encoder sees. When ``cell_mask`` +
      ``cell_ids`` are given, each crop is **isolated to its target
      cell** (neighbours + background blacked out, like the extraction
      masking op) with the cell **boundary** drawn and its centroid
      dotted — so centring, size, and isolation are all verifiable.
      Without a mask, the raw crop is shown with the patch outlined.

    Args:
        nuclear: 2-D backdrop channel at level 0 (map + montage crops).
        coords: ``(N, 2)`` int ``(x, y)`` top-left coords of the kept
            cell patches, in level-0 pixels.
        patch_size: Patch side in level-0 pixels.
        mpp: Microns-per-pixel of ``nuclear`` (the level-0 scale).
        save_to: Destination PNG path.
        tissue_mask: Optional boolean ``(y, x)`` tissue mask at level 0;
            its boundary is drawn in ``contour_color``.
        cell_ids: Optional ``(N,)`` cell ids aligned to ``coords`` — the
            label of each patch's target cell in ``cell_mask``.
        cell_mask: Optional ``(y, x)`` instance-label mask (level 0;
            ``cells/cell_mask``). With ``cell_ids``, each montage crop is
            masked to ``cell_mask == cell_id`` (target cell only) and the
            cell boundary is drawn.
        left_title: Title for the coverage panel.
        max_display: Longest side of the coverage panel, in pixels.
        n_montage: Max number of sample patches in the montage.
        kept_color: Colour of cell centroid dots.
        contour_color: Colour of the tissue-boundary contour.
        sample_color: Colour of the montage-sample outlines + numbers.

    Example:
        >>> import tempfile
        >>> import numpy as np
        >>> from pathlib import Path
        >>> nuclear = np.zeros((128, 128), dtype="uint8")
        >>> coords = np.array([[8, 8], [40, 40], [80, 80]])
        >>> with tempfile.TemporaryDirectory() as d:
        ...     out = Path(d) / "cells.png"
        ...     render_cell_patch_overlay(
        ...         nuclear,
        ...         coords,
        ...         16,
        ...         mpp=0.5,
        ...         save_to=out,
        ...     )
        ...     out.exists()
        True
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    ps = int(patch_size)
    half = ps / 2.0
    kept = np.asarray(coords).reshape(-1, 2)
    n_kept = len(kept)

    # Evenly spaced montage samples through the kept patches.
    n_tiles = min(n_montage, n_kept)
    sample_idx = (
        np.unique(np.linspace(0, n_kept - 1, n_tiles).astype(int))
        if n_tiles
        else np.empty(0, dtype=int)
    )
    mont_cols = 4
    mont_rows = max(1, int(np.ceil(n_montage / mont_cols)))

    fig = plt.figure(figsize=(14, 7))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax_cov = fig.add_subplot(outer[0, 0])

    # ── coverage map ──
    factor = max(1, int(np.ceil(max(nuclear.shape) / max_display)))
    gray = normalize_uint8(nuclear[::factor, ::factor])
    disp_h, disp_w = gray.shape
    ax_cov.imshow(gray, cmap="gray", interpolation="nearest")
    ax_cov.set_axis_off()
    add_scalebar(ax_cov, disp_w, disp_h, mpp * factor)
    ax_cov.set_title(left_title, fontsize=12)
    if tissue_mask is not None:
        md = np.asarray(tissue_mask, dtype=float)[::factor, ::factor]
        ax_cov.contour(
            md, levels=[0.5], colors=[contour_color], linewidths=1.0
        )
        ax_cov.set_xlim(0, disp_w)
        ax_cov.set_ylim(disp_h, 0)
    if n_kept:
        centres = (kept + half) / factor
        ax_cov.scatter(
            centres[:, 0], centres[:, 1], s=2, c=kept_color, marker="."
        )
    side = ps / factor
    for label, i in enumerate(sample_idx, 1):
        bx, by = kept[i] / factor
        ax_cov.add_patch(
            mpatches.Rectangle(
                (bx, by),
                side,
                side,
                fill=False,
                edgecolor=sample_color,
                linewidth=0.8,
            )
        )
        ax_cov.text(
            bx,
            by - 1,
            str(label),
            color=sample_color,
            fontsize=10,
            ha="left",
            va="bottom",
            fontweight="bold",
        )

    legend = [mpatches.Patch(facecolor=kept_color, label=f"cells ({n_kept})")]
    if tissue_mask is not None:
        legend.append(
            mpatches.Patch(
                facecolor="none", edgecolor=contour_color, label="tissue"
            )
        )
    ax_cov.legend(
        handles=legend, loc="lower right", fontsize=8, framealpha=0.6
    )

    # ── patch montage ──
    inner = outer[0, 1].subgridspec(
        mont_rows, mont_cols, hspace=0.3, wspace=0.05
    )
    ids = None if cell_ids is None else np.asarray(cell_ids)
    for slot in range(mont_rows * mont_cols):
        ax = fig.add_subplot(inner[slot // mont_cols, slot % mont_cols])
        ax.set_axis_off()
        if slot >= len(sample_idx):
            continue
        idx = int(sample_idx[slot])
        x, y = (int(v) for v in kept[idx])
        disp = normalize_uint8(_pad_window(nuclear, x, y, ps))
        if cell_mask is not None and ids is not None:
            # isolate the target cell: black out neighbours + background,
            # then outline the cell (boundary of its footprint).
            foot = _pad_window(cell_mask, x, y, ps) == int(ids[idx])
            disp = disp * foot
            ax.imshow(disp, cmap="gray", interpolation="nearest")
            ax.contour(
                foot.astype(float),
                levels=[0.5],
                colors=[sample_color],
                linewidths=0.8,
            )
        else:  # no mask available — outline the whole patch box.
            ax.imshow(disp, cmap="gray", interpolation="nearest")
            ax.add_patch(
                mpatches.Rectangle(
                    (0, 0),
                    ps - 1,
                    ps - 1,
                    fill=False,
                    edgecolor=sample_color,
                    linewidth=1.0,
                )
            )
        ax.plot(half - 0.5, half - 0.5, marker=".", c=kept_color, ms=3)
        ax.set_title(str(slot + 1), fontsize=10, color=sample_color, pad=1)

    head = f"Cell patches: {n_kept:,} cells (keep-edge)"
    geometry = (
        f"Patch size: {ps} px ({ps * mpp:.0f} µm) at {mpp:g} µm/px"
        f"   ·   montage: {len(sample_idx)} sampled"
    )
    fig.suptitle(head, fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.94, geometry, ha="center", fontsize=10, color="0.35")
    fig.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.close(fig)
