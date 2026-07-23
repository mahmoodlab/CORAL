"""Alpha-shape tissue region — wrap detected islands in a boundary.

Otsu thresholding of punctate mIF markers (nuclei, membrane, ECM) yields
many small tissue *islands*. Downstream analysis wants the **tissue
region** those islands belong to — a single boundary enclosing them, plus
the stroma/space between, the way CLAM/TRIDENT produce a tissue contour
for H&E.

The region is an **alpha shape** (a generalized, concave hull) of the
island pixels: a Delaunay triangulation with the large triangles that
bridge genuine background gaps removed. One physical knob controls it —
``max_bridge_um``, the widest gap the boundary spans before it concaves
inward. Large → the convex hull (very encompassing); small → the boundary
hugs each tissue piece. It is resolution-independent (a distance in
microns), so the result no longer depends on the slide's pixel size.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import Delaunay
from skimage.transform import resize

__all__ = ["alpha_shape_region"]

# The triangulation runs on a grid no larger than this on its longest side
# — the tissue boundary is coarse, so a smaller grid is faster and looks
# identical once upsampled back to level 0.
_WORKING_EDGE = 512


def _small_triangles(
    tri: Delaunay, points: np.ndarray, max_circumradius: float
) -> np.ndarray:
    """Boolean per-triangle mask: circumradius below ``max_circumradius``.

    A triangle's circumradius grows with the gap it spans, so dropping the
    large ones removes the bridges across background — the essence of the
    alpha shape.
    """
    a, b, c = (points[tri.simplices[:, k]] for k in range(3))
    ab = np.hypot(*(a - b).T)
    bc = np.hypot(*(b - c).T)
    ca = np.hypot(*(c - a).T)
    area = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )
    circ = np.where(area > 0, (ab * bc * ca) / (4 * area + 1e-9), np.inf)
    return circ < max_circumradius


def alpha_shape_region(
    islands: np.ndarray, mpp: float, *, max_bridge_um: float
) -> np.ndarray:
    """Enclose island pixels in an alpha-shape tissue region.

    Args:
        islands: Boolean ``(y, x)`` mask of detected tissue islands, at
            level 0.
        mpp: Level-0 microns-per-pixel (used to convert ``max_bridge_um``
            to pixels — the result is thus resolution-independent).
        max_bridge_um: Widest background gap the boundary bridges before
            it concaves inward. Larger → more encompassing (→ the convex
            hull); smaller → hugs each tissue piece.

    Returns:
        Boolean ``(y, x)`` tissue-region mask at level 0. Returns
        ``islands`` unchanged when there is too little tissue to enclose.

    Example:
        >>> import numpy as np
        >>> m = np.zeros((60, 60), dtype=bool)
        >>> m[10:20, 10:20] = True  # two separate islands
        >>> m[40:50, 40:50] = True
        >>> region = alpha_shape_region(m, mpp=1.0, max_bridge_um=100)
        >>> bool(region[30, 30])  # the gap between them is now enclosed
        True
    """
    islands = np.asarray(islands, dtype=bool)
    height, width = islands.shape
    factor = max(1, round(max(height, width) / _WORKING_EDGE))
    small = islands[::factor, ::factor]
    points = np.column_stack(np.nonzero(small)).astype(float)
    if len(points) < 4:
        return islands
    tri = Delaunay(points)
    r_cut = (max_bridge_um / (mpp * factor)) / 2.0
    keep = _small_triangles(tri, points, r_cut)

    grid = np.indices(small.shape).reshape(2, -1).T
    simplex = tri.find_simplex(grid)
    inside = (simplex >= 0) & keep[np.clip(simplex, 0, None)]
    region = np.asarray(
        ndi.binary_fill_holes(inside.reshape(small.shape)), dtype=bool
    )
    if factor == 1:
        return region
    return resize(
        region, (height, width), order=0, preserve_range=True
    ).astype(bool)
