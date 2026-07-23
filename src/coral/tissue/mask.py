"""Tissue-mask outputs: GeoJSON polygons + a paper-quality overlay.

Two derived products from a boolean tissue mask:

- :func:`mask_to_geopandas` traces the mask boundary into shapely
  polygons (via scikit-image contours — **no OpenCV**) and scales them
  to **level-0** pixel coordinates, so the GeoJSON CORAL writes can be
  loaded straight into QuPath / napari over the full-resolution slide.
- :func:`render_tissue_overlay` renders the **first-class, iterated**
  boundary visualization the user sees — a translucent tissue fill plus
  a coloured boundary line drawn over the nuclear thumbnail (TRIDENT's
  ``overlay_gdf_on_thumbnail`` logic, reimplemented in PIL). Backdrop,
  colour, fill opacity, line weight, and scalebar are knobs so the
  design can iterate without touching the detection path.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import make_valid
from shapely.geometry import Polygon
from skimage import measure

from coral.viz import add_scalebar, normalize_uint8

__all__ = [
    "geojson_geometry_sha256",
    "mask_to_geopandas",
    "read_tissue_geojson_mask",
    "render_tissue_overlay",
    "save_binary_mask_png",
    "save_stretched_png",
    "write_tissue_geojson",
]


def mask_to_geopandas(mask: np.ndarray) -> gpd.GeoDataFrame:
    """Trace a boolean level-0 mask into polygons (level-0 coordinates).

    One row per tissue region — a ``tissue_id`` and a ``geometry``
    polygon in level-0 pixel space. Holes are not modelled separately:
    the alpha-shape tissue region is already hole-free.

    Args:
        mask: Boolean ``(y, x)`` tissue mask at level 0.

    Returns:
        A ``GeoDataFrame`` with ``tissue_id`` + ``geometry`` columns.

    Example:
        >>> import numpy as np
        >>> mask = np.zeros((10, 10), dtype=bool)
        >>> mask[2:8, 2:8] = True
        >>> gdf = mask_to_geopandas(mask)
        >>> len(gdf)
        1
        >>> bool(gdf.geometry.iloc[0].is_valid)
        True
    """
    # Pad a False border so find_contours also traces boundaries that run
    # along the image edge — a frame-filling (all-True) mask otherwise
    # yields no contour — then shift the coords back by the 1px pad.
    padded = np.pad(np.asarray(mask, dtype=bool), 1).astype(float)
    polygons = []
    for contour in measure.find_contours(padded, level=0.5):
        if len(contour) < 4:
            continue
        # find_contours yields (row, col); shapely wants (x, y).
        yx = contour - 1.0
        polygon = Polygon(yx[:, [1, 0]])
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        if not polygon.is_empty:
            polygons.append(polygon)
    return gpd.GeoDataFrame(
        {"tissue_id": list(range(len(polygons)))},
        geometry=polygons,
    )


def render_tissue_overlay(
    nuclear: np.ndarray,
    mask: np.ndarray,
    *,
    mpp: float,
    save_to: str | Path,
    nuclear_name: str | None = None,
    max_display: int = 1400,
    tissue_color: str = "lime",
    fill_alpha: float = 0.1,
    line_width: float = 0.8,
) -> None:
    """Render + save a two-panel tissue review figure.

    Left panel: the percentile-stretched nuclear channel, titled
    ``Nuclear Channel (<name>)``. Right panel: the same backdrop with a
    translucent green tissue fill + a thin boundary, titled ``Nuclear
    Channel (grayscale) with Detected Tissue (green)``. Both carry a
    scalebar in true level-0 microns; the figure is downsampled to
    ~``max_display`` px per side for a light PNG.

    Args:
        nuclear: 2-D backdrop channel at the detection resolution.
        mask: Boolean ``(y, x)`` tissue mask, same shape as ``nuclear``.
        mpp: Microns-per-pixel of ``nuclear`` (the level-0 scale).
        save_to: Destination PNG path.
        nuclear_name: Marker name of the nuclear channel, for the title.
        max_display: Longest side of the rendered panels, in pixels.
        tissue_color: Matplotlib colour of the fill + boundary.
        fill_alpha: Opacity of the translucent tissue fill in ``[0, 1]``.
        line_width: Boundary line width.

    Example:
        >>> import tempfile
        >>> import numpy as np
        >>> from pathlib import Path
        >>> nuclear = np.zeros((40, 40), dtype="uint8")
        >>> mask = np.zeros((40, 40), dtype=bool)
        >>> mask[10:30, 10:30] = True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     out = Path(d) / "ov.png"
        ...     render_tissue_overlay(nuclear, mask, mpp=0.5, save_to=out)
        ...     out.exists()
        True
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    factor = max(1, int(np.ceil(max(nuclear.shape) / max_display)))
    gray = normalize_uint8(nuclear[::factor, ::factor])
    small = np.asarray(mask[::factor, ::factor], dtype=bool)
    disp_mpp = mpp * factor

    disp_h, disp_w = gray.shape
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax in (ax_l, ax_r):
        ax.imshow(gray, cmap="gray", interpolation="nearest")
        ax.set_axis_off()
        add_scalebar(ax, disp_w, disp_h, disp_mpp)
    ax_l.set_title(
        f"Nuclear Channel ({nuclear_name})"
        if nuclear_name
        else "Nuclear Channel"
    )

    fill = np.zeros((*small.shape, 4))
    fill[small] = (*mcolors.to_rgba(tissue_color)[:3], fill_alpha)
    ax_r.imshow(fill, interpolation="nearest")
    ax_r.contour(
        small, levels=[0.5], colors=[tissue_color], linewidths=line_width
    )
    ax_r.set_title("Nuclear Channel (grayscale) with Detected Tissue (green)")

    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_binary_mask_png(mask: np.ndarray, save_to: str | Path) -> None:
    """Save a boolean mask as a full-resolution 0/255 grayscale PNG figure.

    A viewable snapshot of the tissue mask at level 0. It is a figure only
    — ``tissue.geojson`` is the source of truth downstream.

    Args:
        mask: Boolean ``(y, x)`` mask at level 0.
        save_to: Destination PNG path.

    Example:
        >>> import tempfile, numpy as np
        >>> from pathlib import Path
        >>> from PIL import Image
        >>> m = np.zeros((8, 8), dtype=bool)
        >>> m[2:6, 2:6] = True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "mask.png"
        ...     save_binary_mask_png(m, p)
        ...     bool(((np.asarray(Image.open(p)) > 0) == m).all())
        True
    """
    from PIL import Image

    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(save_to)


def save_stretched_png(
    image: np.ndarray, save_to: str | Path, *, max_display: int = 1400
) -> None:
    """Percentile-stretch a 2-D channel to a light grayscale PNG.

    For the structural max-projection review image — the same contrast
    stretch the overlay backdrop uses, downsampled to ~``max_display`` px
    per side.

    Args:
        image: 2-D channel (e.g. a max-projection of structural markers).
        save_to: Destination PNG path.
        max_display: Longest side of the saved image, in pixels.

    Example:
        >>> import tempfile, numpy as np
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "mp.png"
        ...     save_stretched_png(np.zeros((20, 20), "uint16"), p)
        ...     p.exists()
        True
    """
    from PIL import Image

    arr = np.asarray(image)
    factor = max(1, int(np.ceil(max(arr.shape) / max_display)))
    gray = normalize_uint8(arr[::factor, ::factor])
    Image.fromarray(gray, mode="L").save(save_to)


def write_tissue_geojson(
    gdf: gpd.GeoDataFrame, save_to: str | Path, *, simplify_px: float = 2.0
) -> None:
    """Write tissue polygons as a QuPath GeoJSON FeatureCollection.

    TRIDENT-style — one ``Feature`` per tissue polygon, ``tissue_id`` +
    ``geometry`` in level-0 pixel coordinates — plus a green ``Tissue``
    classification so QuPath shows it as an editable Tissue annotation.
    Boundaries are lightly simplified (``simplify_px``).

    Args:
        gdf: ``tissue_id`` + ``geometry`` polygons (level-0 coordinates).
        save_to: Destination ``.geojson`` path.
        simplify_px: Douglas-Peucker tolerance in level-0 pixels.

    Example:
        >>> import tempfile, numpy as np
        >>> from pathlib import Path
        >>> m = np.zeros((10, 10), dtype=bool)
        >>> m[2:8, 2:8] = True
        >>> gdf = mask_to_geopandas(m)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "t.geojson"
        ...     write_tissue_geojson(gdf, p)
        ...     '"name": "Tissue"' in p.read_text()
        True
    """
    import json

    from shapely.geometry import mapping

    features = [
        {
            "type": "Feature",
            "geometry": mapping(geom.simplify(simplify_px)),
            "properties": {
                "tissue_id": i,
                "classification": {"name": "Tissue", "color": [0, 200, 0]},
            },
        }
        for i, geom in enumerate(gdf.geometry)
    ]
    doc = {"type": "FeatureCollection", "features": features}
    Path(save_to).write_text(json.dumps(doc))


def _iter_tissue_polygons(geojson_path: str | Path) -> Iterator[Polygon]:
    """Yield each tissue Polygon in a geojson, in file order.

    Shared by the geojson readers: loads the FeatureCollection and flattens
    every Polygon / MultiPolygon feature into individual polygons, ignoring
    any non-polygon annotation. Centralizes "what counts as a tissue
    polygon" so the rasterizer and the geometry hash cannot drift.
    """
    import json

    from shapely.geometry import MultiPolygon, shape

    doc = json.loads(Path(geojson_path).read_text())
    for feature in doc.get("features", []):
        geom = shape(feature["geometry"])
        if isinstance(geom, MultiPolygon):
            polys: list[Polygon] = list(geom.geoms)
            yield from polys
        elif isinstance(geom, Polygon):
            yield geom


def read_tissue_geojson_mask(
    geojson_path: str | Path, height: int, width: int
) -> np.ndarray:
    """Rasterize a (possibly QuPath-edited) tissue GeoJSON to a mask.

    The inverse of :func:`write_tissue_geojson`: fills each polygon's
    exterior (minus its holes) into a boolean ``(height, width)`` mask at
    level 0, so a user can edit the tissue boundary in QuPath and re-import
    it via ``coral tissue --custom-mask-path``.

    Args:
        geojson_path: A tissue ``.geojson`` (level-0 pixel coordinates).
        height: Level-0 store height in pixels.
        width: Level-0 store width in pixels.

    Returns:
        Boolean ``(height, width)`` tissue mask.

    Example:
        >>> import tempfile, numpy as np
        >>> from pathlib import Path
        >>> m = np.zeros((10, 10), dtype=bool)
        >>> m[2:8, 2:8] = True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "t.geojson"
        ...     write_tissue_geojson(mask_to_geopandas(m), p)
        ...     back = read_tissue_geojson_mask(p, 10, 10)
        ...     bool(back[5, 5])
        True
    """
    from PIL import Image, ImageDraw

    # Rasterize with Pillow's C polygon filler, not skimage.draw.polygon:
    # on large level-0 masks the latter is pathologically slow (~230s for
    # a few thousand-vertex TMA core vs ~0.1s here). Exteriors fill True and
    # holes fill False in file order, so overlapping polygons compose exactly
    # as skimage's sequential writes did.
    img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for poly in _iter_tissue_polygons(geojson_path):
        xs, ys = poly.exterior.coords.xy
        draw.polygon(list(zip(xs, ys, strict=True)), fill=1)
        for interior in poly.interiors:
            ixs, iys = interior.coords.xy
            draw.polygon(list(zip(ixs, iys, strict=True)), fill=0)
    return np.asarray(img, dtype=bool)


def geojson_geometry_sha256(geojson_path: str | Path) -> str:
    """Hash a tissue GeoJSON's polygon geometry, edit-detection stable.

    A SHA-256 over the *normalized* polygon coordinates — integer pixel
    rings, sorted for order-independence — not the raw file bytes. So a
    cosmetic re-save (reordered keys, added properties, whitespace) yields
    the same hash, while any moved boundary vertex changes it. Lets a
    caller tell whether a boundary was hand-edited (e.g. in QuPath) since
    CORAL wrote it, robustly across file copies / version control (unlike an
    mtime check). Non-polygon features are ignored, as in
    :func:`read_tissue_geojson_mask`.

    Args:
        geojson_path: A tissue ``.geojson`` (level-0 pixel coordinates).

    Returns:
        A 64-character hex SHA-256 digest of the normalized geometry.

    Example:
        >>> import tempfile, numpy as np
        >>> from pathlib import Path
        >>> m = np.zeros((10, 10), dtype=bool)
        >>> m[2:8, 2:8] = True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "t.geojson"
        ...     write_tissue_geojson(mask_to_geopandas(m), p)
        ...     digest = geojson_geometry_sha256(p)
        ...     len(digest)
        64
    """
    import hashlib
    import json

    rings: list[list[list[int]]] = []
    for poly in _iter_tissue_polygons(geojson_path):
        for ring in (poly.exterior, *poly.interiors):
            rings.append(
                [[int(round(x)), int(round(y))] for x, y in ring.coords]
            )
    rings.sort()
    payload = json.dumps(rings, separators=(",", ":"))
    return hashlib.sha256(payload.encode(), usedforsecurity=False).hexdigest()
