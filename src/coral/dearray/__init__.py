"""TMA dearraying: finding where the cores are on a whole-slide store.

``coral ingest-wsi`` writes a whole TMA slide as one OME-Zarr store and
says nothing about where its cores sit. This package finds them, and
writes the answer as **geometry rather than pixels**.

That choice is the whole design, and it comes from measurement. On an 11 GB
29-channel TMA store, detection is 15 s and 54 KB of geometry;
``--export-cores`` on the same slide is 227 s and 4.5 GB, which is 15x the
time and 80,000x the bytes. The copies add 41% to that parent rather than
the doubling you would expect, because blosc has already compressed the
empty glass away: what is left in a TMA store is the tissue, so cropping
the cores out of it saves very little.

And nothing in the science path needs a copy. Patching inside a core is a
set of offsets into the parent store, and reading a patch is reading
chunks. The three things that want *reduced* resolution (a viewer drawing a
core, patching at a target mpp, tissue segmentation at its 1.0 µm/px
inference scale) all read the parent's own pyramid at the scaled window,
because level 1 of a core is the corresponding window of level 1 of the
slide.

So a per-core store earns its cost in exactly one case: handing a core to
something that cannot see the parent. That is interop, and it is a flag
(``--export-cores``), not the default.

Layout follows :mod:`coral.tissue` exactly, one directory per method,
discovered by scanning::

    <store>/dearray/dearray_carta/       # what the detector found
        cores.geojson    # level-0 pixel coordinates
        overlay.png      # QC, on the canvas the detector saw
        dearray.json     # method, params, provenance
    <store>/dearray/dearray_imported/    # what a human supplied
        cores.geojson
        dearray.json

Those two are separate directories on purpose. Detection can never overwrite
boxes a human is responsible for, so there is no edit to detect and no
``--force`` to override it, which is how :mod:`coral.tissue` already handles
an imported mask. Where both exist, the human's boxes are the ones exported.

The correction round trip is :mod:`coral.dearray.corrections`, and the export
rules — what a core store contains, and when a core is re-cut — are in
:mod:`coral.dearray.export`.
"""

from coral.dearray.corrections import match_cores_file
from coral.dearray.paths import (
    DEFAULT_DEARRAY_METHOD,
    IMPORTED_METHOD,
    dearray_dir,
    dearray_rel,
    list_dearray_methods,
)
from coral.dearray.run import (
    DEFAULT_CONF,
    DearrayResult,
    dearray_slide,
    export_method_for,
)

__all__ = [
    "DEFAULT_CONF",
    "DEFAULT_DEARRAY_METHOD",
    "IMPORTED_METHOD",
    "DearrayResult",
    "dearray_dir",
    "dearray_rel",
    "dearray_slide",
    "export_method_for",
    "list_dearray_methods",
    "match_cores_file",
]
