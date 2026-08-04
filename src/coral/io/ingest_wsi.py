"""Canonical ingest for PerkinElmer/Akoya qptiff whole-slide scans.

``convert_wsi_to_canonical`` is the qptiff sibling of
``coral.io.ingest.convert_to_canonical``: same output, different source.
It writes the identical canonical ``(c, y, x)`` OME-Zarr store — same
directory layout, same ``.zattrs`` keys, same ``state.json`` — and reuses
that module's marker resolution, mpp resolution, nuclear-channel
resolution, NGFF attrs, thumbnail, state and structure-map helpers.

Three things differ:

1. **Marker names come from the file.** A qptiff's per-page XML carries
   the antibody target in ``<Biomarker>``, so no sidecar name list is
   needed (``--channel-names`` still overrides it).
2. **Pixels are streamed one channel at a time.** A full-resolution
   whole slide is 35-53 GB, so the store is created empty and filled
   plane by plane; peak memory is one 2-D plane. This is why
   ``_write_canonical_zarr`` (which takes a materialised array) is not
   reused directly, though its attrs writer and nuclear resolution are.
3. **The pyramid is kept.** A qptiff already carries reduced levels, so
   they are streamed in as datasets ``1..N-1`` and every one is listed in
   ``multiscales``. Without them QuPath and napari decode 28800 x 30720
   to draw a thumbnail, which looks like the application has hung.
   ``coral ingest`` still writes level 0 alone; only this path has a
   source pyramid to copy.

A whole slide is either one tissue section or a TMA, and this module does
not care which: it writes the slide as one store either way. Detecting a
TMA's cores is ``coral.dearray``, which the ``ingest-wsi`` command runs
afterwards when the caller passes ``--dearray``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from coral.io.harmonize import CanonicalChannel
from coral.io.ingest import (
    _CHUNK_TILE,
    Resolution,
    _apply_channel_names,
    _apply_resolution,
    _measured_window,
    _resolve_mpp,
    _resolve_nuclear_channel,
    _rgb_to_hex,
    _write_ngff_external_attrs,
    _write_nuclear_thumbnail,
    resolve_cohort_markers,
)
from coral.io.readers.qptiff import (
    QPTIFF_EXTENSIONS,
    qptiff_channel_iter,
    read_qptiff_metadata,
    read_qptiff_overviews,
    read_qptiff_record,
)
from coral.slide.core import CoralSlide
from coral.slide.state import SlideMeta, default_state, load_state, save_state
from coral.slide.structure import write_structure_map
from coral.utils.errors import ReaderError
from coral.utils.progress import channel_bar, writing_channel
from coral.utils.time import now_iso

logger = logging.getLogger(__name__)

__all__ = [
    "convert_wsi_to_canonical",
    "qptiff_marker_names",
    "wsi_output_stem",
]

#: The only source axes order a qptiff Baseline series is written in.
#: Anything else means the file is not what this reader assumes, so we
#: refuse rather than transpose on a guess.
_EXPECTED_AXES = "CYX"


def wsi_output_stem(path: Path) -> str:
    """Output store stem — the qptiff name minus its extension."""
    name = path.name
    for ext in QPTIFF_EXTENSIONS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _names_from(channel_dicts: list[dict[str, Any]]) -> list[str]:
    """Reader channel dicts -> one usable name per channel.

    Channels the file leaves unnamed become ``channel_{i}`` placeholders,
    so the list is always exactly one name per channel — which is what
    :func:`coral.io.ingest.resolve_cohort_markers` expects from its
    ``channel_names`` argument.
    """
    return [
        str(c["name"]) if c["name"] else f"channel_{i}"
        for i, c in enumerate(channel_dicts)
    ]


def qptiff_marker_names(path: str | Path) -> list[str]:
    """Marker names of one qptiff, in channel order — no pixel read.

    Args:
        path: Path to a ``.qptiff`` file.

    Returns:
        One marker name per channel, in channel order.

    Raises:
        ReaderError: If the file cannot be read.
    """
    channel_dicts, _, _ = read_qptiff_metadata(path)
    return _names_from(channel_dicts)


def _canonical_channels(
    channel_dicts: list[dict[str, Any]], axes: str
) -> list[CanonicalChannel]:
    """Build canonical channels for an already-``(c, y, x)`` source.

    The qptiff Baseline series is always ``CYX``, so there is nothing to
    reorder and ``harmonize_to_canonical`` (which needs the materialised
    array) would be a no-op we cannot afford on a whole slide. Anything
    other than ``CYX`` is refused rather than guessed at.
    """
    if axes != _EXPECTED_AXES:
        raise ReaderError(
            f"qptiff Baseline series has axes {axes!r}, expected "
            f"{_EXPECTED_AXES!r}. Refusing to guess the channel axis."
        )
    return [CanonicalChannel(raw=c["marker_raw"]) for c in channel_dicts]


def _write_streamed_canonical_zarr(
    out_path: Path,
    source_path: Path,
    level: int,
    shape_cyx: tuple[int, int, int],
    dtype: np.dtype,
    markers: list[str],
    channels: list[CanonicalChannel],
    mpp: float,
    source_pyramid_levels: int,
    level_shapes: list[tuple[int, int, int]],
    name_source: str,
    nuclear_marker: str | None = None,
) -> tuple[int, np.ndarray]:
    """Write a canonical OME-Zarr store, streaming one channel at a time.

    The streaming counterpart of
    :func:`coral.io.ingest._write_canonical_zarr`: identical store on
    disk (same dataset name, chunking, and the same four CORAL attrs plus
    the shared NGFF attrs), but the dataset is created empty and filled
    plane by plane, so peak memory is one 2-D plane instead of the whole
    ``(c, y, x)`` stack.

    Args:
        out_path: Destination ``.zarr`` directory (created/overwritten).
        source_path: The qptiff to stream planes from.
        level: Pyramid level to read.
        shape_cyx: Canonical ``(c, y, x)`` shape of that level.
        dtype: Pixel dtype of that level.
        markers: Resolved marker names, one per channel.
        channels: Per-channel metadata, one per channel axis.
        mpp: Resolved microns-per-pixel of the written level.
        source_pyramid_levels: Pyramid level count in the source.
        level_shapes: ``(c, y, x)`` per source pyramid level, level 0
            first. Levels below ``level`` become datasets ``1..N-1``.
        name_source: Which XML element the channel names came from,
            ``"Biomarker"`` or ``"Name"``.
        nuclear_marker: Explicit nuclear-channel marker name, or ``None``
            to auto-infer.

    Returns:
        ``(nuclear_index, nuclear_plane)`` — the resolved nuclear channel
        index and its 2-D pixels, kept for the thumbnail so the store is
        not read back.

    Raises:
        ValueError: If no nuclear channel can be identified.
        ReaderError: If a channel plane's shape does not match the level.
    """
    n_channels, height, width = shape_cyx
    chunks = (1, min(_CHUNK_TILE, height), min(_CHUNK_TILE, width))

    # Resolve + validate the nuclear channel BEFORE writing anything, so a
    # nuclear-less panel (or a bad --nuclear-marker) fails cleanly without
    # leaving a partial store on disk. Same ordering as the array ingest.
    nuclear_idx = _resolve_nuclear_channel(
        out_path.name,
        markers,
        nuclear_marker,
        kept=[bool(ch.keep) for ch in channels],
    )

    total_levels = len(level_shapes) - level
    logger.info(
        "  writing level 0 of %d: %d channel(s) at %d x %d",
        total_levels,
        n_channels,
        height,
        width,
    )
    # A channel the marker map has not resolved has an empty marker, so
    # the raw source name is used for the log. Otherwise every unresolved
    # channel reports as a blank, which is exactly the panel (a Polaris
    # fluorophore panel) where you most want to see what is going past.
    labels = [
        (m if m else (ch.raw or f"channel_{i}"))
        for i, (m, ch) in enumerate(zip(markers, channels, strict=True))
    ]
    root = zarr.open_group(str(out_path), mode="w")
    # Created empty and filled plane by plane. `_write_canonical_zarr` hands
    # `create_dataset` a materialised array; a full-resolution whole slide is
    # tens of gigabytes and never is one.
    dataset = root.create_dataset(
        "0", shape=shape_cyx, dtype=dtype, chunks=chunks
    )
    nuclear_plane: np.ndarray | None = None
    n_written = 0
    # A bar rather than a line per channel. The channel name still has to
    # be visible, because a full-resolution level is many minutes of work
    # and silence is indistinguishable from a hang; it rides on the bar's
    # postfix instead of costing a log line each.
    with channel_bar(desc="  level 0", total=n_channels) as bar:
        for index, plane in qptiff_channel_iter(source_path, level=level):
            if plane.shape != (height, width):
                raise ReaderError(
                    f"{source_path.name}: channel {index} is {plane.shape} "
                    f"but level {level} is {(height, width)}"
                )
            writing_channel(bar, labels[index])
            dataset[index] = plane
            _verify_written_plane(dataset, index, plane, source_path, level)
            if index == nuclear_idx:
                nuclear_plane = plane
            n_written += 1
            bar.update(1)
    if n_written != n_channels or nuclear_plane is None:
        raise ReaderError(
            f"{source_path.name}: streamed {n_written} of {n_channels} "
            f"channel(s) at level {level}; the store is incomplete."
        )

    stored_shapes = [shape_cyx]
    stored_shapes += _write_reduced_levels(
        root, source_path, level, level_shapes, dtype, labels
    )

    logger.info("  writing metadata: NGFF attrs, OME-XML, source record")
    root.attrs["channels"] = [ch.model_dump() for ch in channels]
    # The nuclear stain is recorded by NAME so downstream stages read it
    # without re-inferring (see CoralSlide.nuclear_channel).
    root.attrs["nuclear_channel"] = markers[nuclear_idx]
    root.attrs["mpp"] = mpp
    root.attrs["source_pyramid_levels"] = source_pyramid_levels
    record = _source_record(source_path, mpp, name_source, level, nuclear_idx)
    root.attrs["source_qptiff"] = record
    # Display windows measured from the coarsest level just written. Those
    # pixels are already on disk and small (a 34560px slide's coarsest level
    # is about 1080px), so this costs a fraction of a second and saves every
    # reader from inventing its own contrast.
    coarsest = root[str(len(stored_shapes) - 1)]
    windows = {}
    for i in range(int(stored_shapes[0][0])):
        measured = _measured_window(np.asarray(coarsest[i]))
        if measured:
            windows[i] = measured
    # The scanner's own per-channel colour, where it recorded one.
    colors = {}
    for i, entry in enumerate(record.get("channels") or []):
        hexed = _rgb_to_hex(entry.get("Color"))
        if hexed:
            colors[i] = hexed
    _write_ngff_external_attrs(
        root,
        # Resolved marker, falling back to raw so QuPath never sees a blank.
        labels,
        mpp,
        nuclear_marker=markers[nuclear_idx],
        level_shapes=stored_shapes,
        source=record,
        dtype=np.dtype(dtype),
        windows=windows,
        channel_colors=colors,
    )
    logger.info("  verifying the finished store")
    _verify_store(root, out_path, stored_shapes, mpp)
    return nuclear_idx, nuclear_plane


#: Side of the square window read back from each written plane.
_PROBE = 512


def _verify_written_plane(
    dataset: zarr.Array,
    index: int,
    plane: np.ndarray,
    source_path: Path,
    level: int,
) -> None:
    """Read one window back out of the store and compare it to the source.

    Checks the bytes that just went in came out again at the same channel
    index. Reading back **by index** is the point: the failure this exists
    for is a plane landing at the wrong index or in the wrong level, which
    corrupts nothing a viewer will complain about. The image simply goes
    soft or wrong at one zoom and nobody notices for months.

    A centre window rather than the whole plane, because the whole plane
    is 884 MB at full resolution and re-reading every one of them would
    roughly double the ingest for a check that a window already fails.

    Args:
        dataset: The zarr array just written to.
        index: Channel index written.
        plane: The source pixels, still in memory.
        source_path: For the error message.
        level: Source pyramid level, for the error message.

    Raises:
        ReaderError: If the window read back differs from the source.
    """
    height, width = plane.shape
    y = max(0, (height - _PROBE) // 2)
    x = max(0, (width - _PROBE) // 2)
    y_end, x_end = min(height, y + _PROBE), min(width, x + _PROBE)
    written = np.asarray(dataset[index, y:y_end, x:x_end])
    if not np.array_equal(written, plane[y:y_end, x:x_end]):
        raise ReaderError(
            f"{source_path.name}: channel {index} at level {level} read "
            f"back different from the source at rows {y}:{y_end}, columns "
            f"{x}:{x_end}. The store is wrong, not merely incomplete."
        )


def _verify_store(
    root: zarr.Group,
    out_path: Path,
    stored_shapes: list[tuple[int, int, int]],
    mpp: float,
) -> None:
    """Check the finished store agrees with itself before it is declared done.

    Pixels are verified plane by plane as they are written
    (:func:`_verify_written_plane`); this is the structural half. Three
    places record the pixel size and a store where they disagree is worse
    than one that never claimed to know: ``mpp``, the level-0
    ``multiscales`` scale, and ``PhysicalSizeX`` in the OME-XML.

    Args:
        root: The open zarr group.
        out_path: The store directory, for reading the OME-XML back.
        stored_shapes: ``(c, y, x)`` per dataset that should exist.
        mpp: The resolved microns per pixel.

    Raises:
        ReaderError: If a dataset is missing or misshapen, or the three
            pixel sizes disagree.
    """
    for i, shape in enumerate(stored_shapes):
        key = str(i)
        if key not in root:
            raise ReaderError(
                f"{out_path.name}: dataset {key!r} is missing; the store "
                f"claims {len(stored_shapes)} pyramid level(s)."
            )
        actual = tuple(int(v) for v in root[key].shape)
        if actual != tuple(shape):
            raise ReaderError(
                f"{out_path.name}: dataset {key!r} is {actual}, expected "
                f"{tuple(shape)}."
            )

    # Missing attrs or a missing document is itself the failure, so it is
    # reported as one rather than surfacing as a KeyError from the depths
    # of an attribute lookup.
    try:
        # OME-Zarr 0.5 nests the keys under "ome"; 0.4 keeps them flat.
        # Reading either means the check does not care which format the
        # store was written in.
        attrs = root.attrs
        block = attrs.get("ome", attrs)
        scale = block["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ][0]["scale"]
        xml = (out_path / "OME" / "METADATA.ome.xml").read_text()
    except (KeyError, IndexError, OSError) as exc:
        raise ReaderError(
            f"{out_path.name}: the store is missing the metadata needed to "
            f"check its own pixel size ({type(exc).__name__}: {exc})."
        ) from exc
    match = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', xml)
    physical = float(match.group(1)) if match else None
    sizes = {
        "mpp attribute": float(root.attrs["mpp"]),
        "multiscales level-0 scale": float(scale[1]),
        "OME PhysicalSizeX": physical,
    }
    if physical is None or any(
        abs(v - mpp) > 1e-9 for v in sizes.values() if v is not None
    ):
        raise ReaderError(
            f"{out_path.name}: the store disagrees with itself about the "
            f"pixel size ({sizes}); resolved mpp was {mpp}."
        )


#: Channel names that are not antibody targets. ``Sample AF`` is the
#: autofluorescence channel a Vectra Polaris saves alongside the real
#: stains; it is imaged, so it is kept, but a downstream stage that treats
#: it as a biomarker is computing statistics on tissue background.
_AUTOFLUORESCENCE = ("sample af", "autofluorescence", "af")


def _channel_role(name: str | None, is_nuclear: bool) -> str:
    """What a channel is, as opposed to what it is called.

    Every channel is kept regardless. This only records which kind each
    one is, so a later stage does not have to re-derive it from a name it
    may not recognise.

    Args:
        name: The channel's source name.
        is_nuclear: Whether it resolved as the nuclear stain.

    Returns:
        ``"nuclear"``, ``"autofluorescence"`` or ``"marker"``.

    Example:
        >>> _channel_role("Sample AF", False)
        'autofluorescence'
        >>> _channel_role("CD20", False)
        'marker'
        >>> _channel_role("DAPI", True)
        'nuclear'
    """
    if is_nuclear:
        return "nuclear"
    if (name or "").strip().lower() in _AUTOFLUORESCENCE:
        return "autofluorescence"
    return "marker"


def _write_overviews(out_path: Path, source_path: Path) -> list[str]:
    """Save the scanner's Label / Macro / Thumbnail beside the store.

    These are the only images in a qptiff that identify the physical
    slide: the Label is a photograph of the label, the Macro a whole-slide
    overview. CORAL's nuclear preview shows what was imaged, not which
    slide it came off.

    Never fatal. An overview is provenance, not data, and losing it is not
    worth failing an ingest that has already streamed 13 GB of pixels.

    Args:
        out_path: The ``.zarr`` store directory.
        source_path: The source qptiff.

    Returns:
        The file names written, for the run log and ``state.json``.
    """
    from PIL import Image

    try:
        overviews = read_qptiff_overviews(source_path)
    except (ReaderError, OSError) as exc:
        logger.warning(
            "%s: could not read the scanner overview images (%s)",
            source_path.name,
            exc,
        )
        return []
    if not overviews:
        # Normal on Vectra Polaris, which writes only a Thumbnail.
        return []
    out_dir = out_path / "overviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, array in sorted(overviews.items()):
        target = out_dir / f"{name.lower()}.png"
        try:
            Image.fromarray(np.asarray(array)).save(target)
        except (ValueError, OSError) as exc:
            logger.warning(
                "%s: could not save the %s overview (%s)",
                source_path.name,
                name,
                exc,
            )
            continue
        written.append(target.name)
    return written


def _source_record(
    source_path: Path,
    mpp: float,
    name_source: str,
    level: int,
    nuclear_idx: int = -1,
) -> dict[str, Any]:
    """The qptiff's own account of itself, for the store's attributes.

    Everything the scanner wrote, plus how CORAL read it. The split is
    deliberate: ``OME/METADATA.ome.xml`` is the interoperability surface
    and carries what the OME schema has a field for, and this carries the
    rest, because a vendor always records more than any schema models.

    Nobody should have to reopen a 4 GB qptiff to find out what objective
    was used or when the slide was scanned.

    Args:
        source_path: The ingested ``.qptiff``.
        mpp: The microns per pixel CORAL resolved and stored.
        name_source: The XML element the channel names came from.
        level: Source pyramid level stored as dataset ``"0"``.
        nuclear_idx: Index of the resolved nuclear channel, for tagging
            each channel's role.

    Returns:
        A JSON-safe record. Never raises: metadata is worth having, and
        not worth failing a 40-minute ingest over.
    """
    from coral import __version__

    try:
        record = read_qptiff_record(source_path)
    except (ReaderError, OSError) as exc:
        logger.warning(
            "%s: could not read the full source metadata record (%s); the "
            "store keeps its channels, mpp and pyramid but not the "
            "scanner's own account of the acquisition.",
            source_path.name,
            exc,
        )
        return {}
    # Every channel is kept; this only records which kind each one is, so
    # a later stage does not compute biomarker statistics on the
    # autofluorescence channel.
    for i, entry in enumerate(record.get("channels") or []):
        entry["role"] = _channel_role(
            entry.get("Biomarker") or entry.get("Name"), i == nuclear_idx
        )
    nominal = record.get("acquisition", {}).get("mpp_nominal")
    record["provenance"].update(
        {
            # The tag is authoritative and the nominal figure is what the
            # operator set. They disagree by 0.5% on Vectra Polaris, so
            # both are kept rather than one quietly winning.
            #
            # The nominal figure describes level 0 and mpp_stored
            # describes whatever level was ingested, so the key says so:
            # at --level 3 they read 0.5 and 3.98 and comparing them
            # without that label would look like a fault.
            "mpp_stored": float(mpp),
            "mpp_source": "tiff_resolution_tag",
            "mpp_nominal_level0": nominal,
            "channel_name_source": name_source,
            "source_level": level,
            "coral_version": __version__,
        }
    )
    return record


def _write_reduced_levels(
    root: zarr.Group,
    source_path: Path,
    level: int,
    level_shapes: list[tuple[int, int, int]],
    dtype: np.dtype,
    labels: list[str],
) -> list[tuple[int, int, int]]:
    """Stream the source's reduced levels in as datasets ``1..N-1``.

    The scanner already wrote a pyramid, so the levels below the ingested
    one are read straight from the file rather than downsampled from level
    0. That is cheaper, and it is the scanner's own reduction rather than
    a second opinion about how to reduce.

    ``--level N`` shifts the whole thing: level ``N`` becomes dataset
    ``"0"`` and the pyramid is whatever the source has below it, so a
    reduced-level ingest is a smaller store, not a store missing levels.

    Pixels are streamed one channel at a time, like level 0, so peak
    memory stays one plane no matter how many levels there are.

    Args:
        root: Open zarr group for the slide store, with ``"0"`` written.
        source_path: The qptiff to stream from.
        level: Source pyramid level stored as dataset ``"0"``.
        level_shapes: ``(c, y, x)`` per source level, level 0 first.
        dtype: Pixel dtype, the same at every level.
        labels: Channel display names, for the level bar's postfix.

    Returns:
        The ``(c, y, x)`` shape of each dataset written, in level order.

    Raises:
        ReaderError: If a level yields a plane of the wrong shape, or
            fewer channels than the level declares.
    """
    written: list[tuple[int, int, int]] = []
    for offset, shape in enumerate(level_shapes[level + 1 :], start=1):
        n_channels, height, width = (int(v) for v in shape)
        logger.info(
            "  writing level %d of %d: %d channel(s) at %d x %d",
            offset,
            len(level_shapes) - level - 1,
            n_channels,
            height,
            width,
        )
        chunks = (1, min(_CHUNK_TILE, height), min(_CHUNK_TILE, width))
        level_shape = (n_channels, height, width)
        dataset = root.create_dataset(
            str(offset), shape=level_shape, dtype=dtype, chunks=chunks
        )
        n_written = 0
        with channel_bar(desc=f"  level {offset}", total=n_channels) as bar:
            for index, plane in qptiff_channel_iter(
                source_path, level=level + offset
            ):
                if plane.shape != (height, width):
                    raise ReaderError(
                        f"{source_path.name}: level {level + offset} channel "
                        f"{index} is {plane.shape}, expected "
                        f"{(height, width)}"
                    )
                writing_channel(bar, labels[index])
                dataset[index] = plane
                _verify_written_plane(
                    dataset, index, plane, source_path, level + offset
                )
                n_written += 1
                bar.update(1)
        if n_written != n_channels:
            raise ReaderError(
                f"{source_path.name}: level {level + offset} streamed "
                f"{n_written} of {n_channels} channel(s); "
                f"the store is incomplete."
            )
        written.append((n_channels, height, width))
        logger.debug(
            "  %s: wrote level %d, shape (c,y,x)=(%d, %d, %d)",
            root.store.path,
            offset,
            n_channels,
            height,
            width,
        )
    return written


def convert_wsi_to_canonical(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    resolution: Resolution | None = None,
    level: int = 0,
    mpp: float | None = None,
    mpp_map: dict[str, float] | None = None,
    channel_names: list[str] | None = None,
    nuclear_marker: str | None = None,
    quiet: bool = False,
) -> CoralSlide:
    """Ingest one qptiff whole slide into a canonical OME-Zarr store.

    Reads the qptiff's Baseline series at ``level``, takes each channel's
    marker name from the file's own ``<Biomarker>`` XML, resolves those
    names against the cohort marker map, and writes
    ``<output_dir>/<name>.zarr`` — the same store ``coral ingest``
    produces: the ``0`` pixel array, ``OME/METADATA.ome.xml``,
    ``thumbnails/nuclear.png``, ``state.json``, ``structure.txt`` and the
    ``channels`` / ``nuclear_channel`` / ``mpp`` /
    ``source_pyramid_levels`` attributes.

    Pixels are streamed one channel at a time, so a full-resolution whole
    slide does not have to fit in memory.

    If a completed store already exists it is left untouched and reopened.
    When ``resolution`` is omitted, marker names are resolved for this
    single image and a marker map is written to ``output_dir``.

    Args:
        input_path: Source ``.qptiff`` file.
        output_dir: Directory to write ``<name>.zarr`` and the marker map
            into.
        resolution: Cohort resolution from
            :func:`coral.io.ingest.resolve_cohort_markers`; built for this
            image alone when ``None``.
        level: Pyramid level to ingest; 0 (default) is full resolution.
            Reduced levels are for quick end-to-end checks.
        mpp: Global microns-per-pixel fallback.
        mpp_map: Per-image microns-per-pixel overrides, keyed by file name.
        channel_names: Marker names overriding the file's embedded
            ``<Biomarker>`` names; length must equal the channel count.
        nuclear_marker: Force the nuclear channel by marker name; ``None``
            auto-infers (DAPI on a standard qptiff panel).
        quiet: Skip the per-image summary log line.

    Returns:
        An open ``CoralSlide`` handle to the written canonical store.

    Raises:
        ReaderError: If the qptiff cannot be read, ``level`` is out of
            range, or the file's channel metadata is inconsistent.
        ValueError: If microns-per-pixel cannot be resolved; if
            ``channel_names`` length differs from the channel count; or if
            no nuclear channel can be identified.

    Example:
        Ingest a reduced level for a quick end-to-end check::

            slide = convert_wsi_to_canonical("scan.qptiff", "out/", level=2)
    """
    path = Path(input_path)
    out_path = Path(output_dir) / f"{wsi_output_stem(path)}.zarr"

    channel_dicts, source_mpp, meta = read_qptiff_metadata(path, level=level)
    # A qptiff with no <Biomarker> falls back to the fluorophore <Name>
    # (DAPI / ATTO 550 / Cy5), which repeats every cycle and is not a marker.
    # Ingesting that silently would put a panel of dye names into the marker
    # map and every downstream reference to a "marker" would be a lie. Refuse
    # unless the caller supplies the real names.
    if channel_names is None and meta.get("channel_name_source") == "Name":
        raise ReaderError(
            f"{path.name} carries no <Biomarker> element, so its channels are "
            f"named after fluorophores ("
            f"{', '.join(str(n) for n in _names_from(channel_dicts)[:3])}...) "
            f"rather than antibody targets. Pass --channel-names with a text "
            f"file of the real markers, one per line in channel order "
            f"({len(channel_dicts)} of them)."
        )
    names = channel_names or _names_from(channel_dicts)

    if resolution is None:
        # The qptiff carries its own marker names, so they are handed to
        # the shared cohort resolver as an explicit name list — that path
        # never opens the source, so it needs no qptiff support.
        resolution, _ = resolve_cohort_markers(
            [path],
            output_dir,
            channel_names=names,
            nuclear_marker=nuclear_marker,
        )

    if out_path.exists() and (
        load_state(out_path).tasks.ingest.status == "completed"
    ):
        if not quiet:
            logger.info("  %s: already ingested", path.name)
        return CoralSlide.open(out_path)

    started = now_iso()
    channels = _canonical_channels(channel_dicts, str(meta["axes"]))
    if channel_names is not None:
        _apply_channel_names(channels, channel_names)
    markers = _apply_resolution(channels, resolution)
    # --mpp OVERRIDES the qptiff here, where on `coral ingest` it is only a
    # fallback. A scanner can be configured with the wrong objective, and
    # when that happens the file's own figure is confidently wrong and
    # there has to be a way to say so. --mpp-csv still wins over --mpp, so
    # a per-image correction beats a blanket one.
    resolved_mpp = _resolve_mpp(
        path.name,
        mpp if mpp is not None else source_mpp,
        mpp_map=mpp_map,
        mpp_global=mpp,
        level=level,
    )

    shape = meta["shape"]
    shape_cyx = (int(shape[0]), int(shape[1]), int(shape[2]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nuclear_idx, nuclear_plane = _write_streamed_canonical_zarr(
        out_path,
        path,
        level,
        shape_cyx,
        np.dtype(str(meta["dtype"])),
        markers,
        channels,
        resolved_mpp,
        int(meta.get("pyramid_levels", 1)),
        [
            (int(s[0]), int(s[1]), int(s[2]))
            for s in meta.get("level_shapes", [shape])
        ],
        str(meta.get("channel_name_source", "")),
        nuclear_marker=nuclear_marker,
    )
    _write_nuclear_thumbnail(out_path, nuclear_plane)
    overviews = _write_overviews(out_path, path)

    state = default_state(out_path, image_path=str(path.resolve()))
    state.tasks.ingest.status = "completed"
    state.tasks.ingest.started_at = started
    state.tasks.ingest.completed_at = now_iso()
    state.tasks.ingest.outputs = {"image": "0"}
    if overviews:
        state.tasks.ingest.outputs["overviews"] = ", ".join(overviews)
    state.meta = SlideMeta(
        dimensions=(shape_cyx[1], shape_cyx[2]),
        n_markers=len(channels),
        mpp=resolved_mpp,
        # Extra provenance the array ingest has no equivalent for. Kept in
        # state.meta (which allows extra fields) rather than .zattrs, so
        # the store's attribute keys stay identical to `coral ingest`.
        source_format="qptiff",
        source_level=level,
        source_pyramid_levels=int(meta.get("pyramid_levels", 1)),
        # Recorded verbatim, exactly as the file describes itself.
        source_slide_id=meta.get("slide_id"),
        source_channel_names_from=meta.get("channel_name_source"),
    )
    save_state(out_path, state)
    write_structure_map(out_path)

    if mpp_map and path.name in mpp_map:
        mpp_source = "from --mpp-csv"
    elif mpp is not None:
        mpp_source = (
            f"from --mpp, OVERRIDING the file's {source_mpp:.4g}"
            if source_mpp is not None
            else "from --mpp, the file carries none"
        )
    else:
        mpp_source = f"read from source level {level}"
    nuclear_src = "user provided" if nuclear_marker else "auto-inferred"
    if not quiet:
        logger.info(
            "  %s: level %d, shape (c,y,x)=(%d, %d, %d), mpp %.4g (%s); "
            "nuclear=%s (%s); slide id %s -> %s",
            path.name,
            level,
            shape_cyx[0],
            shape_cyx[1],
            shape_cyx[2],
            resolved_mpp,
            mpp_source,
            markers[nuclear_idx],
            nuclear_src,
            meta.get("slide_id") or "(none)",
            out_path.name,
        )

    return CoralSlide.open(out_path)
