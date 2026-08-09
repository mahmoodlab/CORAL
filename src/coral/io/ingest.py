"""Canonical ingest — read an image, standardize it, write an OME-Zarr.

``convert_to_canonical`` turns a real multiplex image into a canonical
``(c, y, x)`` OME-Zarr slide store: it reads the source, reorders the
axes to the canonical layout, resolves each channel's marker name, and
records the result.

Marker names are resolved once per cohort via ``resolve_cohort_markers``,
which matches each raw channel name against the marker registry by exact
key. Matching is backed by a per-job marker map the user can review and
edit (each row carries a canonical name, a NOVEL marker token, or a
blank to fix). The resolved names and their match level are stored in
each slide; editing the map and re-running re-applies the names with no
pixel re-read (``apply_marker_names``).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from coral.dtypes import validate_image_dtype
from coral.io.harmonize import CanonicalChannel, harmonize_to_canonical
from coral.io.readers import (
    extract_channel_names,
    read_channel_tiff_dir,
    read_ometiff,
)
from coral.markers.marker_map import (
    RESOLVED_TOKEN,
    REVIEW_TOKEN,
    apply_blank_drop,
    apply_hoechst_drop,
    apply_panel_to_keep,
    build_marker_map,
    read_marker_map,
    resolve_marker_map,
    validate_and_flag,
    write_marker_map,
)
from coral.markers.normalize import (
    MatchStatus,
    clean_marker_name,
    match_marker,
)
from coral.markers.registry import canonical_key_index
from coral.slide.core import CoralSlide
from coral.slide.state import SlideMeta, default_state, load_state, save_state
from coral.slide.structure import write_structure_map
from coral.tissue.infer import infer_dapi_index, resolve_nuclear_index
from coral.utils.time import now_iso

logger = logging.getLogger(__name__)

__all__ = ["convert_to_canonical", "resolve_cohort_markers"]

#: Resolved markers: ``{raw_observed_name: (resolved_name, match, keep)}``.
Resolution = dict[str, tuple[str, MatchStatus, bool]]

# Level-0 zarr chunk tile — the store is chunked (1, _CHUNK_TILE,
# _CHUNK_TILE). Larger tiles cut ingest time (fewer chunks to compress +
# fewer files to write) but amplify per-patch reads (a patch read
# decompresses its whole chunk). 1024 is the measured balance: ~40%
# faster ingest than 512, with only a modest patch-read cost.
_CHUNK_TILE = 1024

# Integer factor the nuclear thumbnail is downsampled by. The thumbnail is
# a review-only preview (not a data product), so it renders at
# 1/_THUMBNAIL_DOWNSAMPLE resolution — fewer pixels to percentile-stretch
# and PNG-encode. 4× cuts that cost ~4× (~1.7s → ~0.4s) and the file ~4×
# (~12MB → ~3MB) on a large core, while staying a legible preview.
_THUMBNAIL_DOWNSAMPLE = 4

# Plausible microns-per-pixel band for multiplex imaging; _resolve_mpp
# warns outside it. A wrong mpp silently mis-scales every micron-based
# step (tissue morphology, scalebars).
_MPP_PLAUSIBLE_MIN = 0.1
_MPP_PLAUSIBLE_MAX = 1.0


def _read_and_harmonize(
    path: Path,
) -> tuple[np.ndarray, list[CanonicalChannel], float | None, dict[str, Any]]:
    """Read a source image and reorder it to canonical ``(c, y, x)``.

    A directory of per-channel image files is read as a stack; a single
    image file is read whole. The reader reports the source axis order,
    which is then used to reorder the pixels into the canonical
    ``(channel, y, x)`` layout.

    Args:
        path: Source image — either a directory of per-channel image
            files, or a single image file.

    Returns:
        ``(canonical_image, channels, source_mpp, source_meta)``: the
        ``(c, y, x)`` array, its per-channel metadata, the
        microns-per-pixel reported by the reader (may be ``None``), and
        the reader's provenance dict.
    """
    if path.is_dir():
        image, channels, mpp, meta = read_channel_tiff_dir(path)
    else:
        image, channels, mpp, meta = read_ometiff(path)
    canonical, canonical_channels = harmonize_to_canonical(
        image, channels, meta["axes"]
    )
    return canonical, canonical_channels, mpp, meta


def _input_channel_names(path: Path) -> list[str | None]:
    """Raw channel names for one input — names only, no pixel decode.

    A single image file uses its embedded channel names; a directory of
    per-channel files uses the sorted file names. Used by
    :func:`resolve_cohort_markers` to gather a cohort's marker names
    cheaply, before any heavy pixel read.
    """
    if path.is_dir():
        return [f.stem for f in sorted(path.glob("*.tif*"))]
    return list(extract_channel_names(path))


def _per_channel_dir_hint(inputs: list[Path]) -> list[str]:
    """Marker stems when ``inputs`` look like one slide's per-channel files.

    A directory of one-marker-per-file images (each a single-channel TIFF
    named for its marker) is easily handed to ``--image-dir`` by mistake:
    each file is then read as its own slide, so no per-channel stack is
    built and the filenames — the only marker names present — are ignored.
    When most of the file inputs' stems match the marker registry, that is
    a strong signal of this mistake. Returns the matched stems (names
    only, no pixel read) so the caller can point the user at the fix, or
    an empty list when the pattern does not hold.

    Args:
        inputs: The image inputs found under ``--image-dir``.

    Returns:
        The registry-matching file stems when the inputs look like a
        mistaken per-channel directory; otherwise an empty list.

    Example:
        A directory whose files are ``CD4.tiff``, ``CD8.tiff``,
        ``DAPI.tiff`` returns ``["CD4", "CD8", "DAPI"]`` — a signal to
        point ``--image-dir`` at the parent folder instead.
    """
    key_index = canonical_key_index()
    stems = [_output_stem(p) for p in inputs if p.is_file()]
    hits = [
        s for s in stems if match_marker(s, key_index)[1] == RESOLVED_TOKEN
    ]
    if len(hits) >= 2 and len(hits) * 2 >= len(stems):
        return hits
    return []


def _parse_channel_names(path: str | Path) -> list[str]:
    """Parse a channel-names file: one marker name per line.

    A plain sidecar text file listing the marker names in channel order,
    one per line. Use it when the source's embedded channel names are
    missing or wrong. Blank lines are dropped and surrounding whitespace
    is stripped.

    Args:
        path: Text file with one marker name per channel, in order.

    Returns:
        The marker names, in channel order.

    Example:
        A file with the lines ``DAPI``, ``CD3``, a blank line, then
        ``CD8`` parses to ``["DAPI", "CD3", "CD8"]``.
    """
    lines = Path(path).read_text().splitlines()
    return [s.strip() for s in lines if s.strip()]


def _apply_channel_names(
    channels: list[CanonicalChannel], names: list[str]
) -> list[CanonicalChannel]:
    """Override each channel's raw marker name from a supplied list.

    For sources whose embedded channel names are missing or wrong, the
    user supplies one name per channel, in channel order; these replace
    the embedded raw names before normalization and nuclear inference
    run. The list length must equal the channel count.

    Args:
        channels: Canonical channels in ``(c, y, x)`` channel order.
        names: One marker name per channel, in channel order.

    Returns:
        The same list, each channel's ``raw`` set to its override.

    Raises:
        ValueError: If ``len(names)`` does not match the channel count.

    Example:
        >>> from coral.io.harmonize import CanonicalChannel
        >>> chs = [CanonicalChannel(raw=None) for _ in range(2)]
        >>> [c.raw for c in _apply_channel_names(chs, ["DAPI", "CD3"])]
        ['DAPI', 'CD3']
    """
    if len(names) != len(channels):
        raise ValueError(
            f"--channel-names has {len(names)} name(s) but the source has "
            f"{len(channels)} channel(s); the counts must match"
        )
    for channel, name in zip(channels, names, strict=True):
        channel.raw = name
    return channels


def _parse_metadata_csv(path: str | Path) -> dict[str, float]:
    """Parse a per-image microns-per-pixel CSV.

    The CSV has two columns, ``image`` and ``mpp``. ``image`` is the
    exact entry name as it appears in the image directory — just the
    name with its extension, not a path. ``mpp`` is that image's
    microns-per-pixel.

    Args:
        path: CSV file with at least ``image`` and ``mpp`` columns.

    Returns:
        Mapping of image entry name to microns-per-pixel.

    Raises:
        ValueError: If the required columns are missing.

    Example:
        A CSV whose rows are ``case1.tif,0.5`` and ``case2.tif,0.325``
        parses to ``{"case1.tif": 0.5, "case2.tif": 0.325}``.
    """
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "image" not in fields or "mpp" not in fields:
            raise ValueError(
                f"metadata CSV must have 'image' and 'mpp' columns, "
                f"got {fields}"
            )
        return {row["image"]: float(row["mpp"]) for row in reader}


def _resolve_mpp(
    image_name: str,
    source_mpp: float | None,
    mpp_map: dict[str, float] | None = None,
    mpp_global: float | None = None,
) -> float:
    """Resolve a slide's microns-per-pixel by precedence.

    The first available value wins: a per-image CSV entry, then the
    value found in the source, then a single global fallback. If none is
    available, an error is raised.

    Args:
        image_name: Image entry name — the per-image CSV key.
        source_mpp: Microns-per-pixel found in the source (may be None).
        mpp_map: Per-image overrides keyed by image entry name.
        mpp_global: Single fallback value applied to every image.

    Returns:
        The resolved microns-per-pixel.

    Raises:
        ValueError: If no microns-per-pixel is available from any source.
    """
    if mpp_map and image_name in mpp_map:
        value = mpp_map[image_name]
    elif source_mpp is not None:
        value = source_mpp
    elif mpp_global is not None:
        value = mpp_global
    else:
        raise ValueError(
            f"no mpp for {image_name!r}: the source carries none and "
            f"neither --mpp nor a --mpp-csv entry was provided"
        )
    if not _MPP_PLAUSIBLE_MIN <= value <= _MPP_PLAUSIBLE_MAX:
        logger.warning(
            "%s: resolved mpp %.4g um/px is outside the plausible range "
            "[%.2g, %.2g] — check the source metadata or --mpp; "
            "morphology + scalebars depend on it.",
            image_name,
            value,
            _MPP_PLAUSIBLE_MIN,
            _MPP_PLAUSIBLE_MAX,
        )
    return value


def _apply_resolution(
    channels: list[CanonicalChannel],
    resolution: Resolution,
    *,
    set_keep: bool = True,
) -> list[str]:
    """Set each channel's resolved ``marker`` and ``match`` (+ ``keep``).

    Looks up each channel's raw observed name in ``resolution`` and stores
    the resolved name in lower case and its match level. The ``keep`` flag
    (the analysis panel) is defined by **membership** in the kept-only
    marker map:

    - ``set_keep=True`` (fresh ingest): a channel present in ``resolution``
      takes its keep flag; a channel **absent** (not in the kept panel — a
      QC / excluded / unnamed channel) is excluded (``keep=False``).
    - ``set_keep=False`` (guardrail re-apply): ``keep`` is frozen — a
      channel present in ``resolution`` updates its marker/match only, and
      a channel absent from the (kept-only) map is left entirely untouched
      (name and keep preserved).

    Returns the flat list of resolved marker names that downstream stages
    and encoders read.
    """
    markers: list[str] = []
    for i, ch in enumerate(channels):
        if ch.raw is not None and ch.raw in resolution:
            resolved, level, keep = resolution[ch.raw]
            ch.marker = clean_marker_name(resolved)
            ch.match = level
            if set_keep:
                ch.keep = keep
        elif set_keep:
            # Fresh ingest, channel absent from the kept panel -> excluded.
            ch.marker = clean_marker_name(ch.raw) if ch.raw else f"channel_{i}"
            ch.match = REVIEW_TOKEN
            ch.keep = False
        # else (guardrail, absent): leave the frozen name + keep untouched.
        markers.append(ch.marker if ch.marker is not None else f"channel_{i}")
    return markers


def _warn_heterogeneous_channels(
    per_slide: dict[str, set[str]], cohort_names: set[str]
) -> None:
    """Warn per slide that is missing channels present elsewhere.

    Compares each slide's observed channel names against the cohort-wide
    set; a slide short of the union gets one warning listing the missing
    names, so a folder that dropped a channel is surfaced up front rather
    than silently ingested with fewer markers. A no-op when every slide
    carries the full set. Names only — no pixel read.
    """
    for name in sorted(per_slide):
        missing = cohort_names - per_slide[name]
        if missing:
            logger.warning(
                "  %s: missing %d marker(s) present elsewhere in the "
                "cohort: %s",
                name,
                len(missing),
                ", ".join(sorted(missing)),
            )


def resolve_cohort_markers(
    inputs: list[Path],
    output_dir: str | Path,
    *,
    channel_names: list[str] | None = None,
    channels: Any = None,  # noqa: ANN401 — a Selection or None
    keep_hoechst: bool = False,
    nuclear_marker: str | None = None,
) -> tuple[Resolution, int]:
    """Resolve a cohort's marker names against the registry and job map.

    Gathers the raw marker names across all ``inputs`` (names only, no
    pixel decode), then either loads or builds the marker map in the
    output directory. When a map is already present it is the user's
    reviewed copy and is used as-is (after validation). When it is
    absent, one is built by matching each name against the registry and
    written out for the user to review. Returns the per-name resolution
    and the number of markers still needing review.

    Args:
        inputs: Slide sources — image files and/or per-channel
            directories.
        output_dir: The output directory holding the marker map and the
            written slide stores.
        channel_names: One name per channel, applied to every input;
            ``None`` reads each input's embedded names.
        channels: Optional channel ``Selection`` (from ``--subset``) whose
            include/exclude globs seed the ``keep`` column (overwriting any
            prior keep).
        keep_hoechst: Keep Hoechst channels (default drops them).
        nuclear_marker: Explicit nuclear marker; a named Hoechst is kept.

    Returns:
        ``(resolution, n_needs_review)`` — ``resolution`` maps each raw
        observed name to ``(resolved_name, match_level)``.

    Raises:
        ValueError: If no marker names are available from any input (no
            embedded names and no ``--channel-names``).

    Example:
        Build or load a cohort marker map before ingesting slides::

            from pathlib import Path
            from coral.io.ingest import resolve_cohort_markers

            resolution, n_review = resolve_cohort_markers(
                [Path("case1.tif")], "out/"
            )
    """
    out = Path(output_dir)
    key_index = canonical_key_index()
    existing = read_marker_map(out)
    # Channels in one image (the homogeneous panel), captured from the
    # first slide so the log reports channels vs unique names honestly,
    # without multiplying by the same panel repeated on every slide.
    n_channels = 0
    seeded = False  # did we (re)seed the keep column this run?
    raw_names: list[str] = []
    # Each source's observed channel names, kept for the heterogeneity
    # warning below. Only populated when names come from the sources (not
    # --channel-names, which forces the same set on every slide). Both are
    # empty on the rerun path (existing map → no source read).
    per_slide: dict[str, set[str]] = {}
    if existing is not None:
        # A present map is authoritative (the user's edited copy); a rerun
        # never re-reads the sources for names. A mapping the user forced
        # but got wrong is reset to REVIEW before the error is raised.
        validate_and_flag(existing, set(key_index.values()), out)
        map_df = existing
    else:
        for item in inputs:
            try:
                names = channel_names or _input_channel_names(Path(item))
            except Exception as exc:  # noqa: BLE001
                # A source we can't read names from fails with a clear
                # per-slide error in the ingest loop — don't let it sink
                # the whole cohort's marker map.
                logger.debug("skipping %s for name gather: %s", item, exc)
                continue
            kept = [n for n in names if n]
            if channel_names is None:
                per_slide[Path(item).name] = set(kept)
            if not raw_names:  # first slide sets the per-image channel count
                n_channels = len(kept)
            raw_names.extend(kept)
        if not raw_names:
            hint = _per_channel_dir_hint(inputs)
            if hint:
                shown = ", ".join(hint[:5])
                more = "" if len(hint) <= 5 else ", ..."
                raise ValueError(
                    f"no marker names found, but {len(hint)} of the inputs "
                    f"are single files whose names match known markers "
                    f"({shown}{more}). This looks like the per-channel "
                    f"images of ONE slide — point --image-dir at the PARENT "
                    f"folder so this directory is read as one per-channel "
                    f"image. If these really are separate slides, pass "
                    f"--channel-names <file>."
                )
            raise ValueError(
                "no marker names found: the source(s) carry no channel "
                "names and --channel-names was not given. Provide one "
                "marker name per channel via --channel-names <file>."
            )
        map_df = build_marker_map(raw_names, key_index)
        seeded = True

    if channels is not None:
        # The --subset channel selection seeds the keep column (overwriting
        # any prior keep) so it persists into every store and drives
        # downstream selection.
        apply_panel_to_keep(map_df, channels)
        seeded = True

    dropped_blank: list[str] = []
    dropped_hoechst: list[str] = []
    if seeded:
        # Blank/empty + Hoechst QC channels are dropped (keep=no) by
        # default; these run after the channel selection so a bare '*'
        # doesn't resurrect them.
        dropped_blank = apply_blank_drop(map_df, channels=channels)
        dropped_hoechst = apply_hoechst_drop(
            map_df,
            keep_hoechst=keep_hoechst,
            channels=channels,
            nuclear_marker=nuclear_marker,
        )
        out.mkdir(parents=True, exist_ok=True)
        # Persist only the kept analysis panel; excluded / QC rows live in
        # each store's frozen .zattrs (via the full resolution below), not
        # the editable CSV.
        kept_mask = [
            str(k).strip().lower() != "no" for k in map_df["keep_for_analysis"]
        ]
        write_marker_map(map_df.loc[kept_mask], out)

    # Resolve the FULL map (incl. excluded/QC) so .zattrs records every
    # channel's keep; the CSV above carries only the kept rows.
    resolution = resolve_marker_map(map_df)
    n_unique = len(resolution)
    # Channels in one image; on a rerun no sources are read, so fall back
    # to the unique count from the persisted map.
    if not n_channels:
        n_channels = n_unique
    n_duplicates = max(0, n_channels - n_unique)
    n_review = sum(
        1 for _, lvl, _ in resolution.values() if lvl == REVIEW_TOKEN
    )

    # One structured block: what was read, whether it all mapped, and what
    # was auto-excluded.
    logger.info("")
    logger.info("🏷️  Markers:")
    if n_duplicates:
        logger.info(
            "  read %d channels (%d unique names, %d duplicates)",
            n_channels,
            n_unique,
            n_duplicates,
        )
    else:
        logger.info(
            "  read %d channels (%d unique names)", n_channels, n_unique
        )
    review_names = [
        original
        for original, (_, level, _) in resolution.items()
        if level == REVIEW_TOKEN
    ]
    if review_names:
        logger.warning(
            "  %d marker(s) could not be auto-mapped to the registry "
            "and need review:",
            len(review_names),
        )
        for name in review_names:
            logger.warning(
                "    [red]%s[/red]", name, extra={"no_highlight": True}
            )
    else:
        logger.info(
            "  all %d marker(s) auto-mapped to canonical names. "
            "No manual review needed.",
            n_unique,
        )
    _warn_heterogeneous_channels(per_slide, set(raw_names))
    if dropped_blank:
        logger.info(
            "  auto-excluded %d blank/empty channel(s)", len(dropped_blank)
        )
    if dropped_hoechst:
        logger.info(
            "  auto-excluded %d Hoechst channel(s)", len(dropped_hoechst)
        )
    if any(not k for _, _, k in resolution.values()):
        logger.info(
            "  All excluded markers are stored in .zarr. Include them by "
            "providing --subset."
        )
    logger.info("")
    return resolution, n_review


def _resolve_nuclear_channel(
    image_name: str,
    markers: list[str],
    nuclear_marker: str | None,
    *,
    kept: list[bool] | None = None,
) -> int:
    """Resolve the nuclear channel index for a panel, or raise.

    CORAL assumes every slide has a nuclear stain — it anchors tissue
    detection, cell segmentation, and the overlay backdrops. An explicit
    ``nuclear_marker`` wins; otherwise the stain is inferred by name. If
    neither finds a nuclear channel, ingest stops with a clear error
    rather than letting a slide with no nuclear stain flow silently into
    the later stages.

    Args:
        image_name: Image entry name, for the error message.
        markers: Resolved marker names, in channel order.
        nuclear_marker: Explicit override name, or ``None`` to auto-infer.
        kept: Per-channel keep flags; when given, auto-inference prefers a
            kept nuclear before falling back to all channels.

    Returns:
        The nuclear channel index.

    Raises:
        ValueError: If ``nuclear_marker`` names no channel on the panel,
            or no nuclear channel can be identified at all.

    Example:
        >>> _resolve_nuclear_channel("slide", ["DAPI", "CD3"], None)
        0
        >>> _resolve_nuclear_channel("slide", ["PanCK", "CD3"], "CD3")
        1
    """
    if nuclear_marker is not None:
        return resolve_nuclear_index(markers, nuclear_marker)
    if kept is not None:
        # Prefer a KEPT nuclear (so dropping Hoechst makes DRAQ5/DAPI the
        # nuclear, not a now-excluded Hoechst). The ingest pre-flight
        # guarantees a kept nuclear exists when we reach here.
        kept_idxs = [i for i, k in enumerate(kept) if k]
        rel = infer_dapi_index([markers[i] for i in kept_idxs])
        if rel is not None:
            return kept_idxs[rel]
    idx = infer_dapi_index(markers)
    if idx is None:
        unique = list(dict.fromkeys(markers))
        preview = ", ".join(unique[:6]) + ("…" if len(unique) > 6 else "")
        raise ValueError(
            f"{image_name}: no nuclear stain found among {len(markers)} "
            f"channels ({len(unique)} unique: {preview}). Pass "
            f"--channel-names to supply real marker names, or "
            f"--nuclear-marker to name the nuclear channel."
        )
    return idx


# NGFF omero display defaults — do not scan pixels. Window end is a
# conservative uint16 fluorescence display range (IDR-style); QuPath's
# B&C dialog can still auto-adjust. Nuclear channel starts active only.
_OMERO_WINDOW_MAX = 65535.0
_OMERO_WINDOW_END = 4096.0
_OMERO_COLORS = (
    "0000FF",  # blue — typical nuclear
    "00FF00",
    "FF0000",
    "FFFF00",
    "00FFFF",
    "FF00FF",
    "FFFFFF",
    "FFA500",
)

# QuPath/Bio-Formats read channel names + PhysicalSize from this sidecar;
# the transitional ``omero`` block alone is parsed but not applied to the
# MetadataStore (see ome/ZarrReader.parseOmeroMetadata).
_OME_XML_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _ome_pixel_type(dtype: np.dtype) -> tuple[str, int]:
    """Map a numpy dtype to OME Pixels Type + SignificantBits."""
    kind = np.dtype(dtype)
    if kind == np.uint8:
        return "uint8", 8
    if kind == np.uint16:
        return "uint16", 16
    if kind == np.uint32:
        return "uint32", 32
    if kind == np.int8:
        return "int8", 8
    if kind == np.int16:
        return "int16", 16
    if kind == np.int32:
        return "int32", 32
    if kind == np.float32:
        return "float", 32
    if kind == np.float64:
        return "double", 64
    return "uint16", 16


def _write_ome_metadata_xml(
    store_path: Path,
    markers: list[str],
    mpp: float,
    shape_cyx: tuple[int, int, int],
    dtype: np.dtype,
) -> None:
    """Write ``OME/METADATA.ome.xml`` for Bio-Formats / QuPath.

    Channel ``Name`` and ``PhysicalSizeX/Y`` are what QuPath surfaces as
    channel names and micron pixel size. Pixels are not touched.
    """
    from xml.sax.saxutils import escape

    size_c, size_y, size_x = (int(v) for v in shape_cyx)
    pix_type, sig_bits = _ome_pixel_type(dtype)
    mpp_f = float(mpp)
    channels_xml = "\n".join(
        f'      <Channel ID="Channel:0:{i}" Name="{escape(str(name))}" '
        f'SamplesPerPixel="1"/>'
        for i, name in enumerate(markers)
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<OME xmlns="{_OME_XML_NS}" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="'
        f"{_OME_XML_NS} "
        f'{_OME_XML_NS}/ome.xsd">\n'
        '  <Image ID="Image:0" Name="0">\n'
        '    <Pixels BigEndian="false" DimensionOrder="XYCZT" '
        'ID="Pixels:0" Interleaved="false" '
        f'SignificantBits="{sig_bits}" '
        f'PhysicalSizeX="{mpp_f}" PhysicalSizeXUnit="µm" '
        f'PhysicalSizeY="{mpp_f}" PhysicalSizeYUnit="µm" '
        f'SizeC="{size_c}" SizeT="1" SizeX="{size_x}" '
        f'SizeY="{size_y}" SizeZ="1" Type="{pix_type}">\n'
        f"{channels_xml}\n"
        f'      <TiffData FirstC="0" FirstT="0" FirstZ="0" '
        f'PlaneCount="{size_c}"/>\n'
        "    </Pixels>\n"
        "  </Image>\n"
        "</OME>\n"
    )
    ome_dir = store_path / "OME"
    ome_dir.mkdir(parents=True, exist_ok=True)
    (ome_dir / ".zgroup").write_text('{"zarr_format": 2}\n')
    (ome_dir / "METADATA.ome.xml").write_text(xml)


def _write_ngff_external_attrs(
    root: Any,
    markers: list[str],
    mpp: float,
    *,
    nuclear_marker: str | None = None,
) -> None:
    """Write additive OME-NGFF 0.4 attrs for external readers (QuPath).

    Sets ``multiscales`` (axes ``c,y,x`` + level-0 scale), ``omero``
    (labels / display), and ``OME/METADATA.ome.xml`` (channel names +
    PhysicalSize for Bio-Formats). Does not touch CORAL custom attrs or
    pixel data.

    Args:
        root: Open zarr group for the slide store.
        markers: Display labels, one per channel axis.
        mpp: Isotropic microns-per-pixel for the y/x scale transform.
        nuclear_marker: Channel label to mark ``active`` in omero (others
            off). When ``None``, channel 0 is active.
    """
    mpp_f = float(mpp)
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [1.0, mpp_f, mpp_f],
                        },
                    ],
                }
            ],
        }
    ]
    nuclear_key = (nuclear_marker or "").strip().lower()
    active_idx = 0
    if nuclear_key:
        for i, name in enumerate(markers):
            if str(name).strip().lower() == nuclear_key:
                active_idx = i
                break
    root.attrs["omero"] = {
        "channels": [
            {
                "label": name,
                "color": _OMERO_COLORS[i % len(_OMERO_COLORS)],
                "active": i == active_idx,
                "coefficient": 1.0,
                "family": "linear",
                "inverted": False,
                "window": {
                    "min": 0.0,
                    "max": _OMERO_WINDOW_MAX,
                    "start": 0.0,
                    "end": _OMERO_WINDOW_END,
                },
            }
            for i, name in enumerate(markers)
        ]
    }

    # Bio-Formats / QuPath: names + micron scale come from OME-XML.
    if "0" in root:
        arr = root["0"]
        store_path = Path(str(root.store.path))
        shape = tuple(int(s) for s in arr.shape)
        if len(shape) != 3:
            logger.warning(
                "skip OME/METADATA.ome.xml: expected (c,y,x) array, got %s",
                shape,
            )
        else:
            _write_ome_metadata_xml(
                store_path,
                markers,
                mpp_f,
                shape_cyx=(shape[0], shape[1], shape[2]),
                dtype=np.dtype(arr.dtype),
            )


def _write_canonical_zarr(
    out_path: Path,
    image: np.ndarray,
    markers: list[str],
    channels: list[CanonicalChannel],
    mpp: float,
    source_pyramid_levels: int,
    nuclear_marker: str | None = None,
) -> int:
    """Write a canonical ``(c, y, x)`` image to an OME-Zarr store.

    Writes the base level (dataset ``"0"``) plus slide attributes:
    ``channels`` (the single source of truth — lower-case
    ``{marker, raw, match}`` per channel), ``nuclear_channel`` (the
    nuclear stain by name), ``mpp``, and ``source_pyramid_levels``.
    Also writes additive NGFF 0.4 ``multiscales`` + ``omero``.

    Only level 0 is written; multi-resolution pyramid generation is not
    yet implemented. When the source carried more pyramid levels a
    warning is logged.

    Args:
        out_path: Destination ``.zarr`` directory (created/overwritten).
        image: Canonical ``(c, y, x)`` array.
        markers: Resolved marker names, one per channel.
        channels: Per-channel metadata, one per channel axis.
        mpp: Resolved microns-per-pixel of the base level.
        source_pyramid_levels: Pyramid level count from the reader.
        nuclear_marker: Explicit nuclear-channel marker name, or ``None``
            to auto-infer.

    Returns:
        The resolved nuclear channel index (also stored in the attrs).

    Raises:
        ValueError: If no nuclear channel can be identified.
    """
    height, width = int(image.shape[1]), int(image.shape[2])
    chunks = (1, min(_CHUNK_TILE, height), min(_CHUNK_TILE, width))

    # Resolve + validate the nuclear channel BEFORE writing anything, so
    # a nuclear-less panel (or a bad --nuclear-marker) fails cleanly
    # without leaving a partial store on disk.
    nuclear_idx = _resolve_nuclear_channel(
        out_path.name,
        markers,
        nuclear_marker,
        kept=[bool(ch.keep) for ch in channels],
    )

    root = zarr.open_group(str(out_path), mode="w")
    root.create_dataset("0", data=image, chunks=chunks)
    root.attrs["channels"] = [ch.model_dump() for ch in channels]
    # The nuclear stain is recorded by NAME so downstream stages read it
    # without re-inferring (see CoralSlide.nuclear_channel).
    root.attrs["nuclear_channel"] = markers[nuclear_idx]
    root.attrs["mpp"] = mpp
    root.attrs["source_pyramid_levels"] = source_pyramid_levels
    # Prefer resolved marker; fall back to raw so QuPath never sees a blank.
    omero_labels = [
        (m if m else (ch.raw or f"channel_{i}"))
        for i, (m, ch) in enumerate(zip(markers, channels, strict=True))
    ]
    _write_ngff_external_attrs(
        root,
        omero_labels,
        mpp,
        nuclear_marker=markers[nuclear_idx],
    )
    if source_pyramid_levels > 1:
        logger.warning(
            "%s: source has %d pyramid levels; CORAL stores level 0 only "
            "(pyramid materialization is not yet implemented).",
            out_path.name,
            source_pyramid_levels,
        )
    return nuclear_idx


def _downsample_mean(channel: np.ndarray, factor: int) -> np.ndarray:
    """Box-average a 2-D channel down by an integer ``factor``.

    Averages each ``factor × factor`` block into one pixel — unlike
    strided decimation (``channel[::factor, ::factor]``), which just drops
    pixels and can alias fine texture. Edge rows/columns that do not fill a
    whole block are dropped (at most ``factor - 1`` of each).

    Args:
        channel: 2-D intensity array.
        factor: Integer downsample factor; ``<= 1`` returns the input.

    Returns:
        The downsampled 2-D array, same dtype as ``channel``.

    Example:
        >>> import numpy as np
        >>> a = np.array([[0, 0, 4, 4], [0, 0, 4, 4]], dtype="uint16")
        >>> _downsample_mean(a, 2).tolist()
        [[0, 4]]
    """
    if factor <= 1:
        return channel
    h = channel.shape[0] // factor * factor
    w = channel.shape[1] // factor * factor
    blocks = channel[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return blocks.mean(axis=(1, 3)).astype(channel.dtype)


def _write_nuclear_thumbnail(out_path: Path, nuclear: np.ndarray) -> None:
    """Render the nuclear channel to ``thumbnails/nuclear.png`` in a store.

    A percentile-stretched grayscale preview of the resolved nuclear
    channel at ``1 / _THUMBNAIL_DOWNSAMPLE`` level-0 resolution (4×),
    written as a sidecar PNG inside the ``.zarr`` directory. Lets the user
    eyeball the nuclear stain (the backbone of tissue + cell detection)
    without opening the full store.

    Args:
        out_path: The ``.zarr`` store directory.
        nuclear: The 2-D nuclear channel at full resolution.
    """
    from PIL import Image

    from coral.viz import normalize_uint8

    # Area-average down before rendering — this is a review preview, not a
    # data product. Fewer pixels cut the normalize + PNG-encode cost ~16×
    # at 4× (≈7s → ≈0.4s) and the file likewise, and box-averaging keeps
    # the preview free of decimation aliasing.
    small = _downsample_mean(np.asarray(nuclear), _THUMBNAIL_DOWNSAMPLE)
    gray = normalize_uint8(small)
    thumb_dir = out_path / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(thumb_dir / "nuclear.png")


def apply_marker_names(
    zarr_path: str | Path,
    resolution: Resolution,
    *,
    nuclear_marker: str | None = None,
) -> None:
    """Re-apply resolved marker names to an existing store — no pixel read.

    The cheap rerun: reads the store's ``channels`` attribute, re-resolves
    each channel's ``marker`` and ``match`` from ``resolution`` (the
    user's edited map), re-infers the nuclear channel (stored by name),
    and rewrites only the attributes. The full source pixels are never
    re-read; only the nuclear channel is read, and only if it moved, to
    refresh the thumbnail.

    Args:
        zarr_path: An existing ``.zarr`` store.
        resolution: ``{raw_name: (resolved, level)}`` from
            :func:`resolve_cohort_markers`.
        nuclear_marker: Optional nuclear-channel override.

    Example:
        Re-apply an edited marker map without re-reading pixels::

            from coral.io.ingest import apply_marker_names

            apply_marker_names("out/case1.zarr", resolution)
    """
    path = Path(zarr_path)
    root = zarr.open_group(str(path), mode="a")
    channels = [CanonicalChannel(**d) for d in root.attrs["channels"]]
    old_nuclear = root.attrs.get("nuclear_channel")
    # Guardrail re-apply: names/match only — keep is frozen at ingest.
    markers = _apply_resolution(channels, resolution, set_keep=False)
    nuclear_idx = _resolve_nuclear_channel(
        path.name,
        markers,
        nuclear_marker,
        kept=[bool(ch.keep) for ch in channels],
    )
    nuclear_name = markers[nuclear_idx]
    root.attrs["channels"] = [ch.model_dump() for ch in channels]
    root.attrs["nuclear_channel"] = nuclear_name
    # Refresh NGFF labels to match re-resolved markers; scale uses the
    # store's existing mpp (unchanged by this path). Prefer resolved
    # marker; fall back to raw so QuPath never sees a blank.
    omero_labels = [
        (m if m else (ch.raw or f"channel_{i}"))
        for i, (m, ch) in enumerate(zip(markers, channels, strict=True))
    ]
    _write_ngff_external_attrs(
        root,
        omero_labels,
        float(root.attrs["mpp"]),
        nuclear_marker=nuclear_name,
    )
    logger.debug("%s: re-applied marker names (no pixel re-read)", path.name)

    # Refresh the thumbnail only if the nuclear channel moved (or none
    # exists yet) — reads just that one channel from the store, never the
    # source pixels.
    thumb = path / "thumbnails" / "nuclear.png"
    if nuclear_name != old_nuclear or not thumb.exists():
        _write_nuclear_thumbnail(path, np.asarray(root["0"][nuclear_idx]))


def _output_stem(path: Path) -> str:
    """Output store stem — the input name minus its image extension."""
    name = path.name
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def convert_to_canonical(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    resolution: Resolution | None = None,
    mpp: float | None = None,
    mpp_map: dict[str, float] | None = None,
    channel_names: list[str] | None = None,
    nuclear_marker: str | None = None,
    quiet: bool = False,
) -> CoralSlide:
    """Ingest one image into a canonical OME-Zarr slide store.

    Ingests a single slide; to process a cohort, call this once per
    slide. Reads the source image (a single file, or a directory of
    per-channel files), reorders the axes to the canonical
    ``(c, y, x)`` layout, applies the already-resolved cohort marker
    names to each channel, and resolves the slide's microns-per-pixel
    (preferring a per-image override, then the value found in the source,
    then a global fallback). It writes a canonical OME-Zarr store at
    ``<output_dir>/<name>.zarr`` (the pixels plus channel, nuclear,
    microns-per-pixel, and provenance metadata), records the ingest as
    completed in the slide's state file, and returns an open handle to
    the written store.

    If a completed store already exists it is left untouched and
    reopened — an edited marker map is re-applied automatically on the
    next stage command (no pixel re-read), or start a fresh output
    directory (or delete the store) to re-ingest from pixels. When
    ``resolution`` is omitted, marker names are resolved for this single
    image and a marker map is written to ``output_dir``.

    Args:
        input_path: Source image — a single image file, or a directory
            of per-channel image files.
        output_dir: Directory to write ``<name>.zarr`` and the marker
            map into.
        resolution: Cohort resolution from :func:`resolve_cohort_markers`,
            mapping each raw channel name to its resolved marker name and
            match level; built for this image alone when ``None``.
        mpp: Global microns-per-pixel fallback.
        mpp_map: Per-image microns-per-pixel overrides, keyed by image
            entry name.
        channel_names: Marker names overriding the source's embedded
            names; length must equal the channel count. ``None`` keeps
            the embedded names.
        nuclear_marker: Force the nuclear channel by marker name; ``None``
            auto-infers.
        quiet: If True, skip per-image progress log lines (shape / nuclear
            summary). The CLI leaves this False so each image prints its
            details, then ``[i/N] <name> ingested``.

    Returns:
        An open ``CoralSlide`` handle to the written canonical store.

    Raises:
        ValueError: If the source dtype is one CORAL cannot scale, or a
            float source is not normalised to ``[0, 1]`` (see
            :func:`coral.dtypes.validate_image_dtype`); if
            microns-per-pixel cannot be resolved; if ``channel_names``
            length differs from the channel count; if ``nuclear_marker``
            names no channel; or if no nuclear channel can be identified.

    Example:
        Ingesting one slide writes ``case1.zarr`` under the output
        directory and returns a slide whose ingest task is marked
        completed::

            slide = convert_to_canonical("case1.tif", "out/", mpp=0.5)
    """
    path = Path(input_path)
    out_path = Path(output_dir) / f"{_output_stem(path)}.zarr"

    if resolution is None:
        resolution, _ = resolve_cohort_markers(
            [path], output_dir, channel_names=channel_names
        )

    if out_path.exists():
        existing_state = load_state(out_path)
        if existing_state.tasks.ingest.status == "completed":
            if not quiet:
                logger.info("  %s: already ingested", path.name)
            return CoralSlide.open(out_path)

    started = now_iso()
    image, channels, source_mpp, meta = _read_and_harmonize(path)
    if channel_names is not None:
        _apply_channel_names(channels, channel_names)
    markers = _apply_resolution(channels, resolution)
    # Gate the pixels before anything is written, so an image CORAL
    # cannot scale fails cleanly instead of leaving a partial store that
    # mis-scales silently at extract time (same reason the nuclear
    # channel is resolved before the write in _write_canonical_zarr).
    validate_image_dtype(image, path.name, markers)
    resolved_mpp = _resolve_mpp(
        path.name, source_mpp, mpp_map=mpp_map, mpp_global=mpp
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nuclear_idx = _write_canonical_zarr(
        out_path,
        image,
        markers,
        channels,
        resolved_mpp,
        meta.get("pyramid_levels", 1),
        nuclear_marker=nuclear_marker,
    )
    _write_nuclear_thumbnail(out_path, image[nuclear_idx])

    state = default_state(out_path, image_path=str(path.resolve()))
    state.tasks.ingest.status = "completed"
    state.tasks.ingest.started_at = started
    state.tasks.ingest.completed_at = now_iso()
    state.tasks.ingest.outputs = {"image": "0"}
    state.meta = SlideMeta(
        dimensions=(int(image.shape[1]), int(image.shape[2])),
        n_markers=len(channels),
        mpp=resolved_mpp,
    )
    save_state(out_path, state)
    write_structure_map(out_path)

    if mpp_map and path.name in mpp_map:
        mpp_source = "read from metadata-csv"
    elif source_mpp is not None:
        mpp_source = "read from source"
    else:
        mpp_source = "read from --mpp"
    nuclear_src = "user provided" if nuclear_marker else "auto-inferred"
    # Show the EXACT marker_map name (the store keeps the cleaned marker).
    nuc_raw = channels[nuclear_idx].raw
    nuclear_name = (
        resolution[nuc_raw][0]
        if nuc_raw in resolution and resolution[nuc_raw][0]
        else markers[nuclear_idx]
    )
    if not quiet:
        logger.info(
            "  %s: shape (c,y,x)=(%d, %d, %d), mpp %.4g (%s); "
            "nuclear=%s (%s) -> %s",
            path.name,
            len(channels),
            int(image.shape[1]),
            int(image.shape[2]),
            resolved_mpp,
            mpp_source,
            nuclear_name,
            nuclear_src,
            out_path.name,
        )

    return CoralSlide.open(out_path)
