"""Reader for PerkinElmer/Akoya ``.qptiff`` whole-slide TMA scans.

A qptiff is a pyramidal, tiled, LZW-compressed BigTIFF written by Akoya
Fusion. ``tifffile`` exposes it as several series: ``Baseline`` (the real
image, axes ``CYX``, one page per channel per pyramid level) plus
``Thumbnail`` / ``Macro`` / ``Label`` overview images. Only ``Baseline``
is read here.

Marker names come from the per-page PerkinElmer XML. Each page carries
both a ``<Name>`` (the **fluorophore**: ``DAPI``, ``ATTO 550``, ``Cy5``,
which repeats every cycle) and a ``<Biomarker>`` (the **antibody target**:
``CD20``, ``Pax5``, ...). ``<Biomarker>`` is the one that maps 1:1 onto
channels, so that is what this reader returns; the fluorophore is kept in
the provenance dict.

Page 0 also carries a ``<ScanProfile>`` JSON describing the acquisition
plan (every well x every dye slot, including unimaged ``--`` placeholders
and per-cycle DAPI repeats). It has more entries than the file has saved
channels, so it is NOT used to name channels — it is parsed only as a
cross-check that the per-page biomarkers reconcile with the plan.

Returns ``(image, channels, mpp, source_meta)`` in the source's own
``(c, y, x)`` order; harmonization is the caller's job, as for the other
readers.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import tifffile

from coral.io.readers.ome_tiff import _parse_tiff_resolution
from coral.utils.errors import ReaderError

logger = logging.getLogger(__name__)

__all__ = [
    "extract_qptiff_channel_names",
    "qptiff_channel_iter",
    "read_qptiff",
    "read_qptiff_metadata",
    "read_qptiff_overviews",
    "read_qptiff_record",
    "scanprofile_markers",
]

QPTIFF_EXTENSIONS = (".qptiff",)

#: The tifffile series holding the real pyramidal image. The remaining
#: series are the scanner's Thumbnail / Macro / Label overviews.
_BASELINE_SERIES = "Baseline"

# The flat top-level elements are matched by regex rather than parsed:
# it is far cheaper than an XML parse of 19 kB per page (177 pages on a
# 6 GB slide), and the ScanProfile blob inside trips strict XML parsers on
# some writer versions. Deeper elements DO nest (Bands/Band/Cuton,
# CameraSettings/ROI/X), so the full record parses properly — see
# _description_tree, which strips the ScanProfile first.
_ELEMENT_RE = {
    name: re.compile(rf"<{name}>(.*?)</{name}>", re.DOTALL)
    for name in ("Name", "Biomarker", "SlideID", "ScanProfile", "SampleIsTMA")
}

#: ScanProfile marker slot that was planned but never imaged.
_SCANPROFILE_PLACEHOLDER = "--"

#: Bytes taken from each end of the source for its fingerprint.
_FINGERPRINT_EDGE = 1 << 20

#: Files already warned about for having no <Biomarker>. The metadata is
#: read three times per ingest (probe, marker resolution, then the write),
#: and the same seven-line warning three times per file trains people to
#: skip it. Keyed by resolved path, so two files still warn twice.
_WARNED_NO_BIOMARKER: set[str] = set()


def _element(description: str, name: str) -> str | None:
    """First value of a flat PerkinElmer XML element, or ``None``."""
    match = _ELEMENT_RE[name].search(description)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _sample_is_tma(description: str) -> bool | None:
    """Whether the scan profile says this slide is a TMA.

    Tri-state, and the three states are not two. ``None`` means the file makes
    no claim, which is evidence of nothing in either direction: it is not a
    TMA and it is not a section, it is unknown. Older Vectra writers and every
    non-PerkinElmer source land here, and real TMA scans are among them: a
    slide carrying no ``SampleIsTMA`` element at all can still be a TMA.

    So the only load-bearing value is ``False``: the operator said, at
    acquisition, that this slide is one whole tissue section. ``True`` and
    ``None`` are both just "not refused".

    Returns:
        ``True`` or ``False`` when the file states it, ``None`` when the
        element is absent. Never infer a TMA from ``None``.
    """
    raw = _element(description, "SampleIsTMA")
    if raw is None:
        return None
    return raw.strip().lower() == "true"


def _page_description(page: Any) -> str:  # noqa: ANN401 — TiffPage|TiffFrame
    """A page's ``ImageDescription`` string, or ``""`` when absent."""
    tag = page.tags.get("ImageDescription") if page.tags else None
    return str(tag.value) if tag is not None else ""


def _validate_path(path: str | Path) -> Path:
    """Boundary validation: exists, is a file, has a qptiff extension."""
    p = Path(path)
    if not p.exists():
        raise ReaderError(f"File not found: {p}")
    if p.is_dir():
        raise ReaderError(f"Path is a directory, not a qptiff file: {p}")
    if not p.name.lower().endswith(QPTIFF_EXTENSIONS):
        raise ReaderError(
            f"File extension not recognized as qptiff: {p.name}. "
            f"Expected one of: {QPTIFF_EXTENSIONS}"
        )
    return p


def _baseline_series(tif: tifffile.TiffFile) -> Any:  # noqa: ANN401
    """The pyramidal image series, ignoring the overview series.

    Picks the series named ``Baseline``; if the writer used another name,
    falls back to the first series with ``CYX`` axes, then to series 0.
    """
    for series in tif.series:
        if str(series.name) == _BASELINE_SERIES:
            return series
    for series in tif.series:
        if str(series.axes) == "CYX":
            logger.warning(
                "no %r series in this qptiff; using series %r (axes CYX)",
                _BASELINE_SERIES,
                series.name,
            )
            return series
    logger.warning(
        "no %r series in this qptiff; falling back to series 0 (axes %s)",
        _BASELINE_SERIES,
        tif.series[0].axes,
    )
    return tif.series[0]


def _select_level(series: Any, level: int) -> Any:  # noqa: ANN401
    """The requested pyramid level of ``series``, validated."""
    levels = list(series.levels) if series.levels else [series]
    if not 0 <= level < len(levels):
        raise ReaderError(
            f"pyramid level {level} out of range: this qptiff has "
            f"{len(levels)} level(s) (0 = full resolution)"
        )
    return levels[level]


def _channel_names(
    pages: list[Any], source: Path | None = None
) -> tuple[list[str | None], str]:
    """Per-channel marker names + which XML element they came from.

    Prefers ``<Biomarker>`` (the antibody target, unique per channel).
    Falls back to ``<Name>`` (the fluorophore, which repeats every cycle)
    only when no page carries a biomarker, so a file without biomarker
    annotation still ingests and its channels land in the marker map as
    REVIEW rows rather than being silently invented.

    The fallback warning fires once per file, not once per read: the
    metadata is read three times during an ingest and the same warning
    three times teaches people to skip it.

    Args:
        pages: The level's pages, in channel order.
        source: The file, used to warn once per file. ``None`` warns
            every call.

    Returns:
        ``(names, element_name)`` where element is ``"Biomarker"`` or
        ``"Name"``.
    """
    descriptions = [_page_description(page) for page in pages]
    biomarkers = [_element(d, "Biomarker") for d in descriptions]
    key = str(source.resolve()) if source is not None else None
    if any(biomarkers):
        missing = [i for i, b in enumerate(biomarkers) if b is None]
        if missing:
            logger.warning(
                "%d of %d qptiff pages carry no <Biomarker>; those channels "
                "are left unnamed (indices: %s)",
                len(missing),
                len(biomarkers),
                ", ".join(str(i) for i in missing[:10]),
            )
        return biomarkers, "Biomarker"
    if key is None or key not in _WARNED_NO_BIOMARKER:
        if key is not None:
            _WARNED_NO_BIOMARKER.add(key)
        logger.warning(
            "%scarries no <Biomarker> element; falling back to the "
            "fluorophore <Name> (DAPI / ATTO 550 / Cy5 ...), which repeats "
            "per cycle and is NOT a marker name. Supply --channel-names "
            "with the real markers, or fix them in marker_map.csv.",
            f"{Path(key).name}: " if key else "this qptiff ",
        )
    return [_element(d, "Name") for d in descriptions], "Name"


def scanprofile_markers(description: str) -> list[str]:
    """Marker names implied by the page-0 ``<ScanProfile>`` acquisition plan.

    The ScanProfile is the *plan*, not the saved channels: it lists every
    well x every dye slot, so it contains unimaged ``--`` placeholders and
    one ``DAPI`` per cycle. Dropping the placeholders, and dropping the
    per-cycle DAPI repeats (a well contributes nothing when its only real
    entry is DAPI), leaves the non-nuclear markers in acquisition order —
    which should equal the per-page ``<Biomarker>`` values after the single
    leading DAPI channel.

    Used as a cross-check only, never to name channels.

    Args:
        description: A qptiff page's ``ImageDescription`` XML.

    Returns:
        Planned non-nuclear marker names in acquisition order; empty when
        no ScanProfile is present or it cannot be parsed.
    """
    raw = _element(description, "ScanProfile")
    if not raw:
        return []
    try:
        profile = json.loads(raw)
        wells = profile["experimentDescription"]["wells"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.debug("could not parse qptiff ScanProfile: %s", exc)
        return []
    markers: list[str] = []
    for well in wells:
        for item in well.get("items", []):
            name = str(item.get("markerName", "")).strip()
            if name and name != _SCANPROFILE_PLACEHOLDER and name != "DAPI":
                markers.append(name)
    return markers


def _log_scanprofile_crosscheck(
    description: str, names: list[str | None]
) -> None:
    """Log whether the ScanProfile plan reconciles with the page names.

    A match is strong evidence the per-page ``<Biomarker>`` mapping is the
    real channel order rather than a coincidence. A mismatch is logged, not
    raised: the pages are the authority (they describe what was actually
    saved), the profile only describes what was planned.
    """
    planned = scanprofile_markers(description)
    if not planned:
        return
    observed = [n for n in names if n and n.strip().upper() != "DAPI"]
    if observed == planned:
        logger.debug(
            "qptiff ScanProfile cross-check: the %d planned marker(s) match "
            "the per-page names exactly",
            len(planned),
        )
    else:
        logger.warning(
            "qptiff ScanProfile lists %d marker(s) but the pages name %d "
            "non-nuclear channel(s), and the orders differ. Using the "
            "per-page names (what was actually saved); review marker_map.csv.",
            len(planned),
            len(observed),
        )


def _build_channel_dicts(
    names: list[str | None], pages: list[Any]
) -> list[dict[str, Any]]:
    """Reader-contract channel dicts, one per channel, in source order."""
    return [
        {
            "name": name,
            "marker_raw": name,
            "source_index": i,
            "fluorophore": _element(_page_description(page), "Name"),
        }
        for i, (name, page) in enumerate(zip(names, pages, strict=True))
    ]


def read_qptiff(
    path: str | Path, *, level: int = 0
) -> tuple[np.ndarray, list[dict[str, Any]], float | None, dict[str, Any]]:
    """Read one pyramid level of a qptiff's Baseline series.

    Loads the whole level into RAM. A full-resolution whole slide is tens
    of gigabytes, so ingest streams instead — see
    :func:`qptiff_channel_iter`. This function is for reduced levels,
    quick inspection, and tests.

    Args:
        path: Path to a ``.qptiff`` file.
        level: Pyramid level; 0 is full resolution.

    Returns:
        Tuple ``(image, channels, mpp, source_meta)``:

        - ``image``: ``(c, y, x)`` ndarray, dtype preserved from source.
        - ``channels``: per-channel dicts with ``name`` / ``marker_raw``
          (the ``<Biomarker>``), ``source_index`` and ``fluorophore``.
        - ``mpp``: microns-per-pixel **of the requested level**, read from
          that level's ``XResolution`` / ``ResolutionUnit``, else ``None``.
        - ``source_meta``: provenance — ``path``, ``file_size``, ``axes``,
          ``shape``, ``dtype``, ``page_count``, ``pyramid_levels``,
          ``level``, ``level_shapes``, ``slide_id``,
          ``channel_name_source``, ``sample_is_tma``.

    Raises:
        ReaderError: If the path is missing, is a directory, has the wrong
            extension, ``level`` is out of range, or tifffile cannot
            decode the file.

    Example:
        Read the smallest pyramid level of a TMA scan::

            image, channels, mpp, meta = read_qptiff("TMA_1.qptiff", level=5)
    """
    channels, mpp, meta = read_qptiff_metadata(path, level=level)
    p = Path(str(meta["path"]))
    try:
        with tifffile.TiffFile(str(p)) as tif:
            image = _select_level(_baseline_series(tif), level).asarray()
    except Exception as exc:
        raise ReaderError(
            f"tifffile could not decode {p}: {type(exc).__name__}: {exc}"
        ) from exc
    return image, channels, mpp, meta


def read_qptiff_metadata(
    path: str | Path, *, level: int = 0
) -> tuple[list[dict[str, Any]], float | None, dict[str, Any]]:
    """Read a qptiff's channels, mpp and provenance — no pixel decode.

    The cheap half of :func:`read_qptiff`: opens the file, reads the page
    headers, and returns everything except the pixels. Ingest uses this to
    resolve marker names and size the output store before touching pixels.

    Args:
        path: Path to a ``.qptiff`` file.
        level: Pyramid level; 0 is full resolution.

    Returns:
        ``(channels, mpp, source_meta)`` — as :func:`read_qptiff`, minus
        the image.

    Raises:
        ReaderError: If the path is invalid, ``level`` is out of range, or
            tifffile cannot open the file.
    """
    p = _validate_path(path)
    try:
        with tifffile.TiffFile(str(p)) as tif:
            series = _baseline_series(tif)
            levels = list(series.levels) if series.levels else [series]
            chosen = _select_level(series, level)
            pages = list(chosen.pages)
            names, name_source = _channel_names(pages, p)
            page0_description = _page_description(list(levels[0].pages)[0])
            _log_scanprofile_crosscheck(page0_description, names)
            mpp = _parse_tiff_resolution(cast(tifffile.TiffPage, pages[0]))
            channels = _build_channel_dicts(names, pages)
            meta: dict[str, Any] = {
                "path": str(p),
                "file_size": p.stat().st_size,
                "axes": str(chosen.axes),
                "shape": tuple(int(s) for s in chosen.shape),
                "dtype": str(np.dtype(chosen.dtype)),
                "page_count": len(tif.pages),
                "pyramid_levels": len(levels),
                "level": level,
                "level_shapes": [
                    tuple(int(s) for s in lv.shape) for lv in levels
                ],
                # Recorded verbatim from the file, whatever it says.
                "slide_id": _element(page0_description, "SlideID"),
                "channel_name_source": name_source,
                # The scanner's own claim about what is on the slide, from
                # the ScanProfile. Tri-state on purpose: True, False, or
                # None when the file does not say. Older writers omit it
                # entirely, so absence is not a claim that this is a
                # section.
                "sample_is_tma": _sample_is_tma(page0_description),
            }
    except ReaderError:
        raise
    except Exception as exc:
        raise ReaderError(
            f"tifffile could not read {p}: {type(exc).__name__}: {exc}"
        ) from exc
    if len(channels) != meta["shape"][0]:
        raise ReaderError(
            f"{p.name}: level {level} has {meta['shape'][0]} channel(s) but "
            f"{len(channels)} page(s) — the file's channel metadata is "
            f"inconsistent, refusing to guess the mapping."
        )
    return channels, mpp, meta


def qptiff_channel_iter(
    path: str | Path, *, level: int = 0
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(channel_index, plane)`` one channel at a time.

    Peak memory is one 2-D plane instead of the whole ``(c, y, x)`` stack,
    which is what makes a 35+ GB full-resolution whole slide ingestable.

    Args:
        path: Path to a ``.qptiff`` file.
        level: Pyramid level; 0 is full resolution.

    Yields:
        ``(index, plane)`` in channel order, ``plane`` a 2-D array.

    Raises:
        ReaderError: If the path is invalid or ``level`` is out of range.

    Example:
        Stream a level into a preallocated store::

            for i, plane in qptiff_channel_iter("TMA_1.qptiff", level=2):
                dest[i] = plane
    """
    p = _validate_path(path)
    with tifffile.TiffFile(str(p)) as tif:
        chosen = _select_level(_baseline_series(tif), level)
        for i, page in enumerate(chosen.pages):
            yield i, np.asarray(page.asarray())


def extract_qptiff_channel_names(path: str | Path) -> list[str | None]:
    """Per-channel marker names from a qptiff — names only, no pixel read.

    Mirrors :func:`coral.io.readers.extract_channel_names` for qptiff
    inputs, so marker resolution can gather a cohort's names cheaply.

    Args:
        path: Path to a ``.qptiff`` file.

    Returns:
        One name per channel in source order; ``None`` where the file
        carries no name for that channel.

    Raises:
        ReaderError: If the file cannot be read.

    Example:
        >>> extract_qptiff_channel_names("TMA_1.qptiff")[:3]
        ['DAPI', 'CD20', 'Pax5']
    """
    channels, _, _ = read_qptiff_metadata(path)
    return [c["name"] for c in channels]


# --- Full metadata record ---------------------------------------------
#
# Everything the scanner wrote, so a converted store answers questions
# about acquisition without anyone reopening the qptiff.
#
# The record is deliberately two things at once. The typed fields are what
# CORAL resolved and what OME has somewhere to put. The verbatim page XML
# is the rest, kept because a vendor records more than any schema models
# and the alternative is deciding today which of it nobody will want.


def _strip_scan_profile(description: str) -> tuple[str, str | None]:
    """Split a page description into parsable XML and the ScanProfile.

    The ScanProfile carries a JSON blob (Fusion) or a deep nested tree
    (Polaris) and only page 0 has one. Removing it leaves a document that
    parses cleanly, and keeps the largest string in the file from being
    walked once per page.

    Args:
        description: A page's ``ImageDescription``.

    Returns:
        ``(description_without_profile, profile_or_None)``.
    """
    profile = _element(description, "ScanProfile")
    if profile is None:
        return description, None
    stripped = _ELEMENT_RE["ScanProfile"].sub("", description)
    return stripped, profile


def _description_tree(description: str) -> dict[str, Any]:
    """A page's XML as nested dicts, ScanProfile excluded.

    Repeated siblings (``<Band>`` inside ``<Bands>``) become lists;
    leaves become their text. Attributes are dropped: PerkinElmer puts
    everything in element text, and none of the four test files carries a
    meaningful attribute.

    Returns an empty dict rather than raising when the XML will not
    parse. A record that is missing a field is recoverable; an ingest that
    dies on a metadata quirk after streaming 13 GB is not.

    Args:
        description: A page's ``ImageDescription``.

    Returns:
        The parsed tree, or ``{}`` when the description is absent or
        unparsable.

    Example:
        >>> _description_tree("<Root><A>1</A><B><C>2</C></B></Root>")
        {'A': '1', 'B': {'C': '2'}}
    """
    from xml.etree import ElementTree

    stripped, _ = _strip_scan_profile(description)
    if not stripped.strip():
        return {}
    try:
        root = ElementTree.fromstring(stripped)
    except ElementTree.ParseError as exc:
        logger.debug("could not parse a qptiff page description: %s", exc)
        return {}
    return cast(dict[str, Any], _element_to_obj(root))


def _element_to_obj(node: Any) -> Any:  # noqa: ANN401 — XML is untyped
    """One XML element as a dict, a string, or ``None`` if empty."""
    children = list(node)
    if not children:
        text = (node.text or "").strip()
        return text or None
    out: dict[str, Any] = {}
    for child in children:
        value = _element_to_obj(child)
        if child.tag in out:
            existing = out[child.tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                out[child.tag] = [existing, value]
        else:
            out[child.tag] = value
    return out


def _tiff_datetime(page: Any) -> str | None:  # noqa: ANN401 — TiffPage
    """The page's ``DateTime`` tag, verbatim (``2022:04:13 13:00:37``).

    Left in TIFF's own format here. Converting to ISO is the OME writer's
    job, and a reader that reformats loses the ability to say what the
    file actually said.
    """
    tag = page.tags.get("DateTime") if page.tags else None
    value = str(tag.value).strip() if tag is not None else ""
    return value or None


def _tiff_tags(page: Any) -> dict[str, Any]:  # noqa: ANN401 — TiffPage
    """The level-0 TIFF tags worth keeping, as JSON-safe values."""
    wanted = (
        "DateTime",
        "Software",
        "XResolution",
        "YResolution",
        "ResolutionUnit",
        "Compression",
        "PhotometricInterpretation",
        "BitsPerSample",
        "SampleFormat",
        "TileWidth",
        "TileLength",
    )
    out: dict[str, Any] = {}
    for name in wanted:
        tag = page.tags.get(name) if page.tags else None
        if tag is None:
            continue
        value = tag.value
        if isinstance(value, tuple):
            out[name] = [_json_safe(v) for v in value]
        else:
            out[name] = _json_safe(value)
    return out


def _json_safe(value: Any) -> Any:  # noqa: ANN401 — tag values are untyped
    """A tag value reduced to something ``json`` can write.

    tifffile hands back ``IntEnum`` for the coded tags, and those are
    ``int`` subclasses, so a naive check writes a bare ``3`` for
    ``ResolutionUnit``. The name is stored instead: ``"CENTIMETER"``
    means something to a person reading the attributes, and ``3`` needs
    the TIFF specification to decode.

    Example:
        >>> import enum
        >>> class Unit(enum.IntEnum):
        ...     CENTIMETER = 3
        >>> _json_safe(Unit.CENTIMETER)
        'CENTIMETER'
    """
    import enum

    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)


def _content_fingerprint(path: Path) -> str:
    """A cheap identity for the source file.

    ``sha256`` of the first and last megabyte plus the byte count. NOT a
    whole-file digest: hashing 3.9 GB is a minute of pure I/O per slide,
    and this only has to answer "is this the same file the store was
    built from", where the header, the last tile and the exact length
    together are conclusive enough.

    The returned string says which it is, so nobody reads it as a
    checksum.

    Args:
        path: The source file.

    Returns:
        ``"head1M+tail1M+size:<hex>"``.
    """
    import hashlib

    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(_FINGERPRINT_EDGE))
        if size > _FINGERPRINT_EDGE:
            handle.seek(max(0, size - _FINGERPRINT_EDGE))
            digest.update(handle.read(_FINGERPRINT_EDGE))
    return f"head1M+tail1M+size:{digest.hexdigest()}"


def _nominal_mpp(profile: str | None) -> float | None:
    """The operator's nominal pixel size from the ScanProfile.

    ``PixelSizeMicrons`` lives inside JSON on Fusion and inside XML on
    Polaris, so both are tried. This is the figure set at the scanner, not
    a measurement: it reads 0.5 where the resolution tag says 0.49757538.
    The tag wins; this is kept so the disagreement stays visible instead
    of being silently resolved.

    Args:
        profile: The raw ``<ScanProfile>`` contents, or ``None``.

    Returns:
        The nominal microns per pixel, or ``None`` when absent.

    Example:
        >>> _nominal_mpp("<r><PixelSizeMicrons>0.5</PixelSizeMicrons></r>")
        0.5
    """
    if not profile:
        return None
    match = re.search(r'"?PixelSizeMicrons"?\s*[:>]\s*"?([0-9.]+)', profile)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def read_qptiff_record(path: str | Path) -> dict[str, Any]:
    """Everything the qptiff records about how it was acquired.

    Reads page headers only, no pixels. Returns a JSON-safe dict built to
    be written straight into a store's attributes.

    Two halves. ``slide``, ``acquisition``, ``channels`` and ``tiff_tags``
    are parsed values, one entry per level-0 page for the channels. ``xml``
    is every distinct page description verbatim, each listing the pages
    that share it, so nothing the vendor wrote is discarded even where
    CORAL has no field for it.

    Deduplication matters: a 29-channel slide has 177 pages but only 61
    distinct descriptions, and the reduced-resolution pages repeat their
    level-0 twins almost exactly.

    Args:
        path: Path to a ``.qptiff`` file.

    Returns:
        The record. Keys ``slide``, ``acquisition``, ``channels``,
        ``tiff_tags``, ``xml``, ``provenance``.

    Raises:
        ReaderError: If the path is invalid or tifffile cannot open it.
    """
    p = _validate_path(path)
    try:
        with tifffile.TiffFile(str(p)) as tif:
            series = _baseline_series(tif)
            levels = list(series.levels) if series.levels else [series]
            level0 = list(levels[0].pages)
            descriptions = [_page_description(pg) for pg in level0]
            all_descriptions = [_page_description(pg) for pg in tif.pages]
            _, profile = _strip_scan_profile(descriptions[0])
            trees = [_description_tree(d) for d in descriptions]
            tags = _tiff_tags(level0[0])
            acquired = _tiff_datetime(level0[0])
    except ReaderError:
        raise
    except Exception as exc:
        raise ReaderError(
            f"tifffile could not read {p}: {type(exc).__name__}: {exc}"
        ) from exc

    head = trees[0] if trees else {}
    slide = {
        key: head.get(key)
        for key in (
            "SlideID",
            "Barcode",
            "Identifier",
            "StudyName",
            "OperatorName",
            "ComputerName",
            "AcquisitionSoftware",
            "DescriptionVersion",
            "InstrumentType",
            "LampType",
            "CameraType",
            "CameraName",
            "Objective",
        )
    }
    return {
        "slide": slide,
        "acquisition": {
            "datetime_tiff": acquired,
            "mpp_nominal": _nominal_mpp(profile),
            "camera_settings": head.get("CameraSettings"),
            "scan_profile": profile,
        },
        "channels": [
            {
                key: tree.get(key)
                for key in (
                    "Name",
                    "Biomarker",
                    "Color",
                    "ExposureTime",
                    "SignalUnits",
                    "ScaleFactor",
                    "IsUnmixedComponent",
                    "AutofluorescenceSubtracted",
                    "Objective",
                    "Responsivity",
                    "ExcitationFilter",
                    "EmissionFilter",
                )
            }
            for tree in trees
        ],
        "tiff_tags": tags,
        "xml": _distinct_descriptions(all_descriptions),
        "provenance": {
            "source_file": str(p.resolve()),
            "source_size_bytes": p.stat().st_size,
            "source_fingerprint": _content_fingerprint(p),
        },
    }


def _distinct_descriptions(
    descriptions: list[str],
) -> list[dict[str, Any]]:
    """Distinct page descriptions, each with the pages that carry it.

    Order is first appearance, so page 0 (the one with the ScanProfile)
    leads and the list reads in file order.

    Args:
        descriptions: Every page's ``ImageDescription``, in page order.

    Returns:
        ``[{"pages": [...], "description": "..."}]``, empty entries
        skipped.

    Example:
        >>> [d["pages"] for d in _distinct_descriptions(["a", "b", "a"])]
        [[0, 2], [1]]
    """
    seen: dict[str, list[int]] = {}
    for i, text in enumerate(descriptions):
        if text:
            seen.setdefault(text, []).append(i)
    return [
        {"pages": pages, "description": text} for text, pages in seen.items()
    ]


#: Non-pyramidal series a scanner writes beside the image. ``Label`` is a
#: photograph of the physical slide label and ``Macro`` the whole-slide
#: overview; both are how a person confirms a store is the slide they
#: meant. Polaris writes neither, so their absence is normal.
_OVERVIEW_SERIES = ("Label", "Macro", "Thumbnail")


def read_qptiff_overviews(path: str | Path) -> dict[str, np.ndarray]:
    """The scanner's own overview images, by series name.

    These are the only things in a qptiff that tie the digital store back
    to a physical object on a bench: ``Label`` is a photograph of the
    slide label, ``Macro`` is the whole-slide overview. CORAL renders its
    own nuclear preview, which shows what was imaged but not which slide
    it came off.

    Small enough to decode outright (a Macro is 2736 x 1536), unlike the
    Baseline series.

    Args:
        path: Path to a ``.qptiff`` file.

    Returns:
        ``{series_name: array}`` for whichever of Label, Macro and
        Thumbnail the file carries. Empty for a file with none, which is
        normal: Vectra Polaris writes only a Thumbnail.

    Raises:
        ReaderError: If the path is invalid or tifffile cannot open it.
    """
    p = _validate_path(path)
    found: dict[str, np.ndarray] = {}
    try:
        with tifffile.TiffFile(str(p)) as tif:
            for series in tif.series:
                name = str(series.name)
                if name in _OVERVIEW_SERIES:
                    found[name] = np.asarray(series.asarray())
    except ReaderError:
        raise
    except Exception as exc:
        raise ReaderError(
            f"tifffile could not read {p}: {type(exc).__name__}: {exc}"
        ) from exc
    return found
