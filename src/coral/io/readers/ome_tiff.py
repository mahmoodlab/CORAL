"""Reader for single-file (OME-)TIFF inputs.

Handles three real-world variants:

- **OME-TIFF** — channel names in embedded OME-XML, mpp in
  ``PhysicalSizeX``. Fixtures ``core001_TMA.ome.tiff`` and
  ``RCC_TMA001(reg3x4)_...ome.tiff``.
- **ImageJ hyperstack TIFF** — channel names rarely present; axes
  detected by tifffile (``CYX`` or ``TCYX``). Fixtures ``A-1.tif``
  and ``core001.tif``.
- **Raw multi-page TIFF** — no metadata; tifffile reports axes as
  ``QYX`` (unknown leading dim). Fixture ``CRC_TMA_A_...tiff``.

Returns ``(image, channels, mpp, source_meta)`` in whatever raw shape
tifffile reports — harmonization to canonical ``(c, y, x)`` is the
caller's job.

The OME-XML walk follows the reference KRONOS QC pattern.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import numpy as np
import tifffile

from coral.io.readers._memmap import open_with_memmap_fallback
from coral.utils.errors import ReaderError

_TIFF_EXTENSIONS = (".tif", ".tiff", ".ome.tif", ".ome.tiff")
_OME_NAMESPACE = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
_UNIT_SCALE_UM_PER_INCH = 25400.0
_UNIT_SCALE_UM_PER_CM = 10000.0


def read_ometiff(
    path: str | Path,
) -> tuple[np.ndarray, list[dict[str, Any]], float | None, dict[str, Any]]:
    """Read a single-file (OME-)TIFF and return its raw contents.

    No harmonization — the array is returned in whatever shape and
    axes order tifffile reports (``CYX``, ``TCYX``, ``QYX``, etc.).
    The downstream ``harmonize_to_canonical`` maps any of
    those to canonical ``(c, y, x)``.

    Args:
        path: Path to a single TIFF file (``.tif``, ``.tiff``,
            ``.ome.tif``, ``.ome.tiff``).

    Returns:
        Tuple ``(image, channels, mpp, source_meta)``:

        - ``image``: ndarray. Dtype preserved from source. For
          most multiplex inputs this is ``(C, Y, X)`` or
          ``(T, C, Y, X)``.
        - ``channels``: list of dicts with per-channel info. Keys:
          ``name`` (str | None), ``marker_raw`` (str | None),
          ``source_index`` (int). For TCYX inputs the cycle index
          is also recorded as ``cycle`` (int). When the source
          carries no channel names (ImageJ without ``Labels``, raw
          TIFF), ``name`` and ``marker_raw`` are ``None``; the
          downstream harmonize layer fills in placeholder names.
        - ``mpp``: microns-per-pixel from OME ``PhysicalSizeX``,
          else from TIFF ``XResolution`` / ``ResolutionUnit``,
          else ``None``.
        - ``source_meta``: provenance dict — keys ``path``,
          ``file_size``, ``axes`` (tifffile's axes string),
          ``page_count``, ``ome_xml`` (str | None),
          ``imagej_metadata`` (dict | None), ``pyramid_levels``
          (int).

    Raises:
        ReaderError: If the path is missing, is a directory, has
            the wrong extension, or tifffile cannot decode it.

    Example:
        >>> from pathlib import Path
        >>> img, ch, mpp, meta = read_ometiff(Path("tests/data/tiny.ome.tiff"))
        >>> img.shape[0] == len(ch)
        True
        >>> meta["pyramid_levels"] >= 1
        True
    """
    p = _validate_path(path)
    try:
        with tifffile.TiffFile(str(p)) as tif:
            axes = tif.series[0].axes
            shape = tuple(tif.series[0].shape)
            page_count = len(tif.pages)
            ome_xml = tif.ome_metadata if tif.is_ome else None
            imagej_md = tif.imagej_metadata
            pyramid_levels = _detect_pyramid_levels(tif)

            mpp = None
            ome_channel_dicts: list[dict[str, Any]] | None = None
            if ome_xml:
                ome_channel_dicts, mpp = _parse_ome_xml(ome_xml)

            imagej_labels: list[str] | None = None
            if imagej_md:
                imagej_labels = _imagej_labels(imagej_md)

            if mpp is None:
                mpp = _parse_tiff_resolution(
                    cast(tifffile.TiffPage, tif.pages[0])
                )
    except ReaderError:
        raise
    except Exception as exc:
        raise ReaderError(
            f"tifffile could not decode {p}: {type(exc).__name__}: {exc}"
        ) from exc

    image = open_with_memmap_fallback(p)
    n_channel_slots = _infer_channel_count(image, axes)
    channels = _build_channel_dicts(
        n_slots=n_channel_slots,
        axes=axes,
        shape=shape,
        ome_channels=ome_channel_dicts,
        imagej_labels=imagej_labels,
    )

    source_meta: dict[str, Any] = {
        "path": str(p),
        "file_size": p.stat().st_size,
        "axes": axes,
        "shape": shape,
        "page_count": page_count,
        "ome_xml": ome_xml,
        "imagej_metadata": imagej_md,
        "pyramid_levels": pyramid_levels,
    }
    return image, channels, mpp, source_meta


def extract_channel_names(path: str | Path) -> list[str | None]:
    """Return per-channel names from a single OME-TIFF.

    Convenience wrapper around ``read_ometiff`` for callers that
    only need the names. Returns one element per channel slot in
    the source axes order; entries are ``None`` if the source
    carries no name for that channel.

    Args:
        path: Path to a single TIFF file.

    Returns:
        List of channel names (or ``None`` placeholders). Length
        matches the channel-count inferred by the reader.

    Raises:
        ReaderError: If the path is missing, the wrong kind, or
            tifffile cannot decode it.

    Example:
        >>> names = extract_channel_names("tests/data/tiny.ome.tiff")
        >>> names
        ['DAPI', 'CD3', 'CD8', 'Vimentin']
    """
    _, channels, _, _ = read_ometiff(path)
    return [c["name"] for c in channels]


def _validate_path(path: str | Path) -> Path:
    """Boundary validation: exists, is a file, has a TIFF extension."""
    p = Path(path)
    if not p.exists():
        raise ReaderError(f"File not found: {p}")
    if p.is_dir():
        raise ReaderError(
            f"Path is a directory, not a file: {p}. "
            f"Use read_channel_tiff_dir for dir-of-tiffs inputs."
        )
    name_lower = p.name.lower()
    if not any(name_lower.endswith(ext) for ext in _TIFF_EXTENSIONS):
        raise ReaderError(
            f"File extension not recognized as TIFF: {p.name}. "
            f"Expected one of: {_TIFF_EXTENSIONS}"
        )
    return p


def _detect_pyramid_levels(tif: tifffile.TiffFile) -> int:
    """Count pyramid levels in series[0]; non-pyramidal → 1."""
    levels = tif.series[0].levels
    return len(levels) if levels else 1


def _parse_ome_xml(
    xml_str: str,
) -> tuple[list[dict[str, Any]] | None, float | None]:
    """Extract per-channel dicts and mpp from an OME-XML string.

    Returns ``(None, None)`` if parsing fails — the caller falls
    back to other metadata sources.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None, None

    channel_elements = root.findall(".//ome:Channel", _OME_NAMESPACE)
    channels: list[dict[str, Any]] = [
        {
            "name": ce.attrib.get("Name"),
            "id": ce.attrib.get("ID"),
            "color": ce.attrib.get("Color"),
            "emission_wavelength": ce.attrib.get("EmissionWavelength"),
        }
        for ce in channel_elements
    ]

    mpp: float | None = None
    pixels = root.find(".//ome:Pixels", _OME_NAMESPACE)
    if pixels is not None:
        psx = pixels.attrib.get("PhysicalSizeX")
        if psx is not None:
            try:
                mpp_value = float(psx)
                unit = pixels.attrib.get("PhysicalSizeXUnit", "µm")
                if unit in ("µm", "um", "micrometer", None):
                    mpp = mpp_value
                elif unit in ("nm", "nanometer"):
                    mpp = mpp_value / 1000.0
            except ValueError:
                pass

    return (channels if channels else None), mpp


def _imagej_labels(imagej_md: dict[str, Any]) -> list[str] | None:
    """Extract per-channel labels from ImageJ metadata, if present."""
    labels = imagej_md.get("Labels")
    if isinstance(labels, list) and labels:
        return [str(label) for label in labels]
    return None


def _parse_tiff_resolution(page: tifffile.TiffPage) -> float | None:
    """Convert TIFF XResolution + ResolutionUnit to µm/px.

    Returns ``None`` if the tags are missing or non-meaningful.
    Callers should ``cast(tifffile.TiffPage, tif.pages[0])`` —
    tifffile types ``pages[0]`` as the ``TiffPage | TiffFrame``
    union, but page 0 of a multi-page TIFF is always a ``TiffPage``
    in practice.
    """
    tags = page.tags
    x_res_tag = tags.get("XResolution")
    unit_tag = tags.get("ResolutionUnit")
    if x_res_tag is None or unit_tag is None:
        return None

    x_res_value = x_res_tag.value
    if not (isinstance(x_res_value, tuple) and len(x_res_value) == 2):
        return None
    num, den = x_res_value
    if den == 0 or num == 0:
        return None
    x_res = num / den

    unit_code = unit_tag.value
    unit_int = int(unit_code) if hasattr(unit_code, "__int__") else unit_code
    if unit_int == 2:
        unit_scale = _UNIT_SCALE_UM_PER_INCH
    elif unit_int == 3:
        unit_scale = _UNIT_SCALE_UM_PER_CM
    else:
        return None
    return unit_scale / x_res


def _infer_channel_count(image: np.ndarray, axes: str) -> int:
    """Number of channel slots inferred from axes + shape.

    For ``CYX`` / ``QYX`` returns ``image.shape[0]``. For ``TCYX``
    returns ``T * C``. Otherwise falls back to the leading dim
    (best effort; the caller's harmonize layer handles axes
    properly).
    """
    if image.ndim == 2:
        return 1
    if axes == "TCYX" and image.ndim == 4:
        return int(image.shape[0]) * int(image.shape[1])
    return int(image.shape[0])


def _build_channel_dicts(
    *,
    n_slots: int,
    axes: str,
    shape: tuple[int, ...],
    ome_channels: list[dict[str, Any]] | None,
    imagej_labels: list[str] | None,
) -> list[dict[str, Any]]:
    """Build the per-channel dict list returned by the reader.

    Priority for ``name`` / ``marker_raw``:
      1. OME-XML ``<Channel Name=...>`` if present and length matches.
      2. ImageJ ``Labels`` if present and length matches.
      3. ``None`` (downstream harmonize fills with placeholder names).

    For ``TCYX`` axes, ``cycle`` index is recorded; channels are
    laid out cycle-major (cycle 0 channels 0..C-1, then cycle 1
    channels 0..C-1, etc.).
    """
    names: list[str | None]
    if ome_channels and len(ome_channels) == n_slots:
        names = [c.get("name") for c in ome_channels]
    elif imagej_labels and len(imagej_labels) == n_slots:
        names = list(imagej_labels)
    else:
        names = [None] * n_slots

    is_tcyx = axes == "TCYX" and len(shape) == 4
    n_per_cycle = int(shape[1]) if is_tcyx else 0

    channels: list[dict[str, Any]] = []
    for i in range(n_slots):
        entry: dict[str, Any] = {
            "name": names[i],
            "marker_raw": names[i],
            "source_index": i,
        }
        if is_tcyx:
            entry["cycle"] = i // n_per_cycle if n_per_cycle else None
        channels.append(entry)
    return channels
