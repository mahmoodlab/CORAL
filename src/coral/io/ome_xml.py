"""OME-XML fragments built from a source's own acquisition record.

``OME/METADATA.ome.xml`` is the interoperability surface: it is what
Bio-Formats parses and therefore what QuPath, napari and OMERO see. The
minimal version CORAL writes for an array ingest carries channel names and
a pixel size, which is all an array source can supply.

A qptiff supplies a great deal more, and this module turns that into the
typed OME fields the schema defines: acquisition date, instrument,
detector, objective, per-channel fluorophore, colour, excitation and
emission wavelengths, and per-plane exposure. Anything the schema has no
field for stays in the store's ``source_qptiff`` attribute instead: OME-XML
carries what a reader can act on, the attribute carries the rest.

Two rules govern everything here:

**Element order is part of the schema.** OME 2016-06 is a sequence, not a
bag. ``Instrument`` precedes ``Image``; inside ``Pixels`` it is
``Channel*`` then ``TiffData`` then ``Plane*``. Out of order is invalid
even when every value is right, and Bio-Formats will not say so politely.

**A value is written only when it is known.** An attribute derived from a
missing element is omitted, never defaulted, because a plausible wrong
number outlives the person who guessed it.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from xml.sax.saxutils import escape, quoteattr

logger = logging.getLogger(__name__)

__all__ = [
    "channel_optics",
    "instrument_block",
    "ome_color",
    "ome_datetime",
]

#: OME writes exposure as a float with an explicit unit. PerkinElmer
#: writes microseconds: Bio-Formats' own VectraReader parses the element
#: as ``UNITS.MICROSECOND``, which is the authority worth following. The
#: magnitudes agree (a 161180 read as seconds would be a 45-hour tile).
_EXPOSURE_UNIT = "µs"

#: Fully opaque, since the vendor's ``<Color>`` carries no alpha.
_ALPHA = 0xFF


def ome_datetime(tiff_datetime: str | None) -> str | None:
    """TIFF's ``DateTime`` as an ``xsd:dateTime``.

    TIFF writes ``2022:04:13 13:00:37``; OME wants
    ``2022-04-13T13:00:37``. No timezone is added: the file does not say
    which one it was scanned in, and inventing one moves the timestamp.

    Args:
        tiff_datetime: The tag value, or ``None``.

    Returns:
        The ISO form, or ``None`` when absent or unrecognised.

    Example:
        >>> ome_datetime("2022:04:13 13:00:37")
        '2022-04-13T13:00:37'
        >>> ome_datetime("not a date") is None
        True
    """
    if not tiff_datetime:
        return None
    match = re.match(
        r"^\s*(\d{4}):(\d{2}):(\d{2})[ T](\d{2}:\d{2}:\d{2})\s*$",
        tiff_datetime,
    )
    if match is None:
        logger.debug("unrecognised TIFF DateTime: %r", tiff_datetime)
        return None
    year, month, day, clock = match.groups()
    return f"{year}-{month}-{day}T{clock}"


def ome_color(rgb: str | None) -> int | None:
    """A vendor ``"r,g,b"`` string as an OME ``Color``.

    OME packs colour as RGBA in a **signed** 32-bit integer, so anything
    with red above 127 comes out negative. The schema's own example is
    solid red at ``-16776961`` (``0xFF0000FF``), which is the check this
    packing is built to reproduce.

    Args:
        rgb: ``"0,0,255"`` as PerkinElmer writes it, or ``None``.

    Returns:
        The packed value, or ``None`` when absent or malformed.

    Example:
        >>> ome_color("255,0,0")
        -16776961
        >>> ome_color("0,0,255")
        65535
        >>> ome_color("nonsense") is None
        True
    """
    if not rgb:
        return None
    parts = [p.strip() for p in str(rgb).split(",")]
    if len(parts) != 3:
        return None
    try:
        red, green, blue = (int(p) for p in parts)
    except ValueError:
        return None
    if not all(0 <= v <= 255 for v in (red, green, blue)):
        return None
    packed = (red << 24) | (green << 16) | (blue << 8) | _ALPHA
    # Reinterpret as signed 32-bit, which is what the schema's type is.
    return packed - (1 << 32) if packed >= (1 << 31) else packed


def _bands(filter_block: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """The ``<Band>`` entries of a filter block, always as a list.

    A single band parses to a dict rather than a list of one, so callers
    would otherwise have to special-case a one-band filter.
    """
    if not isinstance(filter_block, dict):
        return []
    bands = (filter_block.get("Bands") or {}).get("Band")
    if isinstance(bands, dict):
        return [bands]
    return [b for b in (bands or []) if isinstance(b, dict)]


def _midpoint(band: dict[str, Any]) -> float | None:
    """The centre wavelength of a band, in nanometres."""
    try:
        low = float(band["Cuton"])
        high = float(band["Cutoff"])
    except (KeyError, TypeError, ValueError):
        return None
    return round((low + high) / 2, 1)


def channel_optics(channel: dict[str, Any]) -> dict[str, Any]:
    """Excitation and emission wavelengths for one channel.

    A multi-band filter is shared across a cycle and only one band is
    ``Active`` for any given channel. The emission bands carry no active
    flag, so the active **excitation** index selects the emission band:
    the two filters are written in the same band order, which is what
    makes a triple-band cube work in the first place.

    Args:
        channel: One entry of a ``source_qptiff`` record's ``channels``.

    Returns:
        ``{"excitation": nm | None, "emission": nm | None}``.
    """
    excitation = _bands(channel.get("ExcitationFilter"))
    emission = _bands(channel.get("EmissionFilter"))
    active = next(
        (
            i
            for i, band in enumerate(excitation)
            if str(band.get("Active", "")).strip().lower() == "true"
        ),
        0 if excitation else None,
    )
    if active is None:
        return {"excitation": None, "emission": None}
    return {
        "excitation": _midpoint(excitation[active]),
        "emission": (
            _midpoint(emission[active]) if active < len(emission) else None
        ),
    }


def _attr(name: str, value: Any) -> str:  # noqa: ANN401 — mixed sources
    """One XML attribute, or ``""`` when the value is unknown.

    Omission is deliberate. An absent attribute reads as "the file did not
    say"; a defaulted one reads as fact.
    """
    if value is None or value == "":
        return ""
    return f" {name}={quoteattr(str(value))}"


def instrument_block(record: dict[str, Any]) -> str:
    """The ``<Instrument>`` element for a qptiff, or ``""``.

    Schema order inside ``Instrument`` is ``Microscope``, light sources,
    ``Detector``, ``Objective``. Empty when the source names none of them,
    since an instrument with no content is noise in every reader that
    shows it.

    Args:
        record: A ``source_qptiff`` record.

    Returns:
        The XML fragment, indented to sit directly inside ``<OME>``.
    """
    slide = record.get("slide") or {}
    microscope = slide.get("InstrumentType")
    camera = slide.get("CameraType")
    serial = slide.get("CameraName")
    objective = slide.get("Objective")
    if not any((microscope, camera, serial, objective)):
        return ""

    parts = ['  <Instrument ID="Instrument:0">']
    if microscope:
        parts.append(
            f'    <Microscope{_attr("Model", microscope)} Type="Other"/>'
        )
    if camera or serial:
        parts.append(
            f'    <Detector ID="Detector:0:0"'
            f"{_attr('Model', camera)}{_attr('SerialNumber', serial)}"
            f' Type="CMOS"/>'
        )
    if objective:
        parts.append(
            f'    <Objective ID="Objective:0:0"'
            f"{_attr('Model', objective)}"
            f"{_attr('NominalMagnification', _magnification(objective))}/>"
        )
    parts.append("  </Instrument>")
    return "\n".join(parts) + "\n"


def _magnification(objective: str | None) -> float | None:
    """The leading magnification of an objective name.

    ``"20xLWD"`` is 20, ``"10x"`` is 10. Returns ``None`` rather than
    guessing when the name does not start with a number, because
    ``NominalMagnification`` is a number a reader will draw a scale from.

    Example:
        >>> _magnification("20xLWD"), _magnification("10x")
        (20.0, 10.0)
        >>> _magnification("Plan Apo") is None
        True
    """
    if not objective:
        return None
    match = re.match(r"^\s*([0-9.]+)\s*[xX]", str(objective))
    return float(match.group(1)) if match else None


def channel_element(
    index: int, name: str, channel: dict[str, Any] | None
) -> str:
    """One ``<Channel>``, carrying whatever the source knew about it.

    ``Name`` stays the resolved marker. OME's own labelling priority is
    ``Name`` then ``Fluor``, so on a slide with no ``<Biomarker>`` the
    name is already the fluorophore and ``Fluor`` repeats it. That is
    correct rather than redundant: it records that the label IS a dye.

    Args:
        index: Channel index, for the OME ID.
        name: Resolved marker name.
        channel: The matching ``source_qptiff`` channel entry, or
            ``None`` for a source with no record.

    Returns:
        The XML fragment, indented to sit inside ``<Pixels>``.
    """
    head = (
        f'      <Channel ID="Channel:0:{index}" '
        f'Name="{escape(str(name))}" SamplesPerPixel="1"'
    )
    if not channel:
        return head + "/>"
    optics = channel_optics(channel)
    return (
        head
        + _attr("Fluor", channel.get("Name"))
        + _attr("Color", ome_color(channel.get("Color")))
        + _attr("ExcitationWavelength", optics["excitation"])
        + (
            ' ExcitationWavelengthUnit="nm"'
            if optics["excitation"] is not None
            else ""
        )
        + _attr("EmissionWavelength", optics["emission"])
        + (
            ' EmissionWavelengthUnit="nm"'
            if optics["emission"] is not None
            else ""
        )
        + ' IlluminationType="Epifluorescence"/>'
    )


def plane_elements(channels: list[dict[str, Any]]) -> str:
    """``<Plane>`` entries carrying per-channel exposure.

    Planes come **after** ``TiffData`` in the schema sequence, which is
    the kind of detail that produces a file every value of which is right
    and which no reader will load.

    Args:
        channels: A ``source_qptiff`` record's ``channels``.

    Returns:
        The XML fragment, or ``""`` when no channel records an exposure.
    """
    rows = []
    for i, channel in enumerate(channels):
        exposure = channel.get("ExposureTime")
        if exposure in (None, ""):
            continue
        try:
            value = float(exposure)
        except (TypeError, ValueError):
            continue
        rows.append(
            f'      <Plane TheC="{i}" TheT="0" TheZ="0" '
            f'ExposureTime="{value}" '
            f'ExposureTimeUnit="{_EXPOSURE_UNIT}"/>'
        )
    return ("\n".join(rows) + "\n") if rows else ""
