"""CORAL error hierarchy for public API boundaries.

Public functions in CORAL validate at the boundary and re-raise
third-party exceptions (``tifffile.TiffFileError``,
``xml.etree.ElementTree.ParseError`` etc.) as one of the subclasses
below. Internal helpers raise stdlib errors freely; the conversion
to ``CoralError`` happens at the outermost public-facing call.

Example:
    >>> from coral.utils.errors import CoralError, ReaderError
    >>> issubclass(ReaderError, CoralError)
    True
"""

from __future__ import annotations


class CoralError(Exception):
    """Base for CORAL-raised errors at public API boundaries."""


class ReaderError(CoralError):
    """Input file/dir unreadable, wrong kind, or wrong format.

    Raised by ``read_ometiff`` / ``read_channel_tiff_dir`` when the
    source path is missing, points at the wrong kind of thing
    (file vs directory), fails magic-byte/extension checks, has
    mismatched per-file shapes (dir-of-tiffs case), has disagreeing
    per-file mpp tags (dir-of-tiffs case), or when the underlying
    ``tifffile`` decode fails.
    """


class HarmonizationError(CoralError):
    """Raw axes pattern we can't yet map to canonical ``(c, y, x)``.

    Raised by ``harmonize_to_canonical`` when a reader
    returns an axes string we don't have a dispatch helper for.
    """


class MarkerError(CoralError):
    """Unresolvable marker name.

    Raised by marker normalization only when the caller
    opts into strict mode. Default behaviour is to return the
    cleaned name with match-level ``"none"`` and let the caller
    decide.
    """


class CohortMetadataError(CoralError):
    """Cohort metadata CSV missing, malformed, or contradictory.

    Raised by ``coral.config.cohort`` when the user-supplied
    metadata CSV can't be parsed or has internally-inconsistent
    overrides.
    """
