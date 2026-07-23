"""Stable hashes used as slide IDs across runs.

A slide's stable hash is derived from the string form of its path:
the same path always hashes to the same 12-hex-char string. State
files survive moves of the parent job directory and re-runs that
re-open the same slide.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_slide_hash(slide_path: str | Path) -> str:
    """Return a 12-hex-char stable hash for a slide path.

    The hash is SHA-1 over the UTF-8 bytes of the stringified path.
    We don't resolve symlinks or normalise the path on purpose — the
    caller decides whether to pass an absolute, relative or
    canonicalised path. The same string always hashes to the same
    output.

    Invalid UTF-8 bytes in the path are dropped via ``errors="ignore"``.
    Paths on filesystems with non-UTF8 encodings could therefore
    collide; we accept this trade-off because real Mahmood Lab paths
    are ASCII or valid UTF-8, and ``ignore`` keeps the function total
    (no surprise exceptions during state writes).

    Args:
        slide_path: Slide path; coerced to ``str`` before hashing.

    Returns:
        Lowercase 12-hex-char string.

    Example:
        >>> h = stable_slide_hash("/data/slide.zarr")
        >>> len(h)
        12
        >>> all(c in "0123456789abcdef" for c in h)
        True
        >>> stable_slide_hash("/data/slide.zarr") == h
        True
    """
    raw = str(slide_path).encode("utf-8", errors="ignore")
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(raw)
    return digest.hexdigest()[:12]
