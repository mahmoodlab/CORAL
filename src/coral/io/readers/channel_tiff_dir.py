"""Reader for the per-channel-TIFF-dir layout (Keyence / Fusion).

One TIFF per channel; channel name parsed from the filename stem.
All files must share the same ``(Y, X)`` shape.

mpp is read from each file's TIFF resolution tags. If all files
have tags and all values agree (within 1e-6 µm) → that value is
used. If files disagree → ``ReaderError`` naming both values and
pointing the user at override mechanisms (cohort CSV / CLI flag).
If no files have tags → ``mpp = None``.

Fixture `10.1/` (29 channels: BCL2.tiff, BCL6.tiff, CD11c.tiff, ...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import tifffile

from coral.io.readers._memmap import open_with_memmap_fallback
from coral.io.readers.ome_tiff import _parse_tiff_resolution
from coral.utils.errors import ReaderError

_MPP_AGREEMENT_TOLERANCE_UM = 1e-6


def read_channel_tiff_dir(
    path: str | Path,
    *,
    extensions: tuple[str, ...] = (".tif", ".tiff"),
) -> tuple[np.ndarray, list[dict[str, Any]], float | None, dict[str, Any]]:
    """Read a directory of per-channel TIFFs as a stacked ``(C, Y, X)`` array.

    Files are sorted alphabetically by filename and stacked along a
    new leading channel axis. The filename stem (e.g. ``DAPI.tiff``
    → ``DAPI``) becomes the raw channel name.

    Args:
        path: Directory containing per-channel TIFF files.
        extensions: File extensions to consider. Defaults to
            ``(".tif", ".tiff")``.

    Returns:
        Tuple ``(image, channels, mpp, source_meta)``:

        - ``image``: ``(C, Y, X)`` ndarray, dtype preserved from
          source.
        - ``channels``: list of dicts with keys ``name``,
          ``marker_raw``, ``source_index``, ``source_path``.
        - ``mpp``: agreed µm/px across files, or ``None`` if no
          files carry resolution tags.
        - ``source_meta``: provenance dict — keys ``path``,
          ``n_files``, ``file_names`` (list of stems in order),
          ``axes`` (always ``"YX*"`` — special marker for harmonize
          dispatch), ``shape``, ``pyramid_levels`` (always 1 — per-file
          pyramids unsupported in 01a).

    Raises:
        ReaderError: If ``path`` is not a directory, contains no
            matching files, contains files with mismatched 2-D
            shapes, has per-file mpp values that disagree across
            files, or any single file fails to decode.

    Example:
        >>> img, ch, mpp, meta = read_channel_tiff_dir("tests/data/tiny_dir")
        >>> img.shape[0] == len(ch)
        True
        >>> meta["axes"]
        'YX*'
    """
    p = _validate_dir(path)
    files = _list_channel_files(p, extensions)

    images = []
    mpps: list[float | None] = []
    for fp in files:
        try:
            arr = open_with_memmap_fallback(fp)
        except Exception as exc:
            raise ReaderError(
                f"Failed to decode {fp.name}: {type(exc).__name__}: {exc}"
            ) from exc
        if arr.ndim != 2:
            raise ReaderError(
                f"Per-channel file {fp.name} has ndim={arr.ndim} "
                f"(expected 2). Multi-channel TIFFs in a dir-of-tiffs "
                f"input are not supported; use read_ometiff for "
                f"single-file multi-channel inputs."
            )
        images.append(arr)
        mpps.append(_read_file_mpp(fp))

    _assert_uniform_shape(files, images)
    mpp = _resolve_mpp_agreement(files, mpps)
    image = np.stack(images, axis=0)
    channels = [
        {
            "name": fp.stem,
            "marker_raw": fp.stem,
            "source_index": i,
            "source_path": str(fp),
        }
        for i, fp in enumerate(files)
    ]
    source_meta: dict[str, Any] = {
        "path": str(p),
        "n_files": len(files),
        "file_names": [fp.stem for fp in files],
        "axes": "YX*",
        "shape": image.shape,
        "pyramid_levels": 1,
    }
    return image, channels, mpp, source_meta


def _validate_dir(path: str | Path) -> Path:
    """Boundary validation: exists, is a directory."""
    p = Path(path)
    if not p.exists():
        raise ReaderError(f"Directory not found: {p}")
    if not p.is_dir():
        raise ReaderError(
            f"Path is not a directory: {p}. "
            f"Use read_ometiff for single-file inputs."
        )
    return p


def _list_channel_files(p: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Return sorted list of per-channel TIFF files. Empty → ReaderError."""
    ext_set = {e.lower() for e in extensions}
    files = sorted(
        fp
        for fp in p.iterdir()
        if fp.is_file() and fp.suffix.lower() in ext_set
    )
    if not files:
        raise ReaderError(
            f"No TIFF files found in {p} "
            f"(looked for extensions: {sorted(ext_set)})."
        )
    return files


def _read_file_mpp(fp: Path) -> float | None:
    """Open a single TIFF; read mpp from its resolution tags."""
    try:
        with tifffile.TiffFile(str(fp)) as tif:
            return _parse_tiff_resolution(
                cast(tifffile.TiffPage, tif.pages[0])
            )
    except Exception:
        return None


def _assert_uniform_shape(files: list[Path], images: list[np.ndarray]) -> None:
    """All per-channel arrays must share the same (Y, X) shape."""
    first_shape = images[0].shape
    for fp, arr in zip(files[1:], images[1:], strict=False):
        if arr.shape != first_shape:
            raise ReaderError(
                f"Per-channel shape mismatch: {files[0].name} has "
                f"shape {first_shape} but {fp.name} has shape "
                f"{arr.shape}. All per-channel files must share "
                f"the same (Y, X) dimensions."
            )


def _resolve_mpp_agreement(
    files: list[Path], mpps: list[float | None]
) -> float | None:
    """Decide the single mpp value for the dir.

    - All None → return None.
    - All have values + all agree within tolerance → return that value.
    - Mix of None and values, OR values disagree → ReaderError listing
      the conflicting (file, mpp) pairs.
    """
    pairs = list(zip(files, mpps, strict=False))
    tagged = [(fp, m) for fp, m in pairs if m is not None]
    if not tagged:
        return None
    untagged = [fp for fp, m in pairs if m is None]
    if untagged:
        raise ReaderError(
            f"Some per-channel files have mpp tags and others don't. "
            f"Tagged: {[fp.name for fp, _ in tagged[:3]]}"
            f"{' ...' if len(tagged) > 3 else ''}. "
            f"Untagged: {[fp.name for fp in untagged[:3]]}"
            f"{' ...' if len(untagged) > 3 else ''}. "
            f"Provide a single mpp override: a per-image metadata CSV "
            f"(`mpp` column) or the global `--mpp` flag."
        )
    reference_mpp = tagged[0][1]
    assert reference_mpp is not None
    for fp, mpp in tagged[1:]:
        assert mpp is not None
        if abs(mpp - reference_mpp) > _MPP_AGREEMENT_TOLERANCE_UM:
            conflict_summary = ", ".join(
                f"{fp.name}={m:.6f}" for fp, m in tagged[:5]
            )
            if len(tagged) > 5:
                conflict_summary += f", ... ({len(tagged) - 5} more)"
            raise ReaderError(
                f"Per-file mpp disagrees: {conflict_summary}. "
                f"Provide an mpp override: a per-image metadata CSV "
                f"(`mpp` column) or the global `--mpp` flag."
            )
    return reference_mpp
