"""Download the public cHL CODEX example dataset for the CORAL tutorials.

The dataset is a single classical-Hodgkin-lymphoma (cHL) CODEX region
released with MAPS (Shaban et al., *Nature Communications*, 2024) and
hosted on Zenodo. It ships as one TIFF per marker plus a cell
segmentation mask and a cell-type annotation table:

    cHL_CODEX/
    ├── raw_image/          # one single-channel TIFF per marker
    │   ├── CD3.tiff
    │   ├── DAPI-01.tiff
    │   └── ...
    ├── segmentation/
    │   └── cHL_CODEX_segmentation.tiff
    └── annotation_csv/
        └── cHL_CODEX_annotation.csv

The ``raw_image/`` directory is ingestible by CORAL as-is: point
``convert_to_canonical`` (or ``coral ingest``) at it and CORAL's
per-channel-TIFF-directory reader stacks the files into a canonical
``(c, y, x)`` slide, taking each marker name from its filename stem.

Paper:  https://doi.org/10.1038/s41467-024-45999-1
Data:   https://zenodo.org/records/10067010 (CC-BY-4.0)
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm

_ZENODO_URL = (
    "https://zenodo.org/records/10067010/files/cHL_CODEX.zip?download=1"
)
_ARCHIVE_NAME = "cHL_CODEX.zip"
_EXTRACTED_NAME = "cHL_CODEX"
_DOWNLOAD_SIZE_GB = 3.4


def download_chl_maps_dataset(data_dir: str | Path) -> Path:
    """Download and unzip the cHL CODEX example dataset.

    Fetches the ~3.4 GB archive from Zenodo, extracts it under
    ``data_dir``, and removes the archive to save disk. The call is
    idempotent: if the extracted folder already exists it returns
    immediately without re-downloading.

    Args:
        data_dir: Directory to download and unpack into. Created if
            missing.

    Returns:
        Path to the extracted ``cHL_CODEX/`` directory, whose
        ``raw_image/`` sub-directory can be handed straight to
        ``convert_to_canonical``.

    Example:
        >>> from utils import download_chl_maps_dataset  # doctest: +SKIP
        >>> chl = download_chl_maps_dataset("example-data")  # doctest: +SKIP
        >>> raw_image_dir = chl / "raw_image"               # doctest: +SKIP
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / _EXTRACTED_NAME

    if extracted.exists():
        print(f"Dataset already present at {extracted} — skipping download.")
        return extracted

    archive = root / _ARCHIVE_NAME
    print(
        f"Downloading cHL CODEX dataset (~{_DOWNLOAD_SIZE_GB} GB) from Zenodo."
        "\nThis is a one-time download and may take several minutes."
    )
    _download(_ZENODO_URL, archive)

    print(f"Extracting {archive.name} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    archive.unlink()  # reclaim the ~3.4 GB archive; the extract is enough

    if not extracted.exists():
        raise RuntimeError(
            f"Expected {extracted} after extraction but it is missing; "
            f"the archive layout may have changed."
        )
    print(f"Done. Dataset ready at {extracted}")
    return extracted


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a progress bar, atomically.

    Writes to a ``.part`` sibling first and renames on success so an
    interrupted download never looks complete on the next run.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response:  # noqa: S310 (https only)
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(part, "wb") as handle,
            tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar,
        ):
            shutil.copyfileobj(
                _ProgressReader(response, bar), handle, length=1024 * 1024
            )
    part.rename(dest)


class _ProgressReader:
    """Wrap a file-like object so reads advance a tqdm bar."""

    def __init__(self, raw: object, bar: tqdm) -> None:
        self._raw = raw
        self._bar = bar

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)  # type: ignore[attr-defined]
        self._bar.update(len(chunk))
        return chunk
