"""Local-asset resolution for encoders that need user-supplied files.

A few encoders (CA-MAE, Eva, VIRTUES) load weights or embeddings that are
too large or too license-encumbered to ship with the package. By
convention the user drops those files under ``assets/<EncoderName>/`` at
the repo root, and each encoder's ``build()`` resolves them from there — so
``coral extract --extractor Eva`` works with no extra flags.

A missing file raises a :class:`FileNotFoundError` that names the exact
path to populate and, where known, where to download it — actionable,
unlike the old "needs a checkpoint" :class:`RuntimeError`.
"""

from __future__ import annotations

from pathlib import Path

# assets.py → coral → src → <repo root>; the assets/ dir lives at the root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def assets_root() -> Path:
    """The assets directory — ``<repo root>/assets``.

    Returns:
        The root under which per-encoder asset dirs live. Not guaranteed to
        exist — :func:`require_asset` is what enforces presence.

    Example:
        >>> assets_root().name
        'assets'
    """
    return _REPO_ROOT / "assets"


def model_assets_dir(model: str) -> Path:
    """The per-encoder asset dir ``<assets_root>/<model>``.

    Args:
        model: The encoder's registry name (e.g. ``"Eva"``, ``"CA-MAE"``).

    Returns:
        The directory the encoder reads its local files from.

    Example:
        >>> model_assets_dir("Eva").name
        'Eva'
    """
    return assets_root() / model


def require_asset(
    model: str,
    relpath: str,
    *,
    what: str,
    source: str | None = None,
) -> Path:
    """Resolve ``<assets>/<model>/<relpath>`` or raise if it is absent.

    Args:
        model: The encoder name (its ``assets/`` subdir).
        relpath: The file (or dir) expected under that subdir.
        what: Human description of the asset, used in the error message.
        source: Optional where-to-get-it hint (e.g. a Zenodo record).

    Returns:
        The resolved, existing path.

    Raises:
        FileNotFoundError: If the path does not exist, naming it and how to
            populate it.

    Example:
        >>> try:
        ...     require_asset("Demo", "weights.pkl", what="the weights")
        ... except FileNotFoundError as e:
        ...     "weights.pkl" in str(e)
        True
    """
    path = model_assets_dir(model) / relpath
    if not path.exists():
        hint = f" Download it from {source}." if source else ""
        raise FileNotFoundError(
            f"{model} needs {what} at {path}, which does not exist. Place "
            f"it under assets/{model}/ in the CORAL repo root.{hint}"
        )
    return path
