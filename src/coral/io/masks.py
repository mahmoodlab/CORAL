"""Readers + validation for user-provided cell / tissue masks.

Ingests externally-produced masks into the canonical slide
store. Masks arrive as image files (instance-label TIFFs, binary PNGs);
these helpers read one to a 2-D array and validate it against the
slide's level-0 geometry before the slide writers persist it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_mask_image", "validate_mask_shape"]


def read_mask_image(path: str | Path) -> np.ndarray:
    """Read a mask image file to a 2-D numpy array.

    ``.tif``/``.tiff`` are read via ``tifffile``; everything else
    (``.png``, …) via Pillow. A 3-D ``(H, W, C)`` image — e.g. a
    grayscale mask saved as RGB — is reduced to its first channel.

    Args:
        path: Path to the mask image (instance-label or binary).

    Returns:
        A 2-D array of the mask's pixel values.

    Example:
        >>> import tempfile, numpy as np, tifffile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "m.tiff"
        ...     tifffile.imwrite(p, np.array([[0, 5], [9, 0]], dtype="int32"))
        ...     read_mask_image(p).tolist()
        [[0, 5], [9, 0]]
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        arr = np.asarray(tifffile.imread(path))
    else:
        from PIL import Image

        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:  # grayscale-as-RGB → first channel
        arr = arr[..., 0]
    return arr


def validate_mask_shape(mask: np.ndarray, height: int, width: int) -> None:
    """Raise if ``mask`` isn't 2-D matching level-0 ``(height, width)``.

    Args:
        mask: The mask array to validate.
        height: Level-0 image height (y extent), in pixels.
        width: Level-0 image width (x extent), in pixels.

    Raises:
        ValueError: If ``mask`` is not 2-D, or its shape isn't
            ``(height, width)``.

    Example:
        >>> import numpy as np
        >>> validate_mask_shape(np.zeros((4, 4)), 4, 4)  # ok
        >>> try:
        ...     validate_mask_shape(np.zeros((4, 5)), 4, 4)
        ... except ValueError:
        ...     print("bad shape")
        bad shape
    """
    if mask.ndim != 2 or mask.shape != (height, width):
        msg = (
            f"mask shape {mask.shape} does not match the slide's level-0 "
            f"(height, width) = ({height}, {width})."
        )
        raise ValueError(msg)
