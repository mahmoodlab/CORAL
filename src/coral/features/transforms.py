"""Shared image transforms for RGB-style encoders (pseudo-RGB + ImageNet).

The **only** genuinely shared encoder utility: UNI and DINOv2 both collapse
a multiplex patch to the *same* fixed pseudo-RGB blend and ImageNet-normalise
it. Everything else (Eva's gene mapping, CA-MAE's native channels) is
each model's own preprocessing, lifted standalone — not abstracted here.

Torch-free at import: the functions operate on whatever ``(B, C, H, W)``
tensor they are handed (via tensor methods like ``new_zeros``), so this
module loads and type-checks without the ``uni``/``kronos2`` extras.
"""

from __future__ import annotations

from typing import Any

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Pseudo-RGB colormap (channel index -> RGB), vendored verbatim from the
# reference ``multiplex_to_rgb``. Order-dependent + caps the panel at 60.
_COLORMAP: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.5, 0.0),
    (0.5, 0.0, 1.0),
    (0.5, 1.0, 0.0),
    (1.0, 0.0, 0.5),
    (0.0, 0.5, 1.0),
    (0.5, 1.0, 1.0),
    (1.0, 0.5, 1.0),
    (0.5, 0.5, 0.5),
    (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.5),
    (0.25, 0.25, 0.75),
    (0.75, 0.25, 0.25),
    (0.75, 0.75, 0.0),
    (0.0, 0.75, 0.75),
    (0.75, 0.0, 0.75),
    (1.0, 0.75, 0.0),
    (0.0, 1.0, 0.75),
    (0.75, 0.0, 1.0),
    (1.0, 0.25, 0.0),
    (0.0, 1.0, 0.25),
    (0.25, 0.0, 1.0),
    (1.0, 0.0, 0.75),
    (0.0, 0.75, 1.0),
    (0.75, 1.0, 0.0),
    (1.0, 0.75, 0.75),
    (0.75, 1.0, 0.75),
    (0.75, 0.75, 1.0),
    (0.9, 0.9, 0.9),
    (0.3, 0.3, 0.3),
    (0.6, 0.4, 0.2),
    (0.2, 0.6, 0.4),
    (0.4, 0.2, 0.6),
    (0.8, 0.2, 0.2),
    (0.2, 0.8, 0.2),
    (0.2, 0.2, 0.8),
    (0.8, 0.8, 0.2),
    (0.8, 0.2, 0.8),
    (0.2, 0.8, 0.8),
    (0.6, 0.3, 0.0),
    (0.0, 0.6, 0.3),
    (0.3, 0.0, 0.6),
    (0.6, 0.0, 0.3),
    (0.0, 0.3, 0.6),
    (0.3, 0.6, 0.0),
    (0.9, 0.6, 0.3),
    (0.3, 0.9, 0.6),
    (0.6, 0.3, 0.9),
    (0.9, 0.3, 0.6),
    (0.3, 0.6, 0.9),
    (0.6, 0.9, 0.3),
    (0.7, 0.5, 0.3),
    (0.3, 0.7, 0.5),
)

MAX_RGB_CHANNELS = len(_COLORMAP)


def multiplex_to_rgb(image_batch: Any) -> Any:  # noqa: ANN401 — torch tensor
    """Blend multiplex channels into a pseudo-RGB image.

    ``rgb = Σ_i (channel_i · colormap_i) / C`` over the ``C`` channels of a
    ``(B, C, H, W)`` torch tensor → ``(B, 3, H, W)``. Vendored verbatim from
    the reference; the result depends on **channel order**.

    Args:
        image_batch: ``(B, C, H, W)`` float tensor, ``C ≤ 60``.

    Returns:
        ``(B, 3, H, W)`` pseudo-RGB tensor (same device + dtype).

    Raises:
        ValueError: If ``C`` exceeds the 60-colour colormap.

    Example:
        >>> import importlib.util
        >>> if importlib.util.find_spec("torch"):  # doctest: +SKIP
        ...     import torch
        ...
        ...     multiplex_to_rgb(torch.ones((1, 2, 1, 1))).shape
        torch.Size([1, 3, 1, 1])
    """
    c = image_batch.shape[1]
    if c > MAX_RGB_CHANNELS:
        msg = f"pseudo-RGB caps at {MAX_RGB_CHANNELS} channels, got {c}"
        raise ValueError(msg)
    b, _, h, w = image_batch.shape
    rgb = image_batch.new_zeros((b, 3, h, w))  # same device + dtype
    for i in range(c):
        color = image_batch.new_tensor(_COLORMAP[i]).view(1, 3, 1, 1)
        rgb += image_batch[:, i, :, :].unsqueeze(1) * color
    return rgb / float(c)


def colorize_per_channel(image_batch: Any) -> Any:  # noqa: ANN401 — tensor
    """Colorize each channel into its own pseudo-RGB image (no blend).

    Where :func:`multiplex_to_rgb` blends every channel into one image
    (``mean_i channel_i · colormap_i``), this keeps them separate: channel
    ``i`` becomes ``channel_i · colormap_i`` as a standalone ``(B, 3, H, W)``
    image, stacked along a new axis → ``(B, C, 3, H, W)``. Used by the
    post-average encoders, which run the backbone on each channel's image and
    average the embeddings afterwards. Same order-dependent colormap; **no**
    ÷C (that averaging happens later, in embedding space).

    Args:
        image_batch: ``(B, C, H, W)`` float tensor, ``C ≤ 60``.

    Returns:
        ``(B, C, 3, H, W)`` pseudo-RGB tensor (same device + dtype).

    Raises:
        ValueError: If ``C`` exceeds the 60-colour colormap.

    Example:
        >>> import importlib.util
        >>> if importlib.util.find_spec("torch"):  # doctest: +SKIP
        ...     import torch
        ...
        ...     colorize_per_channel(torch.ones((1, 2, 4, 4))).shape
        torch.Size([1, 2, 3, 4, 4])
    """
    c = image_batch.shape[1]
    if c > MAX_RGB_CHANNELS:
        msg = f"pseudo-RGB caps at {MAX_RGB_CHANNELS} channels, got {c}"
        raise ValueError(msg)
    b, _, h, w = image_batch.shape
    out = image_batch.new_zeros((b, c, 3, h, w))  # same device + dtype
    for i in range(c):
        color = image_batch.new_tensor(_COLORMAP[i]).view(1, 3, 1, 1)
        out[:, i] = image_batch[:, i, :, :].unsqueeze(1) * color
    return out


def imagenet_normalize(rgb: Any) -> Any:  # noqa: ANN401 — torch tensor
    """ImageNet-normalise a ``(B, 3, H, W)`` RGB torch tensor.

    Example:
        >>> import importlib.util
        >>> if importlib.util.find_spec("torch"):  # doctest: +SKIP
        ...     import torch
        ...
        ...     imagenet_normalize(torch.zeros((1, 3, 2, 2))).shape
        torch.Size([1, 3, 2, 2])
    """
    mean = rgb.new_tensor(_IMAGENET_MEAN)
    std = rgb.new_tensor(_IMAGENET_STD)
    return (rgb - mean[None, :, None, None]) / std[None, :, None, None]
