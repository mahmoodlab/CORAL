"""LoRA (Low-Rank Adaptation) adapters for parameter-efficient fine-tuning.

Fully unfreezing even a *single* ViT-Base transformer block costs ~7.1M
parameters (qkv 1.77M + proj 0.59M + mlp.fc1 2.36M + mlp.fc2 2.36M). LoRA
instead freezes each ``nn.Linear`` weight ``W`` and learns a low-rank
update ``delta = (alpha / r) * B @ A`` with ``A`` of shape ``(r, in)`` and
``B`` of shape ``(out, r)``. That costs only ``r * (in + out)`` parameters
per adapted layer, so a handful of blocks fit in a ~600-800k budget —
roughly 0.7% of the backbone.

Self-contained (no ``peft`` dependency) and purely additive: adapters are
installed by *wrapping* a module's ``nn.Linear`` children, so the original
backbone code and pretrained weights are left untouched.

Kept out of the notebook so the tutorial cells stay focused on the
fine-tuning protocol rather than the adapter plumbing.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: Linear layers adapted inside each transformer block, as attribute paths
#: relative to the block. Valid for a DINOv2-style block with an MLP FFN
#: (a SwiGLU FFN would name its layers differently).
DEFAULT_LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank residual.

    The base layer's weight and bias are frozen; only the low-rank factors
    ``lora_A`` and ``lora_B`` are trainable. ``lora_B`` is zero-initialised
    so the wrapper is an **exact identity** with respect to the base layer
    at step 0 — fine-tuning therefore starts from the pretrained function
    rather than from a perturbed one.

    Args:
        base: The pretrained linear layer to adapt (frozen in place).
        rank: LoRA rank ``r``; the inner dimension of the update.
        alpha: Scaling numerator; the effective scale is ``alpha / rank``.
        dropout: Dropout applied to the adapter's input.

    Raises:
        ValueError: If ``rank`` is not positive.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> base = nn.Linear(8, 4)
        >>> adapted = LoRALinear(base, rank=2, alpha=4)
        >>> x = torch.ones(1, 8)
        >>> bool(torch.equal(adapted(x), base(x)))  # identity at init
        True
        >>> [n for n, p in adapted.named_parameters() if p.requires_grad]
        ['lora_A', 'lora_B']
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        """Freeze ``base`` and attach zero-initialised low-rank factors."""
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        # Freeze the pretrained projection: gradients only reach A and B.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank

        # Match the base weight's dtype/device so this is a drop-in.
        factory = {"dtype": base.weight.dtype, "device": base.weight.device}
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, **factory)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, **factory)
        )
        # Kaiming init on A, as in the LoRA paper; B stays zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base projection plus the scaled low-rank update."""
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + self.scaling * delta

    def extra_repr(self) -> str:
        """Show rank and effective scale in the module repr."""
        return f"rank={self.rank}, scaling={self.scaling:.4g}"


def _get_submodule(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    """Resolve a dotted attribute path to ``(parent_module, leaf_name)``."""
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Return the backbone's transformer blocks in depth order.

    A block is identified **structurally** — a module exposing both
    ``attn`` and ``mlp`` children — which is robust to whether the model
    stores them in a flat ``ModuleList`` or in chunks. KRONOS2 uses
    ``block_chunks=4``, so a naive ``model.blocks`` walk would find the
    four chunks rather than the twelve blocks.

    Args:
        model: The ViT backbone to inspect.

    Returns:
        The transformer blocks, shallowest first.

    Raises:
        ValueError: If no blocks are found.

    Example:
        A ViT-Base backbone yields its twelve blocks in order, so
        ``iter_transformer_blocks(backbone)[-6:]`` is the deepest half.
    """
    blocks = [
        m for m in model.modules() if hasattr(m, "attn") and hasattr(m, "mlp")
    ]
    if not blocks:
        raise ValueError(
            "Could not locate any transformer blocks (modules with .attn "
            "and .mlp) on the backbone; cannot inject LoRA."
        )
    return blocks


def inject_lora_last_blocks(
    model: nn.Module,
    num_blocks: int,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    targets: Sequence[str] = DEFAULT_LORA_TARGETS,
) -> list[str]:
    """Inject LoRA adapters into the last ``num_blocks`` blocks, in place.

    Blocks are taken from the **end** of the network, the standard choice
    for partial fine-tuning: the deepest blocks carry the most
    task-specific, most adaptable representation, while early blocks
    encode generic low-level structure worth keeping frozen.

    Args:
        model: The backbone to adapt (mutated in place).
        num_blocks: How many of the final blocks to adapt. Clamped to the
            backbone's depth, with a warning.
        rank: LoRA rank ``r``.
        alpha: LoRA scaling numerator (effective scale is ``alpha / rank``).
        dropout: Dropout applied to each adapter's input.
        targets: Dotted attribute paths, relative to a block, of the
            ``nn.Linear`` layers to wrap. Non-linear targets are skipped
            with a warning.

    Returns:
        Fully-qualified names of the wrapped layers, for logging.

    Example:
        Adapting the last six blocks of a ViT-Base at rank 8 on all four
        default targets wraps 24 layers and costs ~590k parameters::

            wrapped = inject_lora_last_blocks(
                backbone,
                num_blocks=6,
                rank=8,
                alpha=16,
            )
    """
    blocks = iter_transformer_blocks(model)
    depth = len(blocks)
    if num_blocks <= 0:
        return []
    if num_blocks > depth:
        logger.warning(
            "num_blocks=%d exceeds backbone depth %d; clamping to %d.",
            num_blocks,
            depth,
            depth,
        )
        num_blocks = depth

    wrapped: list[str] = []
    for offset, block in enumerate(blocks[depth - num_blocks :]):
        block_idx = depth - num_blocks + offset
        for target in targets:
            parent, leaf = _get_submodule(block, target)
            base_layer = getattr(parent, leaf)
            if not isinstance(base_layer, nn.Linear):
                logger.warning(
                    "Skipping LoRA target block%d.%s: expected nn.Linear, "
                    "got %s.",
                    block_idx,
                    target,
                    type(base_layer).__name__,
                )
                continue
            setattr(
                parent,
                leaf,
                LoRALinear(
                    base_layer, rank=rank, alpha=alpha, dropout=dropout
                ),
            )
            wrapped.append(f"block{block_idx}.{target}")
    return wrapped


def freeze_all(model: nn.Module) -> None:
    """Freeze every parameter in ``model`` (call *before* injecting LoRA).

    Example:
        >>> from torch import nn
        >>> m = nn.Linear(4, 4)
        >>> freeze_all(m)
        >>> any(p.requires_grad for p in m.parameters())
        False
    """
    for param in model.parameters():
        param.requires_grad_(False)


def trainable_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    """Collect the ``requires_grad`` parameters across ``modules``.

    Example:
        >>> from torch import nn
        >>> m = nn.Linear(4, 4)
        >>> len(trainable_parameters(m))  # weight + bias
        2
    """
    params: list[nn.Parameter] = []
    for module in modules:
        params.extend(p for p in module.parameters() if p.requires_grad)
    return params


def count_trainable_parameters(*modules: nn.Module) -> int:
    """Total number of trainable scalars across ``modules``.

    Example:
        >>> from torch import nn
        >>> count_trainable_parameters(nn.Linear(4, 4))  # 16 + 4
        20
    """
    return sum(p.numel() for p in trainable_parameters(*modules))
