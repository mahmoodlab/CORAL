import torch
from torch import nn, Tensor
import numpy as np
import math
from typing import Literal

class RelativePositionBias(nn.Module):
    """
    Global 2D relative position bias for a max grid of (H_max, W_max).
    Works for smaller H,W by subsetting.
    """
    def __init__(self, num_heads: int, window_size: tuple[int, int]):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size  # (H_max, W_max)
        H, W = window_size

        num_rel = (2 * H - 1) * (2 * W - 1)
        self.bias_table = nn.Parameter(torch.zeros(num_rel, num_heads))

        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, H, W)
        coords_flat = coords.view(2, -1)  # (2, H*W)

        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel_coords[0] += H - 1
        rel_coords[1] += W - 1
        rel_coords[0] *= (2 * W - 1)
        rel_pos_index = rel_coords[0] + rel_coords[1]  # (N, N)

        self.register_buffer("rel_pos_index", rel_pos_index, persistent=False)

    def forward(self,
                H: int,
                W: int,
                has_cls_token: bool = True) -> torch.Tensor:
        """
        Returns (num_heads, N, N) bias for current grid HxW (+ optional CLS).
        """
        H_max, W_max = self.window_size
        assert H <= H_max and W <= W_max
        device = self.bias_table.device

        # indices for top-left HxW window within max grid
        idx_h = torch.arange(H, device=device)
        idx_w = torch.arange(W, device=device)
        idx = (idx_h[:, None] * W_max + idx_w[None, :]).reshape(-1)  # (H*W,)

        rel_idx = self.rel_pos_index[idx][:, idx]  # (N_img, N_img)
        bias = self.bias_table[rel_idx.view(-1)].view(
            H * W, H * W, self.num_heads
        )  # (N_img, N_img, heads)
        bias = bias.permute(2, 0, 1)  # (heads, N_img, N_img)

        if has_cls_token:
            # pad row/col for CLS with zeros
            zeros_row = bias.new_zeros(self.num_heads, 1, H * W)
            zeros_col = bias.new_zeros(self.num_heads, H * W + 1, 1)
            bias = torch.cat([zeros_row, bias], dim=1)  # (heads, N_img+1, N_img)
            bias = torch.cat([zeros_col, bias], dim=2)  # (heads, N_img+1, N_img+1)

        return bias


# Official implementation from DINOv3
# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods
