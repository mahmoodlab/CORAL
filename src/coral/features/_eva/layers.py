# -*- coding: utf-8 -*-
"""
@Time   :  2025/03/19 17:50
@Author :  Yufan Liu
@Desc   :  Some modules and layers for the model
"""


import torch
import torch.nn as nn
from timm.layers import Mlp
from timm.models.vision_transformer import Attention, Block
from torch.nn import functional as F

from coral.markers._eva_marker_to_gene import EVA_MARKER_TO_GENE as marker_to_gene


# ----------------------------- Marker Embeddings ---------------------------- #
class MarkerEmbeddingGenePT(nn.Module):
    """GenePT-based marker embedding module that utilizes pre-computed GenePT embeddings for markers.

    This module maps marker names to their corresponding dense vector representations using
    pre-computed GenePT embeddings. For markers without GenePT embeddings, it uses learned
    embeddings initialized with Xavier uniform initialization.

    Args:
        marker_dict: Dictionary containing pre-computed GenePT embeddings for markers.
        unknown_marker_embed_dim: Dimension of embeddings for unknown markers.
                                  Defaults to 3072 (GenePT embedding dimension).

    Note:
        - The module pre-initializes embedding layers for all markers that don't have GenePT embeddings
        - All embeddings are initialized using Xavier uniform initialization
    """

    def __init__(self, marker_dict, unknown_marker_embed_dim=3072):
        super().__init__()
        self.genept_embeddings = marker_dict
        self.unknown_marker_embeddings = nn.ModuleDict()
        self.unknown_marker_embed_dim = unknown_marker_embed_dim
        self.register_buffer("_device_tracker", torch.empty(0))  # For robust device tracking

        # Initialize embeddings for all markers that don't have GenePT embeddings
        for marker_name in marker_to_gene.keys():
            m_gene = marker_to_gene[marker_name]
            if m_gene not in self.genept_embeddings:
                self.unknown_marker_embeddings[marker_name] = nn.Embedding(1, self.unknown_marker_embed_dim)
                nn.init.xavier_uniform_(self.unknown_marker_embeddings[marker_name].weight)

    def forward(self, marker_names):
        """Generate embeddings for a list of marker names.

        Args:
            marker_names: List of marker names to embed

        Returns:
            Marker embeddings of shape [num_markers, embedding_dim]
            where embedding_dim is either GenePT dimension (3072) for known markers
            or unknown_marker_embed_dim for unknown markers
        """
        target_device = self._device_tracker.device

        embeddings = []
        for m in marker_names:
            m_gene = marker_to_gene[m]
            if m_gene in self.genept_embeddings:
                emb_tensor = torch.tensor(self.genept_embeddings[m_gene], device=target_device, dtype=torch.float)
                embeddings.append(emb_tensor)
            else:
                idx_tensor = torch.zeros(1, dtype=torch.long, device=target_device)
                embeddings.append(self.unknown_marker_embeddings[m](idx_tensor).squeeze(0))

        final_embeddings = torch.stack(embeddings)
        return final_embeddings


# ---------------------------------------------------------------------------- #

# --------------------------- Neural network layers -------------------------- #
class MaskedAttention(Attention):
    """Attention mechanism with optional masking.

    Extends Attention module to support attention masking for preventing attention to certain positions.

    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads. Defaults to 8.
        qkv_bias: Whether to include bias in qkv projections. Defaults to False.
        qk_norm: Whether to normalize query and key. Defaults to False.
        proj_bias: Whether to include bias in output projection. Defaults to True.
        attn_drop: Dropout rate for attention weights. Defaults to 0.0.
        proj_drop: Dropout rate for output projection. Defaults to 0.0.
        norm_layer: Normalization layer to use. Defaults to nn.LayerNorm.
        fused_attn: Whether to use fused attention. Defaults to True.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        fused_attn=True,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.fused_attn = fused_attn

    def forward(self, x, attn_mask=None):
        """Forward pass with optional attention masking.

        Args:
            x: Input tensor [B, N, C]
            attn_mask: Optional attention mask. Defaults to None.

        Returns:
            Output tensor [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if attn_mask is not None:
            attn_mask = attn_mask == 0
            
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=attn_mask
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedBlock(Block):
    """Transformer block with masked attention support.

    Extends Block module to use MaskedAttention layer for attention masking.

    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
        qkv_bias: Whether to include bias in qkv projections. Defaults to False.
        qk_norm: Whether to normalize query and key. Defaults to False.
        proj_bias: Whether to include bias in output projection. Defaults to True.
        proj_drop: Dropout rate for output projection. Defaults to 0.0.
        attn_drop: Dropout rate for attention weights. Defaults to 0.0.
        init_values: Initial value for LayerScale. Defaults to None.
        drop_path: Stochastic depth rate. Defaults to 0.0.
        act_layer: Activation layer. Defaults to nn.GELU.
        norm_layer: Normalization layer. Defaults to nn.LayerNorm.
        mlp_layer: MLP layer. Defaults to Mlp.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        self.attn = MaskedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x, attn_mask=None):
        """Forward pass with optional attention masking.

        Args:
            x: Input tensor [B, N, C]
            attn_mask: Optional attention mask. Defaults to None.

        Returns:
            Output tensor [B, N, C]
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbedChannelFree(nn.Module):
    """Channel agnostic patch embedding module that applies the same 2D convolution to each channel.
    Each channel is processed independently with the same convolution weights.
    The number of input channels can be arbitrary.
    """

    def __init__(
        self,
        img_size,
        token_size=16,
        embed_dim=256,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.token_size = (token_size, token_size) if isinstance(token_size, int) else token_size
        self.embed_dim = embed_dim

        # Create a single conv layer that will be applied to each channel
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=token_size, stride=token_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Calculate grid size and number of patches
        self.grid_size = (self.img_size[0] // self.token_size[0], self.img_size[1] // self.token_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        """Forward pass of the PatchEmbedChannelFree module.

        Args:
            x: Input tensor of shape (B, C, H, W),
               where B is batch size, C is number of channels,
               H is height, and W is width.

        Returns:
            Output tensor of shape (B, C, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = x.view(B * C, 1, H, W)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.view(B, C, -1, self.embed_dim)

        x = self.norm(x)
        return x

# ---------------------------------------------------------------------------- #
