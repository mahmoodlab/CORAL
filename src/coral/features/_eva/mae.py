# -*- coding: utf-8 -*-
"""
@Time   :  2025/05/04 17:30
@Author :  Yufan Liu
@Desc   :  Two-stage Eva MAE implementation
           Also see: https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""


import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from coral.features._eva.layers import MarkerEmbeddingGenePT
from coral.features._eva.layers import MaskedBlock as Block
from coral.features._eva.layers import PatchEmbedChannelFree
from coral.features._eva.masking import random_masking
from coral.features._eva.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with Vision Transformer for spatial transcriptomics.

    Processes multi-channel spatial data with channel-specific marker embeddings.
    Architecture: Channel Mixer -> Patch Mixer -> Decoder.

    Args:
        conf: Configuration object with model hyperparameters for dataset, encoder, decoder,
             channel mixer, and patch mixer components.
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # ---------------------------------------------------------------------------- #
        # --------------------------- I Encoder components --------------------------- #
        # ---------------------------------------------------------------------------- #

        # ----------------------------- 1. Channel Former ---------------------------- #

        # ------------------------ patchify and embed patches ------------------------ #
        self.patch_embed = PatchEmbedChannelFree(
            img_size=conf.ds.patch_size, token_size=conf.ds.token_size, embed_dim=conf.ds.token_size**2
        )
        # ---------------------------------------------------------------------------- #
        self.num_patches = self.patch_embed.num_patches
        self.channel_proj = nn.Sequential(
            nn.Linear(conf.ds.token_size**2, conf.cm.dim * 2),
            nn.LayerNorm(conf.cm.dim * 2),
            nn.GELU(),
            nn.Linear(conf.cm.dim * 2, conf.cm.dim),
            nn.LayerNorm(conf.cm.dim),
        )
        self.channel_enc_blocks = nn.ModuleList(
            [
                Block(
                    dim=conf.cm.dim,
                    num_heads=conf.cm.n_heads,
                    mlp_ratio=conf.cm.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(conf.cm.n_layers)
            ]
        )
        self.channel_norm = nn.LayerNorm(conf.cm.dim)

        # -------------------------- Masker and strategies -------------------------- #
        self.mask_strategy = conf.ds.mask_strategy
        self.mask_ratio = conf.ds.mask_ratio
        if self.mask_strategy == "specified":
            # For specified strategy, we need to pass channels parameter during call
            self.masker = random_masking(self.mask_ratio, self.mask_strategy)
            self.mask_channels = list(getattr(conf.ds, "mask_channels"))
        else:
            self.masker = random_masking(self.mask_ratio, self.mask_strategy)
        # ---------------------------------------------------------------------------- #

        # ----------------------------- Marker Embeddings ---------------------------- #
        self.marker_cls_token = nn.Parameter(torch.zeros(1, 1, conf.cm.dim))
        self.marker_dim = conf.ds.marker_dim
        # CORAL vendoring edit: load the GenePT pickle from the path the
        # extractor injects into the conf (assets/Eva/-resolved), instead
        # of the upstream hardcoded CWD-relative "marker_embeddings/..." —
        # same pickle bytes, no CWD symlink.
        with open(conf.ds.genept_pkl, "rb") as _f:
            marker_dict = pickle.load(_f)
        self.marker_embed = MarkerEmbeddingGenePT(marker_dict, self.marker_dim)
        self.marker_proj = nn.Sequential(
            nn.Linear(self.marker_dim, conf.cm.dim),
            nn.LayerNorm(conf.cm.dim),
        )
        # ---------------------------------------------------------------------------- #

        # ------------------------------ 2. Patch Former ----------------------------- #
        self.linker_proj = nn.Sequential(
            nn.Linear(conf.cm.dim, conf.cm.dim * 2),
            nn.LayerNorm(conf.cm.dim * 2),
            nn.GELU(),
            nn.Linear(conf.cm.dim * 2, conf.pm.dim),
            nn.LayerNorm(conf.pm.dim),
        )  # project Dim from channel to patch

        # Positional embeddings and tokens
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, conf.pm.dim)
        )  # share with decoder as patch-level representation
        self.enc_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, conf.pm.dim), requires_grad=False
        )  # Fixed sin-cos embedding

        # Encoder transformer blocks
        self.patch_enc_blocks = nn.ModuleList(
            [
                Block(
                    dim=conf.pm.dim,
                    num_heads=conf.pm.n_heads,
                    mlp_ratio=conf.pm.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(conf.pm.n_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(conf.pm.dim)
        self.enc_proj = nn.Sequential(
            nn.Linear(conf.pm.dim, conf.pm.out_dim * 2),
            nn.LayerNorm(conf.pm.out_dim * 2),
            nn.GELU(),
            nn.Linear(conf.pm.out_dim * 2, conf.de.dim),
            nn.LayerNorm(conf.de.dim),
        )

        # ---------------------------------------------------------------------------- #

        # ---------------------------------------------------------------------------- #
        # --------------------------- II Decoder components -------------------------- #
        # ---------------------------------------------------------------------------- #

        # self.decoder_embed = nn.Linear(conf.pm.dim, conf.de.dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, conf.de.dim))  # match with BCND

        # ---------------- Positional encoding for patch-level flatten -------------- #
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, conf.de.dim), requires_grad=False)
        self.flatten_dim_mapper = "(B C) N D"
        # ---------------------------------------------------------------------------- #

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=conf.de.dim,
                    num_heads=conf.de.n_heads,
                    mlp_ratio=conf.de.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(conf.de.n_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(conf.de.dim)
        self.decoder_pred = nn.Linear(conf.de.dim, conf.ds.token_size**2, bias=True)
        # ---------------------------------------------------------------------------- #
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize model weights using standard transformer initialization."""
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.enc_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.enc_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize decoder position embeddings (patch-only)
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding weights
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize tokens
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)  # all along the whole model
        torch.nn.init.normal_(self.marker_cls_token, std=0.02)

        # Initialize other layers
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for a specific module.

        Args:
            m: Module to initialize (Linear or LayerNorm).
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _embed_marker(self, markers: list[list], expand_num=None):
        """Embed markers and expand to patch dimensions.

        Args:
            markers: List of marker lists for each channel in batch
            expand_num: Number of patches to expand to. Defaults to self.num_patches.

        Returns:
            Expanded marker embeddings [B, C, N, marker_dim]
        """
        m_embed = []
        for m in markers:
            m_embed.append(self.marker_embed(m))
        m_embed = torch.stack(m_embed, dim=0)  # [B, C, marker_dim]
        if not expand_num:
            expand_num = self.num_patches

        m_embed = m_embed.unsqueeze(2).expand(-1, -1, expand_num, -1)
        return m_embed

    def channel_forward(self, image, marker, channel_mask=None, infer_mask=None):
        """Process input through channel mixer with masking.

        Args:
            image: Input images [B, C, H, W]
            marker: Marker embeddings for each channel
            infer_mask: Pre-defined mask for inference. Defaults to None.

        Returns:
            Tuple of (processed features [B, N, C+1, D], raw mask [C, N] or [B, C, N])
        """
        # define a channel former forward process
        # image: [B, C, H, W]
        B, C, H, W = image.shape
        x = self.patch_embed(image)  # [B, N, P*P*C], with channel flatten, [B, C, N, P*P] with channel-agnostic

        x = self.channel_proj(x)  # [B, C, N, D]

        # Add marker embeddings
        marker_embeddings = self._embed_marker(marker).to(x.device)
        marker_embeddings = self.marker_proj(marker_embeddings)
        x = x + marker_embeddings

        # --------------------------- Generate mask matrix --------------------------- #
        if infer_mask is not None:
            raw_mask = infer_mask
        else:
            if self.mask_strategy == "specified":
                raw_mask = self.masker(x, self.mask_channels)  # Pass channels parameter
            else:
                raw_mask = self.masker(x)  # [C, N] or [B, C, N]

        # channel_mask if provided (for padding)
        if channel_mask is not None:
            # channel_mask: [B, C] -> [B, C, N] to match raw_mask
            channel_mask = channel_mask[..., None].expand(-1, -1, self.num_patches)  # [B, C] to [B, C, N]
            # Combine channel_mask with raw_mask: if either is 1, result is 1
            if raw_mask.dim() == 2:  # [C, N]
                # Expand raw_mask to [B, C, N] to match channel_mask
                raw_mask = raw_mask[None, ...].expand(B, -1, -1)  # [B, C, N]
            # Now both are [B, C, N], combine them
            raw_mask = torch.logical_or(raw_mask.bool(), channel_mask.bool()).float()

        if raw_mask.dim() == 2:  # [C, N]
            cls_mask = torch.zeros([C + 1, self.num_patches], device=x.device)
            cls_mask[1:, :] = raw_mask  # [C+1, N]
            cls_mask = cls_mask[None, ...].permute(0, 2, 1).expand(B, -1, -1)  # [B, N, C+1]
            cls_mask = cls_mask.reshape(-1, C + 1)
            attn_mask = (cls_mask.unsqueeze(1) + cls_mask.unsqueeze(2)).clamp(max=1)  # [B*N, C+1, C+1]
            attn_mask = attn_mask.unsqueeze(1)  # [B*N, 1, C+1, C+1], broadcast to each head
        else:
            raise ValueError(f"raw_mask shape not supported: {raw_mask.shape}")
        # ---------------------------------------------------------------------------- #

        # replace invisible tokens as a unified value, and mask them out in attention layers.
        mask_token = self.mask_token.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        if raw_mask.dim() == 2:
            x = torch.where(raw_mask[None, ..., None] == 1, mask_token.to(x.device), x)
        else:
            raise ValueError(f"raw_mask shape not supported: {raw_mask.shape}")

        x = rearrange(x, "B C N D -> (B N) C D")
        cls_tokens = self.marker_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply channel transformer blocks
        for block in self.channel_enc_blocks:
            x = block(x, attn_mask=attn_mask)  # Process channel-wise relationships
        x = self.channel_norm(x)

        # The return x value will contain cls token, use or not in patch-former
        x = rearrange(x, "(B N) C D -> B N C D", B=B, C=C + 1)
        return x, raw_mask

    def patch_forward(self, input_x):
        """Process features through patch mixer.

        Args:
            input_x: Features from channel mixer [B, N, C + 1, D]

        Returns:
            Processed features [B, C, N+1, D] (before enc_proj)
        """

        x = input_x[:, :, 0:1, :]
        x = self.linker_proj(x)
        B, N, C, D = x.shape
        x = rearrange(x, "B N C D -> (B C) N D", B=B, N=N, C=C, D=D)
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([x, cls_token], dim=1)
        x = x + self.enc_pos_embed

        for blk in self.patch_enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        x = rearrange(x, "(B C) N D -> B C N D", B=B, C=C, N=N + 1)
        return x

    def forward_encoder(self, image, marker, channel_mask=None, infer_mask=None):
        """Complete forward pass through encoder.

        Args:
            image: Input images [B, C, H, W]
            marker: Marker embeddings for each channel
            infer_mask: Pre-defined mask for inference. Defaults to None.

        Returns:
            Tuple of (encoded features [B, C, N+1, D], raw mask [C, N] or [B, C, N])
        """
        x, raw_mask = self.channel_forward(image, marker, channel_mask, infer_mask)
        x = self.patch_forward(x)
        return x, raw_mask

    def forward_decoder(self, x: torch.Tensor, marker, channel_mask=None) -> torch.Tensor:
        """Reconstruct input from encoded representation.

        Args:
            x: Encoded representation [B, C, N+1, D] (before enc_proj)
            marker: Marker embeddings for each channel

        Returns:
            Reconstructed patches [B, C, N+1, token_size**2]
        """
        x = self.enc_proj(x)
        B, C, N, D = x.shape
        x = x.repeat(1, len(marker[0]), 1, 1)
        C = len(marker[0])
        marker_embeddings = self._embed_marker(marker, expand_num=N).to(x.device)
        marker_embeddings = self.marker_proj(marker_embeddings)
        x = x + marker_embeddings

        # --------------------------- Generate attention mask for decoder --------------------------- #
        if channel_mask is not None:
            # channel_mask: [B, C] -> [B, C, N] to match the decoder input shape
            channel_mask = channel_mask[..., None].expand(-1, -1, N)  # [B, C] to [B, C, N]
            # For patch flatten: [B, C, N] -> [B*C, N] -> [B*C*N, N]
            channel_mask = channel_mask.reshape(B * C, N)  # [B*C, N]
            channel_mask = channel_mask.reshape(-1, N)  # [B*C*N, N]
            attn_mask = (channel_mask.unsqueeze(1) + channel_mask.unsqueeze(2)).clamp(max=1)  # [B*C*N, N, N]
            attn_mask = attn_mask.unsqueeze(1)  # [B*C*N, 1, N, N]
        else:
            attn_mask = None
        # ---------------------------------------------------------------------------- #

        x = rearrange(x, f"B C N D -> {self.flatten_dim_mapper}")
        # project encoder output to match decoder
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = rearrange(x, f"{self.flatten_dim_mapper} -> B C N D", N=N, C=C, B=B)
        return x

    def forward(
        self, imgs: torch.Tensor, marker_in, channel_mask=None, marker_out=None, infer_mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complete forward pass through MAE model.

        Args:
            imgs: Input images [B, C, H, W]
            marker_in: Input marker embeddings for encoding
            marker_out: Output marker embeddings for decoding. Defaults to marker_in.
            infer_mask: Pre-defined mask for inference. Defaults to None.

        Returns:
            Tuple of (reconstructed patches [B, C, N+1, token_size**2], mask [C, N] or [B, C, N])
        """
        encoder_x, raw_mask = self.forward_encoder(
            image=imgs, marker=marker_in, channel_mask=channel_mask, infer_mask=infer_mask
        )
        if not marker_out:
            marker_out = marker_in
        pred = self.forward_decoder(encoder_x, marker_out, channel_mask)
        return pred, raw_mask
