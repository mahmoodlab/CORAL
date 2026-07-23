# -*- coding: utf-8 -*-
"""
@Time   :  2025/11/14 01:15
@Author :  Yufan Liu
@Desc   :  Eva model for inference
"""

import torch
import torch.nn as nn
from einops import rearrange

from coral.features._eva.mae import MaskedAutoencoderViT


class EvaMAE(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.token_size = conf.ds.token_size
        self.img_size = conf.ds.patch_size
        self.model = MaskedAutoencoderViT(conf)

    def forward(self, img, marker_in, channel_mask=None, marker_out=None, infer_mask=None):
        img = img.permute(0, 3, 1, 2)
        image_recon_cls, raw_mask = self.model.forward(
            imgs=img, marker_in=marker_in, channel_mask=channel_mask, marker_out=marker_out, infer_mask=infer_mask
        )
        image_recon = image_recon_cls[:, :, 1:, :]
        image_cls = image_recon_cls[:, :, 0, :]

        image_recon = rearrange(
            image_recon,
            "N C (H W) (P1 P2) -> N (H P1) (W P2) C",
            P1=self.token_size,
            P2=self.token_size,
            H=self.img_size // self.token_size,
            N=image_recon_cls.shape[0],
        )

        return image_recon, image_cls, raw_mask

    def recon(self, img, marker_in, channel_mask=None, marker_out=None, infer_mask=None):
        return self.forward(img, marker_in, channel_mask, marker_out, infer_mask)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, conf, device="cpu"):
        model = cls(conf)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        device = torch.device(device)
        model = model.to(device)
        
        return model

    def extract_features(self, patch, bms, device, cls=False, channel_mode="full"):
        """Extract features (embeddings) from the model.

        Args:
            patch: Input patch tensor of shape [B, H, W, C]
            bms: Biomarkers information
            device: Device to run the model on
            cls: Whether to use CLS token or average patches
            channel_mode: Channel selection mode ("full", "HE", "MIF")

        Returns:
            feat: Extracted features (embeddings) of shape [B, D]
        """
        x = patch.permute(0, 3, 1, 2).float().to(device)

        if channel_mode == "HE":
            x = x[:, -3:, :, :]
            if bms is not None:
                if isinstance(bms, list) and len(bms) > 0:
                    if isinstance(bms[0], list):
                        bms = [bm[-3:] for bm in bms]
                    else:
                        bms = bms[-3:]
        elif channel_mode == "MIF":
            x = x[:, :-3, :, :]
            if bms is not None:
                if isinstance(bms, list) and len(bms) > 0:
                    if isinstance(bms[0], list):
                        bms = [bm[:-3:] for bm in bms]
                    else:
                        bms = bms[:-3:]

        image_out, raw_mask = self.model.forward_encoder(x, bms)
        if cls:
            image_cls = image_out[:, :, 0, :]
        else:
            image_cls = image_out[:, :, 1:, :].mean(2)
        image_cls = image_cls.squeeze(1)
        batch_size = image_cls.size(0)
        feat = image_cls.view(batch_size, -1)
        return feat
