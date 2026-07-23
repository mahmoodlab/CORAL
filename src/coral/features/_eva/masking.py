# -*- coding: utf-8 -*-
"""
@Time   :  2025/03/27 09:42
@Author :  Yufan Liu
@Desc   :  Maksing strategies
"""
import torch

def random_masking(ratio: float, strategy):
    """Generate masks for reconstruction based on specified strategy.

    Args:
        ratio: Masking ratio
        strategy: Masking strategy:
            - random: Randomly mask 2D content [C, N]
            - patch: Mask all channels of selected patches
            - channel: Mask all patches of a channel
            - he: Mask all H&E channels (last 3)
            - mif: Mask all MIF channels
            - specified: Mask specified channels (requires channels list during call)
    """

    def random_mask(x, ratio=ratio):
        """Randomly mask 2D content across [C, N] plane.

        Args:
            x (Tensor): Input tensor of shape [B, C, N, D]
            ratio (float): Masking ratio between 0 and 1

        Returns:
            Tensor: Binary mask of shape [C, N] 
        """
        B, C, N, D = x.shape
        device = x.device

        noise = torch.rand(C, N, device=device)

        num_elements = C * N
        num_keep = int(num_elements * (1 - ratio))

        ids_shuffle = torch.argsort(noise.reshape(-1))

        mask = torch.ones([C, N], device=device)
        mask_flat = mask.reshape(-1)
        mask_flat[ids_shuffle[:num_keep]] = 0
        mask = mask_flat.reshape(C, N)

        return mask

    def patch_mask(x, ratio=ratio):
        """Mask all channels of selected patches.

        Args:
            x (Tensor): Input tensor of shape [B, C, N, D]
            ratio (float): Masking ratio between 0 and 1

        Returns:
            Tensor: Binary mask of shape [C, N]
        """
        B, C, N, D = x.shape
        device = x.device

        noise = torch.rand(N, device=device)

        num_elements = N
        num_keep = int(num_elements * (1 - ratio))

        ids_shuffle = torch.argsort(noise)
        ids_keep = ids_shuffle[:num_keep]

        mask = torch.ones([N], device=device)
        mask[ids_keep] = 0

        mask = mask.unsqueeze(0).expand(C, -1)  # [C, N]

        return mask

    def channel_mask(x, ratio=ratio):
        """Mask all patches of selected channels.

        Args:
            x (Tensor): Input tensor of shape [B, C, N, D]
            ratio (float or int): If float (0-1), use as masking ratio; if int, use as number of channels to mask

        Returns:
            Tensor: Binary mask of shape [C, N] 
        """
        B, C, N, D = x.shape
        device = x.device

        noise = torch.rand(C, device=device)

        if isinstance(ratio, int):
            # Use as number of channels to mask
            num_keep = C - ratio
        else:
            # Use as ratio
            num_keep = int(C * (1 - ratio))

        ids_shuffle = torch.argsort(noise)
        ids_keep = ids_shuffle[:num_keep]

        mask = torch.ones([C], device=device)
        mask[ids_keep] = 0

        mask = mask.unsqueeze(1).expand(-1, N)  # [C, N]

        return mask

    def he_mask(x, ratio=ratio):
        """Mask H&E channels (last 3 channels).

        Args:
            x (Tensor): Input tensor of shape [B, C, N, D]
            ratio (float): Ignored for this strategy

        Returns:
            Tensor: Binary mask of shape [C, N]
        """
        B, C, N, D = x.shape
        device = x.device
        mask = torch.ones([C], device=device)
        mask[:-3] = 0

        mask = mask.unsqueeze(1).expand(-1, N)  # [C, N]

        return mask

    def mif_mask(x, ratio=ratio):
        """Mask MIF channels (all except last 3 channels).

        Args:
            x (Tensor): Input tensor of shape [B, C, N, D]
            ratio (float): Ignored for this strategy

        Returns:
            Tensor: Binary mask of shape [C, N]
        """
        B, C, N, D = x.shape
        device = x.device
        mask = torch.ones([C], device=device)
        mask[-3:] = 0

        mask = mask.unsqueeze(1).expand(-1, N)  # [C, N]

        return mask

    def specified_mask(x, channels):
        """Mask specified channels.

        Args:
            x (Tensor): Input tensor of shape [B, C, N, D]
            channels (list): List of channel indices to mask

        Returns:
            Tensor: Binary mask of shape [C, N] 
        """
        assert isinstance(channels, list), "channels must be a list"
        B, C, N, D = x.shape
        device = x.device
        mask = torch.zeros([C], device=device)
        mask[channels] = 1

        mask = mask.unsqueeze(1).expand(-1, N)  # [C, N]

        return mask

    strategies = {
        "random": random_mask,
        "patch": patch_mask,
        "channel": channel_mask,
        "he": he_mask,
        "mif": mif_mask,
        "specified": specified_mask,
    }

    return strategies[strategy]
