# -*- coding: utf-8 -*-
"""
@Time   :  2025/11/14 01:16
@Author :  Yufan Liu
@Desc   :  Helper functions for loading Eva model and extracting embeddings
"""

from huggingface_hub import hf_hub_download

from coral.features._eva.eva import EvaMAE


def load_from_hf(repo_id, conf, device="cpu", checkpoint_filename="Eva_model.ckpt", cache_dir=None, force_download=False):
    """Load Eva model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/eva-base")
        conf: Configuration object
        device: Device to load model on (default: "cpu")
        checkpoint_filename: Name of the checkpoint file in the repo (default: "Eva_model.ckpt")
        cache_dir: Directory to cache the downloaded file (default: ~/.cache/huggingface)
        force_download: Whether to force re-download even if cached
        
    Returns:
        model: Loaded model
    """
    try:
        print(f"Downloading model from HuggingFace: {repo_id}")
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_filename,
            cache_dir=cache_dir,
            force_download=force_download,
        )
        print(f"Checkpoint downloaded to: {checkpoint_path}")
        
        model = EvaMAE.from_checkpoint(checkpoint_path, conf, device=device)
        print(f"Model loaded successfully from HuggingFace Hub")
        return model
        
    except Exception as e:
        print(f"Failed to load Eva model from HuggingFace: {e}")
        raise


def load_from_checkpoint(checkpoint_path, conf, device="cpu"):
    """Load Eva model from local checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        conf: Configuration object
        device: Device to load model on (default: "cpu")

    Returns:
        model: Loaded model
    """
    try:
        model = EvaMAE.from_checkpoint(checkpoint_path, conf, device=device)
        return model
    except Exception as e:
        print(f"Failed to load Eva model: {e}")
        raise


def extract_features(patch, bms, model, device, cls=False, channel_mode="full"):
    """Extract features (embeddings) from Eva model.

    Args:
        patch: Input patch tensor of shape [B, H, W, C]
        bms: Biomarkers information
        model: Eva model
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

    image_out, raw_mask = model.model.forward_encoder(x, bms)
    if cls:
        image_cls = image_out[:, :, 0, :]
    else:
        image_cls = image_out[:, :, 1:, :].mean(2)
    image_cls = image_cls.squeeze(1)
    batch_size = image_cls.size(0)
    feat = image_cls.view(batch_size, -1)
    return feat


def create_model(conf, checkpoint_path=None, repo_id=None, device="cpu", checkpoint_filename="Eva_model.ckpt"):
    """Create Eva model from checkpoint or HuggingFace Hub.

    Args:
        conf: Configuration object
        checkpoint_path: Path to local checkpoint file (optional if repo_id is provided)
        repo_id: HuggingFace repository ID (optional if checkpoint_path is provided)
        device: Device to load model on (default: "cpu")
        checkpoint_filename: Name of checkpoint file in HuggingFace repo (default: "Eva_model.ckpt")

    Returns:
        model: Loaded Eva model
    """
    if repo_id is not None:
        return load_from_hf(repo_id, conf, device=device, checkpoint_filename=checkpoint_filename)
    elif checkpoint_path is not None:
        return load_from_checkpoint(checkpoint_path, conf, device=device)
    else:
        raise ValueError("Either checkpoint_path or repo_id must be provided")

