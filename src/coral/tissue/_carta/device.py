"""
PyTorch device selection for Rungs 2–3.

Preference order: MPS (Apple Silicon) → CUDA → CPU.
"""
from __future__ import annotations


def get_torch_device(prefer: str = "auto"):
    """
    Return a torch.device for inference / training.

    Parameters
    ----------
    prefer : "auto" | "mps" | "cuda" | "cpu"
        "auto" picks the best available backend.
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine")
        return torch.device("mps")

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine")
        return torch.device("cuda")

    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device=None) -> str:
    """Human-readable device summary for logging."""
    import torch

    if device is None:
        device = get_torch_device()
    name = str(device)
    if name == "mps":
        return "mps (Apple Metal)"
    if name.startswith("cuda"):
        return f"{name} ({torch.cuda.get_device_name(device)})"
    return "cpu"
