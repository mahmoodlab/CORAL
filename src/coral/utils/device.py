"""Device resolution for the ``--device`` / ``--gpu`` flags.

Turns a device spec (``None``/``"auto"``, ``"cpu"``, ``"cuda"``,
``"cuda:N"``, or ``"mps"``) into a concrete torch device string, validating
availability at this boundary so callers never see a raw torch error.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from coral.utils.errors import CoralError

logger = logging.getLogger(__name__)


def resolve_device(device: str | None = None) -> str:
    """Resolve a device spec to a concrete torch device string.

    Args:
        device: The requested device. ``None`` or ``"auto"`` picks the best
            available (CUDA, then Apple MPS, then CPU); ``"cpu"`` forces the
            CPU; ``"cuda"`` the default CUDA device; ``"cuda:N"`` a specific
            GPU by index (e.g. ``"cuda:1"``); ``"mps"`` the Apple-Silicon
            GPU. Case-insensitive.

    Returns:
        A torch device string: ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or
        ``"mps"``.

    Raises:
        CoralError: If the requested device is unavailable (no CUDA/MPS, or an
            index out of range), the spec is unrecognised, or torch is not
            installed for a request that needs it.

    Example:
        On a Mac with no ``--device`` flag this returns ``"mps"`` when Apple
        MPS is available (else ``"cpu"``); on an NVIDIA box it returns
        ``"cuda"`` (else ``"cpu"``). ``resolve_device("cuda:1")`` targets the
        second GPU, raising if fewer are present.

        >>> resolve_device("cpu")
        'cpu'
    """
    spec = (device or "").strip().lower() or None
    if spec == "auto":
        spec = None

    # PyTorch reads PYTORCH_ENABLE_MPS_FALLBACK once, at import time, so the
    # CPU fallback for MPS-unimplemented ops must be armed *before* the
    # ``import torch`` below — setting it afterwards is silently ignored. MPS
    # exists only on Apple Silicon, which we can detect without torch; the
    # variable is inert on machines with no MPS backend.
    if spec in (None, "mps") and sys.platform == "darwin":
        _enable_mps_fallback()

    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        # torch is an optional extra; CPU / auto must still work without it.
        if spec is None or spec == "cpu":
            return "cpu"
        msg = (
            f"--device {spec!r} requested but torch is not installed; "
            "install a torch-backed extra (e.g. `uv sync --extra kronos2`)."
        )
        raise CoralError(msg) from exc

    if spec is None:
        return _auto_device(torch)
    if spec == "cpu":
        return "cpu"
    if spec == "mps":
        return _resolve_mps(torch)
    if spec == "cuda" or spec.startswith("cuda:"):
        return _resolve_cuda(torch, spec)
    msg = (
        f"unknown device {device!r}; valid: auto, cpu, cuda, "
        "cuda:N (a GPU index, e.g. cuda:0), mps."
    )
    raise CoralError(msg)


def _auto_device(torch: Any) -> str:  # noqa: ANN401 — the torch module
    """Best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available(torch):
        _log_mps_selected()
        return "mps"
    return "cpu"


def _resolve_cuda(torch: Any, spec: str) -> str:  # noqa: ANN401 — torch
    """Validate a ``cuda`` / ``cuda:N`` spec against the visible GPUs."""
    if not torch.cuda.is_available():
        msg = (
            f"--device {spec!r} requested but no CUDA device is available; "
            "use --device cpu (or mps on Apple Silicon)."
        )
        raise CoralError(msg)
    if spec == "cuda":
        return "cuda"
    index_str = spec.split(":", 1)[1]
    try:
        index = int(index_str)
    except ValueError as exc:
        msg = f"invalid CUDA index in --device {spec!r}; expected cuda:N."
        raise CoralError(msg) from exc
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        msg = (
            f"--device {spec!r} is out of range; {count} CUDA device(s) "
            f"available (valid indices 0..{count - 1})."
        )
        raise CoralError(msg)
    return f"cuda:{index}"


def _resolve_mps(torch: Any) -> str:  # noqa: ANN401 — the torch module
    """Validate an ``mps`` request and arm the CPU fallback."""
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        _log_mps_selected()
        return "mps"
    built = mps is not None and getattr(mps, "is_built", lambda: False)()
    detail = (
        "the MPS backend is present but no MPS device is available here"
        if built
        else "this torch build has no MPS backend"
    )
    msg = (
        f"--device 'mps' requested but unavailable — {detail}. MPS needs "
        "Apple Silicon on macOS. Use --device cpu."
    )
    raise CoralError(msg)


def _mps_available(torch: Any) -> bool:  # noqa: ANN401 — the torch module
    """True if this torch build reports a usable Apple MPS device."""
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps is not None and mps.is_available())


def _enable_mps_fallback() -> None:
    """Arm the MPS→CPU fallback for ops with no MPS kernel.

    Sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` only when unset, so an explicit
    ``0`` (opt-out) or ``1`` is preserved. PyTorch reads this variable when
    it is imported, so this must run *before* ``import torch`` to take
    effect; ``resolve_device`` calls it there.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _log_mps_selected() -> None:
    """Log that MPS was chosen and report the CPU-fallback setting."""
    logger.info(
        "MPS selected (PYTORCH_ENABLE_MPS_FALLBACK=%s; when 1, ops with no "
        "MPS kernel fall back to CPU).",
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
    )
