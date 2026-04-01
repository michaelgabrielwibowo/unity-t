"""
Device resolution utility for TRIBE v2 streaming engine.

Priority order:
  1. "dml"  — DirectML (Intel/AMD iGPU via DirectX 12, Windows)
  2. "cuda" — NVIDIA CUDA GPU
  3. "mps"  — Apple Silicon (MPS)
  4. "cpu"  — Universal fallback

Usage
-----
    from streaming.device_utils import resolve_device, to_device

    device = resolve_device("dml")
    tensor = to_device(my_tensor, device)
"""

from __future__ import annotations

import logging
from typing import Union

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """
    Resolve the best available compute device.

    Parameters
    ----------
    requested : str
        One of "auto", "cuda", "dml", "mps", "cpu".

    Returns
    -------
    torch.device
        The resolved device.  Falls back gracefully if the requested
        device is unavailable.
    """
    req = requested.lower().strip()

    if req == "dml":
        device = _try_directml()
        if device is not None:
            logger.warning("DirectML (DML) support is experimental and may crash upstream in neuralset!")
            return device
        logger.warning("DirectML unavailable — falling back to CPU")
        return torch.device("cpu")

    if req == "cuda":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
            return dev
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")

    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.warning("Apple MPS support is experimental and may crash upstream in neuralset!")
            logger.info("Using Apple MPS")
            return torch.device("mps")
        logger.warning("MPS requested but not available — falling back to CPU")
        return torch.device("cpu")

    if req == "cpu":
        logger.info("Using CPU")
        return torch.device("cpu")

    # auto — pick best available
    if req == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info("Auto-selected CUDA: %s", torch.cuda.get_device_name(0))
            return dev
        dml = _try_directml()
        if dml is not None:
            return dml
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Auto-selected Apple MPS")
            return torch.device("mps")
        logger.info("Auto-selected CPU (no GPU found)")
        return torch.device("cpu")

    logger.warning("Unknown device '%s' — using CPU", requested)
    return torch.device("cpu")


def _try_directml() -> torch.device | None:
    """
    Attempt to initialise a DirectML device via torch-directml.

    Returns the device if successful, None otherwise.
    """
    try:
        import torch_directml  # type: ignore

        # torch_directml.device() returns the first available DML device
        dml_device = torch_directml.device()

        # Smoke-test: allocate a tiny tensor to confirm it works
        _ = torch.zeros(4, device=dml_device)

        dev_name = torch_directml.device_name(0) if hasattr(torch_directml, "device_name") else "DirectML iGPU"
        logger.info("Using DirectML: %s", dev_name)
        return dml_device

    except ImportError:
        logger.debug("torch-directml not installed")
        return None
    except Exception as exc:
        logger.warning("DirectML init failed: %s", exc)
        return None


def device_info(device: torch.device) -> dict:
    """Return a human-readable info dict for the resolved device."""
    info = {"device": str(device), "type": device.type}

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory
        info["vram_gb"] = round(total / 1024 ** 3, 2)

    elif device.type == "privateuseone":  # DirectML internal type
        try:
            import torch_directml  # type: ignore
            info["name"] = torch_directml.device_name(0)
        except Exception:
            info["name"] = "DirectML iGPU"
        info["vram_gb"] = "shared"

    elif device.type == "cpu":
        import os
        info["name"] = "CPU"
        info["threads"] = os.cpu_count()

    return info


def to_device(obj, device: torch.device):
    """
    Move a tensor, module, or dict/list of tensors to *device*.
    Handles DirectML devices that don't support .to() on some dtypes.
    """
    if isinstance(obj, torch.Tensor):
        # DirectML: cast fp16 → fp32 (DirectML doesn't support fp16 ops yet)
        if device.type == "privateuseone" and obj.dtype == torch.float16:
            obj = obj.float()
        return obj.to(device)

    if isinstance(obj, torch.nn.Module):
        if device.type == "privateuseone":
            obj = obj.float()
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved)

    return obj
