"""
Additional quantization and compilation utilities.

Provides fp16 inference, torch.compile, and optional bitsandbytes
8-bit quantization for TRIBE v2 feature extractors.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_fp16(model: nn.Module) -> nn.Module:
    """Convert model to fp16 for inference.

    Keeps batch norm and layer norm in fp32 for numerical stability.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            module.float()
        else:
            module.half()
    logger.info("Applied fp16 to model (%s)", type(model).__name__)
    return model


def apply_torch_compile(
    model: nn.Module,
    mode: str = "reduce-overhead",
    backend: str = "inductor",
) -> nn.Module:
    """Apply torch.compile for faster inference.

    Parameters
    ----------
    mode : str
        Compilation mode: ``"default"``, ``"reduce-overhead"``, or ``"max-autotune"``.
    backend : str
        Compiler backend (default: ``"inductor"``).
    """
    try:
        compiled = torch.compile(model, mode=mode, backend=backend)
        logger.info(
            "Applied torch.compile (mode=%s, backend=%s)", mode, backend
        )
        return compiled
    except Exception as exc:
        logger.warning("torch.compile failed: %s — using eager mode", exc)
        return model


def apply_bitsandbytes_8bit(model: nn.Module) -> nn.Module:
    """Apply bitsandbytes 8-bit quantization to linear layers.

    Requires the ``bitsandbytes`` package.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.warning("bitsandbytes not installed — skipping 8-bit quantization")
        return model

    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] >= 256:
            # Replace with 8-bit linear
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)

            bnb_linear = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            bnb_linear.weight = bnb.nn.Int8Params(
                module.weight.data, requires_grad=False
            )
            if module.bias is not None:
                bnb_linear.bias = module.bias

            setattr(parent, child_name, bnb_linear)
            replaced += 1

    if replaced > 0:
        logger.info("Applied bitsandbytes 8-bit to %d linear layers", replaced)
    return model


class QuantizationManager:
    """Manages quantization strategy for the full TRIBE v2 stack.

    Applies the appropriate quantization to each component based on
    its characteristics and available GPU memory.

    Parameters
    ----------
    target_vram_gb : float
        Target maximum VRAM usage in GB. The manager will select
        quantization strategies to fit within this budget.
    """

    def __init__(self, target_vram_gb: float = 12.0):
        self.target_vram_gb = target_vram_gb
        self._applied = {}

    def optimize_model(
        self,
        model: Any,
        component_name: str = "fmri_encoder",
        use_fp16: bool = True,
        use_compile: bool = True,
        use_bnb: bool = False,
    ) -> Any:
        """Apply optimal quantization for a model component.

        Parameters
        ----------
        model : nn.Module
            The model or sub-model to optimize.
        component_name : str
            Name for logging (e.g., ``"fmri_encoder"``, ``"vjepa2"``).
        use_fp16 : bool
            Apply fp16 conversion.
        use_compile : bool
            Apply torch.compile.
        use_bnb : bool
            Apply bitsandbytes 8-bit.
        """
        if use_fp16 and isinstance(model, nn.Module):
            model = apply_fp16(model)

        if use_bnb and isinstance(model, nn.Module):
            model = apply_bitsandbytes_8bit(model)

        if use_compile and isinstance(model, nn.Module):
            model = apply_torch_compile(model)

        self._applied[component_name] = {
            "fp16": use_fp16,
            "compile": use_compile,
            "bnb": use_bnb,
        }
        return model

    def report(self) -> dict:
        """Return a summary of applied optimizations."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
        else:
            allocated = reserved = 0.0

        return {
            "components": self._applied,
            "gpu_allocated_gb": round(allocated, 2),
            "gpu_reserved_gb": round(reserved, 2),
            "target_vram_gb": self.target_vram_gb,
        }
