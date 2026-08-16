"""Backend selection and mixed-precision helpers."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported device {requested!r}; use auto, cpu, mps, or cuda")
    return device


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)
