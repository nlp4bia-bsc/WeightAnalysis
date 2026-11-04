"""Explicit CUDA selection without silent CPU fallback or global device state."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve gpu/CUDA aliases and fail before reading checkpoints if unavailable."""
    value = str(device).lower()
    if value == "gpu" or value.startswith("gpu:"):
        value = "cuda" + value[3:]
    try:
        resolved = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("device must be cpu, gpu, cuda, or cuda:<index>") from exc
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, gpu, cuda, or cuda:<index>")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU unavailable. Install CUDA-enabled PyTorch and use an NVIDIA GPU, or pass device='cpu'."
            )
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        return torch.device("cuda", index)
    if resolved.index is not None:
        raise ValueError("Use device='cpu' without an index")
    return resolved
