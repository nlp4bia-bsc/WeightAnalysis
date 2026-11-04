"""Typed, single-concern numerical settings for checkpoint computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MetricOptions:
    """Control numerical cost and alignment without carrying experiment paths."""

    device: str = "gpu"
    spectral_mode: Literal["auto", "exact", "topk", "skip"] = "auto"
    max_svd_elements: int = 3_000_000
    topk: int = 32
    subspace_rank: int = 8
    topk_fraction: float = 0.01
    sign_floor: float = 1e-8
    min_aligned_fraction: float = 0.99
    min_fingerprint_components: int = 3
    factorial_chunk_elements: int = 30_000

    def __post_init__(self) -> None:
        import re

        if not isinstance(self.device, str) or not re.fullmatch(
            r"cpu|(?:gpu|cuda)(?::\d+)?", self.device.lower()
        ):
            raise ValueError("device must be cpu, gpu, cuda, or cuda:<index>")
        if self.spectral_mode not in ("auto", "exact", "topk", "skip"):
            raise ValueError("Unknown spectral_mode")
        for name in (
            "max_svd_elements",
            "topk",
            "subspace_rank",
            "min_fingerprint_components",
            "factorial_chunk_elements",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.min_aligned_fraction <= 1:
            raise ValueError("min_aligned_fraction must lie in (0, 1]")
        if not 0 < self.topk_fraction <= 1:
            raise ValueError("topk_fraction must lie in (0, 1]")
        if not 0 <= self.sign_floor < float("inf"):
            raise ValueError("sign_floor must be finite and nonnegative")
