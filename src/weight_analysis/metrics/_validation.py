"""Shared numerical input contracts for the selective metric API."""

from __future__ import annotations

from collections.abc import Iterable

import torch

EPS = 1e-14


def floating_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point torch.Tensor")
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain nonempty finite values")

    return value.detach().to(device="cpu", dtype=torch.float64)


def tensor_pair(final: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    final = floating_tensor(final, name="final")
    base = floating_tensor(base, name="base")
    if final.shape != base.shape:
        raise ValueError(f"final and base shapes differ: {final.shape} versus {base.shape}")

    return final, base


def metric_names(metrics: Iterable[str], supported: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(metrics, str):
        raise ValueError("metrics must be a sequence of metric names, not a string")
    names = tuple(dict.fromkeys(metrics))
    if not names:
        raise ValueError("metrics must contain at least one metric name")
    unknown = sorted(set(names) - set(supported))
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; supported metrics are {supported}")

    return names


def orthonormal_columns(value: torch.Tensor, *, name: str) -> torch.Tensor:
    matrix = floating_tensor(value, name=name)
    if matrix.ndim != 2 or matrix.shape[1] > matrix.shape[0]:
        raise ValueError(f"{name} must have shape (ambient_dimension, rank) with rank <= dimension")
    identity = torch.eye(matrix.shape[1], dtype=matrix.dtype)
    if not torch.allclose(matrix.T @ matrix, identity, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{name} columns must be orthonormal; compute a principal_basis first")

    return matrix
