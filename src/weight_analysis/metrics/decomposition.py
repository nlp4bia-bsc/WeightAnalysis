"""Device-resident decomposition and interpolation without process-wide RNG changes."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _range_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    width = min(rank + 8, min(matrix.shape))
    generator = torch.Generator(device=matrix.device).manual_seed(0)
    omega = torch.randn(
        matrix.shape[1], width, generator=generator, device=matrix.device, dtype=matrix.dtype
    )
    q = torch.linalg.qr(matrix @ omega, mode="reduced").Q
    for _ in range(4):
        right = torch.linalg.qr(matrix.T @ q, mode="reduced").Q
        q = torch.linalg.qr(matrix @ right, mode="reduced").Q
    return q


def truncated_singular_values(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Approximate leading singular values with eight extra directions and four iterations."""
    q = _range_basis(matrix, rank)
    return torch.linalg.svdvals(q.T @ matrix)[:rank]


def truncated_left_basis(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute approximate leading left directions without a full matrix SVD."""
    q = _range_basis(matrix, rank)
    left, values, _ = torch.linalg.svd(q.T @ matrix, full_matrices=False)
    return (q @ left[:, :rank]), values[:rank]


def interpolate_uniform(values: torch.Tensor, points: int) -> torch.Tensor:
    """Linearly resample a uniform one-dimensional profile on its current device."""
    return functional.interpolate(
        values.reshape(1, 1, -1), size=points, mode="linear", align_corners=True
    ).reshape(-1)


def entry_quantiles(matrix: torch.Tensor, points: int) -> torch.Tensor:
    """Return exact linear entry quantiles, including tensors above torch.quantile's limit."""
    flat = matrix.reshape(-1)
    axis = torch.linspace(0, 1, points, device=matrix.device, dtype=matrix.dtype)
    if flat.numel() <= 2**24:
        return torch.quantile(flat, axis)
    ordered = torch.sort(flat).values
    positions = axis * (flat.numel() - 1)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    return torch.lerp(ordered[lower], ordered[upper], positions - lower)


def interpolate_positions(
    x: torch.Tensor, values: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Interpolate sorted distinct coordinates, clamping outside the observed range."""
    if x.numel() == 1:
        return values[0].expand(query.numel(), *values.shape[1:])
    query = query.clamp(x[0], x[-1])
    upper = torch.searchsorted(x, query).clamp(1, x.numel() - 1)
    lower = upper - 1
    weight = (query - x[lower]) / (x[upper] - x[lower])
    weight = weight.reshape(-1, *([1] * (values.ndim - 1)))
    return torch.lerp(values[lower], values[upper], weight)
