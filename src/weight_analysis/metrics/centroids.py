"""Geometric centroids, projections, and leave-one-out descriptions of aligned updates."""

from __future__ import annotations

import torch


def update_geometry(updates: torch.Tensor, *, leave_one_out=True):
    """Describe n aligned update vectors; no fitted model or causal test is produced.

    Return tensor-valued energies and products so disjoint coordinate blocks can be
    reduced before calculating nonlinear norms, projection fractions, or cosines.
    """
    if updates.ndim != 2 or min(updates.shape) < 1 or updates.shape[0] < 2:
        raise ValueError("At least two nonempty aligned updates are required")
    updates = updates.detach().double()
    if not torch.isfinite(updates).all():
        raise ValueError("Updates contain nonfinite values")
    mean = updates.mean(dim=0)
    centered = updates - mean
    n = updates.shape[0]
    result = {
        "mean": mean,
        "update_energy": updates.square().sum(dim=1),
        "mean_energy": mean.square().sum(),
        "dot_mean": updates @ mean,
        "centered_energy": centered.square().sum(dim=1),
    }
    if leave_one_out:
        other = (updates.sum(dim=0) - updates) / (n - 1)
        result.update(
            loo_mean_energy=other.square().sum(dim=1),
            loo_dot=(updates * other).sum(dim=1),
            loo_centered_energy=(updates - other).square().sum(dim=1),
        )
    return result


def projection_summary(energy, direction_energy, dot):
    """Finalize global products after coordinate aggregation, retaining undefined states."""
    import math

    from ._validation import EPS

    if energy < 0 or direction_energy < 0:
        raise ValueError("Negative geometric energy")
    if energy <= EPS**2 or direction_energy <= EPS**2:
        return {
            "projection_fraction": math.nan,
            "projection_coefficient": math.nan,
            "direction_residual_energy": math.nan,
            "alignment_cosine": math.nan,
        }
    projected = dot * dot / direction_energy
    tolerance = 1e-10 * max(energy, projected, 1e-30)
    if projected > energy + tolerance:
        raise ValueError("Projection exceeds update energy; incompatible or corrupt statistics")
    return {
        "projection_fraction": min(1.0, projected / energy),
        "projection_coefficient": dot / direction_energy,
        "direction_residual_energy": max(0.0, energy - projected),
        "alignment_cosine": max(-1.0, min(1.0, dot / math.sqrt(energy * direction_energy))),
    }
