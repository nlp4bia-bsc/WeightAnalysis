"""Depth-profile correlations and quantile Wasserstein quadrature.

The historical functions retain the numerical definitions of WeightAnalysis 2.0.0.
See docs/mathematical_audit.md for assumptions and limitations.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def quantile_w1(qa: list[float], qb: list[float], *, device=None) -> float:
    """Approximate Wasserstein-1 by trapezoidal quadrature of uniform quantile curves."""
    if device is not None or isinstance(qa, torch.Tensor):
        from ..devices import resolve_device

        target = resolve_device(device) if device is not None else qa.device
        a = torch.as_tensor(qa, dtype=torch.float64, device=target)
        b = torch.as_tensor(qb, dtype=torch.float64, device=target)
        if a.shape != b.shape or a.ndim != 1 or a.numel() < 2:
            raise ValueError("quantile shape mismatch")
        return float(torch.trapezoid((a - b).abs(), dx=1 / (a.numel() - 1)))
    a = np.asarray(qa, float)
    b = np.asarray(qb, float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("quantile shape mismatch")
    return (
        float(np.trapezoid(np.abs(a - b), dx=1 / (a.size - 1)))
        if hasattr(np, "trapezoid")
        else float(np.trapz(np.abs(a - b), dx=1 / (a.size - 1)))
    )


def paper_trace_similarity(
    fingerprint_a: dict[str, list[float] | np.ndarray],
    fingerprint_b: dict[str, list[float] | np.ndarray],
    *,
    device=None,
) -> dict:
    """TRACE Eq. (5), scalar and component-level Pearson correlations.

    Undefined for shared components with fewer than 2 layers, nonfinite values,
    or constant magnitude across depth. Returned NaN is not a zero similarity.
    """
    if device is not None:
        return _device_trace(fingerprint_a, fingerprint_b, device=device)
    scores: dict[str, float] = {}
    shared = sorted(set(fingerprint_a) & set(fingerprint_b))
    for component in shared:
        a = np.asarray(fingerprint_a[component], dtype=np.float64)
        b = np.asarray(fingerprint_b[component], dtype=np.float64)
        if a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < 2:
            continue
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        length = max(len(a), len(b))
        axis = np.linspace(0.0, 1.0, length)
        aa = np.interp(axis, np.linspace(0.0, 1.0, len(a)), a)
        bb = np.interp(axis, np.linspace(0.0, 1.0, len(b)), b)
        asd, bsd = np.std(aa), np.std(bb)
        if asd < 1e-10 or bsd < 1e-10:
            continue
        az = (aa - aa.mean()) / asd
        bz = (bb - bb.mean()) / bsd
        score = np.dot(az, bz) / length
        scores[component] = float(np.clip(score, -1.0, 1.0))
    return {
        "trace_similarity": float(np.mean(list(scores.values()))) if scores else math.nan,
        "trace_n_components": len(scores),
        "trace_per_component": scores,
    }


def _device_trace(fingerprint_a, fingerprint_b, *, device):
    from ..devices import resolve_device
    from .decomposition import interpolate_uniform

    device = resolve_device(device)
    scores = {}
    for component in sorted(set(fingerprint_a) & set(fingerprint_b)):
        a = torch.as_tensor(fingerprint_a[component], dtype=torch.float64, device=device)
        b = torch.as_tensor(fingerprint_b[component], dtype=torch.float64, device=device)
        if a.ndim != 1 or b.ndim != 1 or min(a.numel(), b.numel()) < 2:
            continue
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            continue
        length = max(a.numel(), b.numel())
        a, b = interpolate_uniform(a, length), interpolate_uniform(b, length)
        sa, sb = a.std(unbiased=False), b.std(unbiased=False)
        if sa < 1e-10 or sb < 1e-10:
            continue
        scores[component] = float((((a - a.mean()) / sa * (b - b.mean()) / sb).mean()).clamp(-1, 1))
    return {
        "trace_similarity": sum(scores.values()) / len(scores) if scores else math.nan,
        "trace_n_components": len(scores),
        "trace_per_component": scores,
    }
