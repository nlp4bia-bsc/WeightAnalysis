"""Tensor-valued vector reductions with a single final host transfer per metric set."""

from __future__ import annotations

import math

import torch

from ._validation import EPS


def _ratio(numerator, denominator):
    numerator, denominator = numerator.double(), denominator.double()
    return torch.where(denominator > EPS, numerator / denominator, torch.nan)


def _finish(values):
    numbers = torch.stack(list(values.values())).detach().cpu().tolist()
    return dict(zip(values, numbers))


def drift_on_device(final, base, names):
    """Compute selected drift reductions without per-statistic device synchronization."""
    final, base = final.reshape(-1), base.reshape(-1)
    delta = final - base
    norms = {}

    def norm(name, value):
        if name not in norms:
            norms[name] = torch.linalg.vector_norm(value)
        return norms[name]

    values = {}
    for name in names:
        if name in ("relative_frobenius", "relative_l2"):
            value = _ratio(norm("delta", delta), norm("base", base))
        elif name in ("delta_l2", "delta_rms", "delta_energy"):
            value = norm("delta", delta)
            if name == "delta_rms":
                value = value / math.sqrt(delta.numel())
            elif name == "delta_energy":
                value = value.square()
        elif name in ("base_l2", "final_l2"):
            value = norm(name, base if name == "base_l2" else final)
        elif name == "cosine_final_base":
            bn, fn = norm("base", base), norm("final", final)
            value = torch.where(
                (bn > EPS) & (fn > EPS), torch.dot(final, base) / (bn * fn), torch.nan
            ).clamp(-1, 1)
        elif name == "delta_l1":
            value = delta.abs().sum()
        elif name == "delta_mae":
            value = delta.abs().mean()
        elif name == "max_abs_delta":
            value = delta.abs().max()
        else:
            value = delta.mean()
        values[name] = value
    return _finish(values)


def pairs_on_device(a, b, names, sign_floor):
    """Compute selected pair reductions and batch their scalar output transfer."""
    a, b = a.reshape(-1), b.reshape(-1)
    cache, values = {}, {}

    def norm(name, value):
        if name not in cache:
            cache[name] = torch.linalg.vector_norm(value)
        return cache[name]

    for name in names:
        if name in ("cosine", "angle_over_pi"):
            an, bn = norm("a", a), norm("b", b)
            value = torch.where(
                (an > EPS) & (bn > EPS), torch.dot(a, b) / (an * bn), torch.nan
            ).clamp(-1, 1)
            if name == "angle_over_pi":
                value = value.acos() / math.pi
        elif name == "pearson":
            ac, bc = a - a.mean(), b - b.mean()
            an, bn = torch.linalg.vector_norm(ac), torch.linalg.vector_norm(bc)
            value = torch.where(
                (an > EPS) & (bn > EPS), torch.dot(ac, bc) / (an * bn), torch.nan
            ).clamp(-1, 1)
        elif name in ("l2_distance", "rmse", "normalized_l2"):
            value = norm("delta", a - b)
            if name == "rmse":
                value = value / math.sqrt(a.numel())
            elif name == "normalized_l2":
                value = _ratio(value, torch.hypot(norm("a", a), norm("b", b)))
        elif name == "l1_mean_distance":
            value = (a - b).abs().mean()
        elif name in ("norm_a", "norm_b"):
            value = norm(name[-1], a if name == "norm_a" else b)
        else:
            mask = (a.abs() > sign_floor) & (b.abs() > sign_floor)
            same = a.sign() == b.sign()
            numerator = (mask & (same if name == "sign_agreement" else ~same)).sum()
            value = _ratio(numerator, mask.sum())
        values[name] = value
    return _finish(values)
