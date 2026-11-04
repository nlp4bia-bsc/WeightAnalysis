"""Vector geometry and pretraining drift.

The historical functions retain the numerical definitions of WeightAnalysis 2.0.0.
See docs/mathematical_audit.md for assumptions and limitations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ._validation import EPS


def drift(final: torch.Tensor, base: torch.Tensor) -> dict[str, float | int]:
    """Return historical pretraining drift statistics without strict new-API validation."""
    if final.shape != base.shape or not final.is_floating_point():
        raise ValueError("incompatible drift")
    a = final.reshape(-1).to(torch.float64)
    b = base.reshape(-1).to(torch.float64)
    d = a - b
    n = a.numel()
    if n == 0:
        raise ValueError("empty parameter")
    dn = torch.linalg.vector_norm(d).item()
    bn = torch.linalg.vector_norm(b).item()
    fn = torch.linalg.vector_norm(a).item()
    return dict(
        n_parameters=n,
        delta_l2=dn,
        base_l2=bn,
        final_l2=fn,
        relative_l2=(dn / bn if bn > EPS else math.nan),
        delta_rms=dn / math.sqrt(n),
        delta_mae=float(d.abs().mean()),
        max_abs_delta=float(d.abs().max()),
        delta_mean=float(d.mean()),
        cosine_final_base=float(torch.dot(a, b) / (fn * bn)) if fn > EPS and bn > EPS else math.nan,
        delta_l1=float(d.abs().sum()),
        delta_energy=dn * dn,
    )


@dataclass
class VectorAccumulator:
    """Accumulate historical raw-moment statistics over aligned tensor blocks."""

    n: int = 0
    xx: float = 0.0
    yy: float = 0.0
    xy: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    d2: float = 0.0
    l1: float = 0.0
    agree: int = 0
    both: int = 0
    conflict: int = 0

    def update(self, x: torch.Tensor, y: torch.Tensor, threshold: float = 1e-8) -> None:
        """Add one block with the same coordinates and shape in both vectors."""
        if x.shape != y.shape:
            raise ValueError("different vector shape")
        a = x.reshape(-1).to(torch.float64)
        b = y.reshape(-1).to(torch.float64)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite vector")
        self.n += a.numel()
        self.xx += float(torch.dot(a, a))
        self.yy += float(torch.dot(b, b))
        self.xy += float(torch.dot(a, b))
        self.sx += float(a.sum())
        self.sy += float(b.sum())
        d = a - b
        self.d2 += float(torch.dot(d, d))
        self.l1 += float(d.abs().sum())
        mask = (a.abs() > threshold) & (b.abs() > threshold)
        self.both += int(mask.sum())
        self.agree += int(((a * b > 0) & mask).sum())
        self.conflict += int(((a * b < 0) & mask).sum())

    def result(self) -> dict[str, float | int]:
        """Return historical vector distances, correlations and sign proportions."""
        if self.n == 0:
            raise ValueError("empty accumulator")
        xn = math.sqrt(self.xx)
        yn = math.sqrt(self.yy)
        cosine = float(np.clip(self.xy / (xn * yn), -1, 1)) if xn > EPS and yn > EPS else math.nan
        cx = self.xx - self.sx**2 / self.n
        cy = self.yy - self.sy**2 / self.n
        pearson = (
            float(np.clip((self.xy - self.sx * self.sy / self.n) / math.sqrt(cx * cy), -1, 1))
            if cx > EPS and cy > EPS
            else math.nan
        )
        return dict(
            n_parameters=self.n,
            cosine=cosine,
            angle_over_pi=math.acos(cosine) / math.pi if math.isfinite(cosine) else math.nan,
            pearson=pearson,
            l2_distance=math.sqrt(self.d2),
            normalized_l2=math.sqrt(self.d2 / (self.xx + self.yy))
            if self.xx + self.yy > EPS
            else math.nan,
            l1_mean_distance=self.l1 / self.n,
            rmse=math.sqrt(self.d2 / self.n),
            norm_a=xn,
            norm_b=yn,
            sign_agreement=self.agree / self.both if self.both else math.nan,
            sign_conflict=self.conflict / self.both if self.both else math.nan,
            sign_compared=self.both,
        )


DRIFT_METRICS = (
    "relative_frobenius",
    "relative_l2",
    "delta_l2",
    "delta_rms",
    "delta_mae",
    "delta_l1",
    "delta_energy",
    "base_l2",
    "final_l2",
    "cosine_final_base",
    "max_abs_delta",
    "delta_mean",
)
PAIR_METRICS = (
    "cosine",
    "angle_over_pi",
    "pearson",
    "l2_distance",
    "normalized_l2",
    "l1_mean_distance",
    "rmse",
    "norm_a",
    "norm_b",
    "sign_agreement",
    "sign_conflict",
)


def drift_metrics(
    final: torch.Tensor,
    base: torch.Tensor,
    *,
    metrics: tuple[str, ...] | list[str] = ("relative_frobenius",),
) -> dict[str, float]:
    """Compute only requested drift statistics on a shared parameter tensor.

    Relative Frobenius distance is ||final-base||₂ / ||base||₂. Vectorization
    makes the definition valid for parameters of any shape. A denominator at
    most 1e-14 produces NaN, matching the historical null convention.
    """
    from ._validation import metric_names, tensor_pair

    names = metric_names(metrics, DRIFT_METRICS)
    final, base = tensor_pair(final, base)
    final, base = final.reshape(-1), base.reshape(-1)
    delta = final - base
    norms: dict[str, float] = {}

    def norm(name: str, tensor: torch.Tensor) -> float:
        if name not in norms:
            norms[name] = float(torch.linalg.vector_norm(tensor))

        return norms[name]

    result = {}
    for name in names:
        if name in ("relative_frobenius", "relative_l2"):
            denominator = norm("base", base)
            value = norm("delta", delta) / denominator if denominator > EPS else math.nan
        elif name == "delta_l2":
            value = norm("delta", delta)
        elif name == "delta_rms":
            value = norm("delta", delta) / math.sqrt(delta.numel())
        elif name == "delta_energy":
            value = norm("delta", delta) ** 2
        elif name in ("base_l2", "final_l2"):
            value = norm(
                "base" if name == "base_l2" else "final", base if name == "base_l2" else final
            )
        elif name == "cosine_final_base":
            base_norm, final_norm = norm("base", base), norm("final", final)
            value = (
                float(torch.dot(final, base)) / (base_norm * final_norm)
                if base_norm > EPS and final_norm > EPS
                else math.nan
            )
            value = float(np.clip(value, -1, 1))
        elif name == "delta_l1":
            value = float(delta.abs().sum())
        elif name == "delta_mae":
            value = float(delta.abs().mean())
        elif name == "max_abs_delta":
            value = float(delta.abs().max())
        else:
            value = float(delta.mean())
        result[name] = value

    return result


def relative_frobenius(final: torch.Tensor, base: torch.Tensor) -> float:
    """Return ||final-base||_F / ||base||_F without any spectral computation."""
    return drift_metrics(final, base)["relative_frobenius"]


def pair_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    metrics: tuple[str, ...] | list[str] = ("cosine",),
    sign_floor: float = 1e-8,
) -> dict[str, float]:
    """Compare aligned vectors with only the requested distance or similarity.

    Pearson uses explicitly centered vectors to avoid cancellation in raw
    second moments. This is a strict alternative to VectorAccumulator's
    historical streaming-moment implementation.
    """
    from ._validation import metric_names, tensor_pair

    names = metric_names(metrics, PAIR_METRICS)
    if not math.isfinite(sign_floor) or sign_floor < 0:
        raise ValueError("sign_floor must be finite and nonnegative")
    a, b = tensor_pair(a, b)
    a, b = a.reshape(-1), b.reshape(-1)
    result = {}
    cache: dict[str, float] = {}
    for name in names:
        if name in ("cosine", "angle_over_pi", "normalized_l2", "norm_a", "norm_b"):
            if "norm_a" not in cache:
                cache["norm_a"] = float(torch.linalg.vector_norm(a))
                cache["norm_b"] = float(torch.linalg.vector_norm(b))
        if name in ("cosine", "angle_over_pi"):
            denominator = cache["norm_a"] * cache["norm_b"]
            value = (
                float(torch.dot(a, b)) / denominator
                if min(cache["norm_a"], cache["norm_b"]) > EPS
                else math.nan
            )
            value = float(np.clip(value, -1, 1))
            if name == "angle_over_pi":
                value = math.acos(value) / math.pi
        elif name == "pearson":
            centered_a, centered_b = a - a.mean(), b - b.mean()
            centered_norm_a = float(torch.linalg.vector_norm(centered_a))
            centered_norm_b = float(torch.linalg.vector_norm(centered_b))
            denominator = centered_norm_a * centered_norm_b
            value = (
                float(torch.dot(centered_a, centered_b)) / denominator
                if min(centered_norm_a, centered_norm_b) > EPS
                else math.nan
            )
            value = float(np.clip(value, -1, 1))
        elif name in ("l2_distance", "rmse", "normalized_l2"):
            if "l2" not in cache:
                cache["l2"] = float(torch.linalg.vector_norm(a - b))
            value = cache["l2"]
            if name == "rmse":
                value /= math.sqrt(a.numel())
            if name == "normalized_l2":
                denominator = math.hypot(cache["norm_a"], cache["norm_b"])
                value = value / denominator if denominator > EPS else math.nan
        elif name == "l1_mean_distance":
            value = float((a - b).abs().mean())
        elif name in ("norm_a", "norm_b"):
            value = cache[name]
        else:
            mask = (a.abs() > sign_floor) & (b.abs() > sign_floor)
            denominator = int(mask.sum())
            signs_equal = torch.sign(a[mask]) == torch.sign(b[mask])
            numerator = (
                int(signs_equal.sum()) if name == "sign_agreement" else int((~signs_equal).sum())
            )
            value = numerator / denominator if denominator else math.nan
        result[name] = value

    return result
