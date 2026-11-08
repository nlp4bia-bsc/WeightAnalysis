"""Singular-value diagnostics and normalized spectral fingerprints.

The historical functions retain the numerical definitions of WeightAnalysis 2.0.0.
See docs/mathematical_audit.md for assumptions and limitations.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.sparse.linalg import svds

from ._validation import EPS


def spectral(
    matrix: torch.Tensor,
    mode: str = "auto",
    max_exact_elements: int = 3_000_000,
    topk: int = 32,
    device: str = "cpu",
    curve_points: int = 32,
    quantile_points: int = 21,
) -> dict:
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return dict(status="not_matrix")
    if str(device).lower().startswith(("gpu", "cuda")):
        values = spectral_metrics(
            matrix,
            mode=mode,
            metrics=SPECTRAL_METRICS,
            max_exact_elements=max_exact_elements,
            topk=topk,
            device=device,
            curve_points=curve_points,
            quantile_points=quantile_points,
        )
        rank = min(matrix.shape)
        count = (
            rank
            if values["status"] == "exact"
            else min(topk, rank - 1)
            if values["status"] == "topk_approx"
            else 0
        )
        return dict(
            matrix_rows=matrix.shape[0],
            matrix_cols=matrix.shape[1],
            max_possible_rank=rank,
            n_singular_computed=count,
            **values,
        )
    if not torch.isfinite(matrix).all():
        raise ValueError("nonfinite spectral matrix")
    r, c = matrix.shape
    rank = min(r, c)
    if mode == "skip":
        return dict(status="disabled")
    full = mode == "exact" or mode == "auto" and matrix.numel() <= max_exact_elements or rank <= 2
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    norm = float(torch.linalg.vector_norm(matrix.to(torch.float64)))
    if norm <= EPS:
        return dict(
            status="zero_matrix",
            matrix_rows=r,
            matrix_cols=c,
            matrix_fro=0.0,
            stable_rank=math.nan,
            effective_rank=math.nan,
            energy_effective_rank=math.nan,
            spectral_energy_curve=None,
        )
    with torch.no_grad():
        if full:
            inp = matrix.to(device=device, dtype=torch.float64)
            s = torch.linalg.svdvals(inp).detach().cpu().numpy()
            del inp
            status = "exact"
        else:
            k = min(max(1, topk), rank - 1)
            if k < 1:
                raise ValueError("invalid truncated rank")
            # scipy svds returns unordered singular values; explicit RNG fixed via v0.
            inp = matrix.detach().cpu().numpy()
            try:
                s = svds(inp, k=k, return_singular_vectors=False, v0=np.ones(min(inp.shape)))
            except Exception as ex:
                raise RuntimeError(f"Truncated SVD failed for {r}x{c}: {ex}") from ex
            status = "topk_approx"
    s = np.sort(np.maximum(np.asarray(s, dtype=np.float64), 0))[::-1]
    energy = s * s
    capture = float(energy.sum() / (norm * norm))
    capture = float(np.clip(capture, 0, 1))
    sigma_p = s / s.sum() if full else None
    eng_p = energy / energy.sum() if full else None
    erank = (
        float(np.exp(-np.sum(sigma_p[sigma_p > 0] * np.log(sigma_p[sigma_p > 0]))))
        if full
        else math.nan
    )
    e_erank = (
        float(np.exp(-np.sum(eng_p[eng_p > 0] * np.log(eng_p[eng_p > 0])))) if full else math.nan
    )
    cdf = np.cumsum(eng_p) if full else None
    if full:
        x = np.linspace(0, 1, len(s) + 1)
        curve = np.interp(np.linspace(0, 1, curve_points), x, np.r_[0, cdf]).tolist()
    else:
        curve = None
    # Normalized weight distributions: classical quantiles, *not* exact TRACE.
    # May be expensive for large matrices but does not retain tensor copies.
    flat = matrix.detach().cpu().numpy().ravel()
    rms = norm / math.sqrt(flat.size)
    wq = np.quantile(flat / rms, np.linspace(0, 1, quantile_points)).tolist()
    return dict(
        status=status,
        matrix_rows=r,
        matrix_cols=c,
        max_possible_rank=rank,
        n_singular_computed=len(s),
        matrix_fro=norm,
        spectral_norm=float(s[0]),
        nuclear_norm=float(s.sum()) if full else math.nan,
        stable_rank=norm * norm / (s[0] * s[0]) if s[0] > EPS else math.nan,
        top1_energy=float(energy[:1].sum() / norm**2),
        top5_energy=float(energy[:5].sum() / norm**2),
        top10_energy=float(energy[:10].sum() / norm**2),
        topk_energy_capture=capture,
        effective_rank=erank,
        effective_rank_fraction=erank / rank if full else math.nan,
        energy_effective_rank=e_erank,
        spectral_entropy_sigma=math.log(erank) if full else math.nan,
        spectral_entropy_energy=math.log(e_erank) if full else math.nan,
        rank90_fraction=(float((np.searchsorted(cdf, 0.90) + 1) / rank) if full else math.nan),
        rank95_fraction=(float((np.searchsorted(cdf, 0.95) + 1) / rank) if full else math.nan),
        rank99_fraction=(float((np.searchsorted(cdf, 0.99) + 1) / rank) if full else math.nan),
        spectral_energy_curve=curve,
        normalized_weight_quantiles=wq,
    )


SPECTRAL_METRICS = (
    "matrix_fro",
    "spectral_norm",
    "nuclear_norm",
    "stable_rank",
    "effective_rank",
    "effective_rank_fraction",
    "energy_effective_rank",
    "spectral_entropy_sigma",
    "spectral_entropy_energy",
    "top1_energy",
    "top5_energy",
    "top10_energy",
    "topk_energy_capture",
    "rank90_fraction",
    "rank95_fraction",
    "rank99_fraction",
    "spectral_energy_curve",
    "normalized_weight_quantiles",
)


def spectral_metrics(
    matrix: torch.Tensor,
    *,
    metrics: tuple[str, ...] | list[str] = ("effective_rank",),
    mode: str = "exact",
    max_exact_elements: int = 3_000_000,
    topk: int = 32,
    device: str | None = None,
    curve_points: int = 32,
    quantile_points: int = 21,
    return_spectrum: bool = False,
) -> dict:
    """Compute selected spectral statistics with at most one decomposition.

    Frobenius norm and normalized entry quantiles never invoke SVD. Exact-only
    statistics are NaN (or None for curves) for a truncated spectrum. The
    returned status records whether singular values were exact, approximate,
    unnecessary, or undefined for an all-zero matrix.
    """
    from ._validation import floating_tensor, metric_names

    names = metric_names(metrics, SPECTRAL_METRICS)
    if mode not in ("exact", "auto", "topk", "skip"):
        raise ValueError("mode must be exact, auto, topk or skip")
    if max_exact_elements < 1 or topk < 1 or curve_points < 2 or quantile_points < 2:
        raise ValueError(
            "SVD thresholds and topk must be positive; profile grids need at least 2 points"
        )
    matrix = floating_tensor(matrix, name="matrix", device=device)
    device = matrix.device
    from .decomposition import entry_quantiles, interpolate_uniform, truncated_singular_values

    if matrix.ndim != 2:
        raise ValueError("matrix must have exactly two dimensions")
    norm = float(torch.linalg.vector_norm(matrix))
    result = {"status": "not_required"}
    cheap = {"matrix_fro", "normalized_weight_quantiles"}
    needs_svd = bool(set(names) - cheap) or return_spectrum
    if return_spectrum:
        result["singular_values"] = None
        result["spectrum_complete"] = False
    if needs_svd and mode == "skip":
        result["status"] = "disabled"
    if norm <= EPS:
        result["status"] = "zero_matrix"
        if return_spectrum and norm == 0:
            result["singular_values"] = [0.0] * min(matrix.shape)
            result["spectrum_complete"] = True
        for name in names:
            result[name] = (
                0.0
                if name == "matrix_fro"
                else (
                    None
                    if name in ("spectral_energy_curve", "normalized_weight_quantiles")
                    else math.nan
                )
            )

        return result

    if "matrix_fro" in names:
        result["matrix_fro"] = norm
    if "normalized_weight_quantiles" in names:
        result["normalized_weight_quantiles"] = entry_quantiles(
            matrix / (norm / math.sqrt(matrix.numel())), quantile_points
        ).tolist()
    if not needs_svd:
        return result
    if mode == "skip":
        for name in set(names) - cheap:
            result[name] = None if name == "spectral_energy_curve" else math.nan

        return result

    rank = min(matrix.shape)
    full = mode == "exact" or (mode == "auto" and matrix.numel() <= max_exact_elements) or rank <= 2
    if full:
        singular = torch.linalg.svdvals(matrix)
        result["status"] = "exact"
    else:
        count = min(topk, rank - 1)
        if matrix.is_cuda:
            singular = truncated_singular_values(matrix, count)
        else:
            singular = torch.from_numpy(
                svds(
                    matrix.numpy(),
                    k=count,
                    return_singular_vectors=False,
                    v0=np.ones(rank),
                    which="LM",
                ).copy()
            )
        result["status"] = "topk_approx"
    singular = torch.sort(singular.clamp_min(0), descending=True).values
    if return_spectrum:
        result["singular_values"] = singular.tolist()
        result["spectrum_complete"] = bool(full)
    energy = singular.square()
    total_energy = norm**2
    sigma_probability = singular / singular.sum()
    energy_probability = energy / energy.sum()
    sigma_positive = sigma_probability[sigma_probability > 0]
    energy_positive = energy_probability[energy_probability > 0]
    sigma_entropy = float(-(sigma_positive * sigma_positive.log()).sum())
    energy_entropy = float(-(energy_positive * energy_positive.log()).sum())
    cumulative = torch.cumsum(energy_probability, dim=0)
    for name in set(names) - cheap:
        if name == "spectral_norm":
            value = float(singular[0])
        elif name == "stable_rank":
            value = total_energy / float(singular[0] ** 2)
        elif name == "topk_energy_capture":
            value = float((energy.sum() / total_energy).clamp(0, 1))
        elif name in ("top1_energy", "top5_energy", "top10_energy"):
            count = int(name.removeprefix("top").removesuffix("_energy"))
            value = float(energy[:count].sum() / total_energy)
        elif not full:
            value = None if name == "spectral_energy_curve" else math.nan
        elif name == "nuclear_norm":
            value = float(singular.sum())
        elif name in ("effective_rank", "effective_rank_fraction"):
            value = math.exp(sigma_entropy)
            if name == "effective_rank_fraction":
                value /= rank
        elif name == "energy_effective_rank":
            value = math.exp(energy_entropy)
        elif name == "spectral_entropy_sigma":
            value = sigma_entropy
        elif name == "spectral_entropy_energy":
            value = energy_entropy
        elif name == "spectral_energy_curve":
            value = interpolate_uniform(
                torch.cat((cumulative.new_zeros(1), cumulative)), curve_points
            ).tolist()
        else:
            cutoff = int(name.removeprefix("rank").removesuffix("_fraction")) / 100
            value = float(
                (torch.searchsorted(cumulative, cumulative.new_tensor(cutoff)) + 1) / rank
            )
        result[name] = value

    return result
