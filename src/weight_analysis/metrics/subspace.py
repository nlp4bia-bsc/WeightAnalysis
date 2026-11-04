"""Principal subspaces and principal-angle similarity.

The historical functions retain the numerical definitions of WeightAnalysis 2.0.0.
See docs/mathematical_audit.md for assumptions and limitations.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.sparse.linalg import svds

from ._validation import EPS


def basis(
    matrix: torch.Tensor, rank: int, device: str = "cpu", max_exact_elements: int = 3_000_000
) -> tuple[torch.Tensor | None, str]:
    """Return the historical requested-rank singular basis and approximation status."""
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return None, "not_matrix"
    if str(device).lower().startswith(("gpu", "cuda")):
        from ..devices import resolve_device
        from ._validation import floating_tensor
        from .decomposition import truncated_left_basis

        matrix = floating_tensor(matrix, name="matrix", device=resolve_device(device))
        if float(torch.linalg.vector_norm(matrix)) <= EPS:
            return None, "zero_matrix"
        k = min(rank, min(matrix.shape))
        if k < 1:
            return None, "disabled"
        if matrix.numel() <= max_exact_elements or k == min(matrix.shape):
            return torch.linalg.svd(matrix, full_matrices=False).U[:, :k], "exact_svd"
        left, _ = truncated_left_basis(matrix, k)
        return left, "topk_approx"
    if not torch.isfinite(matrix).all():
        raise ValueError("nonfinite subspace matrix")
    if float(torch.linalg.vector_norm(matrix)) <= EPS:
        return None, "zero_matrix"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    k = min(rank, min(matrix.shape))
    if k < 1:
        return None, "disabled"
    if matrix.numel() <= max_exact_elements or k == min(matrix.shape):
        u = torch.linalg.svd(matrix.to(device=device, dtype=torch.float64), full_matrices=False).U[
            :, :k
        ]
        return u.detach().cpu(), "exact_svd"
    kk = min(k, min(matrix.shape) - 1)
    # SciPy svds returns largest kk when which='LM'; singular vectors sorted.
    u, s, _ = svds(matrix.detach().cpu().numpy(), k=kk, which="LM", v0=np.ones(min(matrix.shape)))
    u = u[:, np.argsort(s)[::-1]].copy()
    return torch.from_numpy(u.astype(np.float64)), "topk_approx"


def subspaces(a: tuple[torch.Tensor | None, str], b: tuple[torch.Tensor | None, str]) -> dict:
    """Return historical principal-angle summaries for presumed orthonormal bases."""
    u, st_a = a
    v, st_b = b
    if u is None or v is None:
        return dict(subspace_status=f"{st_a}|{st_b}")
    if u.shape[0] != v.shape[0] or u.shape[1] != v.shape[1]:
        return dict(subspace_status="incompatible_coordinate_or_rank")
    vals = torch.linalg.svdvals(u.T @ v).clamp(0.0, 1.0)
    rank = u.shape[1]
    return dict(
        subspace_status=f"{st_a}|{st_b}",
        subspace_rank=rank,
        subspace_mean_cos2=float(vals.square().mean()),
        subspace_mean_cos=float(vals.mean()),
        subspace_chordal_normalized=float(
            torch.sqrt((1 - vals.square()).clamp_min(0).sum()) / math.sqrt(rank)
        ),
        subspace_angle_mean_rad=float(torch.acos(vals).mean()),
    )


def paper_subspace_layer_alignment(a: torch.Tensor, b: torch.Tensor, j: int = 3) -> float:
    """Paper's mean of J least-aligned principal directions, Eq. (8)."""
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("TRACE subspace comparisons require aligned shapes")
    if not 1 <= j:
        raise ValueError("j must be positive")
    aa = a.to(torch.float64)
    bb = b.to(torch.float64)
    # Matrices are already orthonormal singular-vector bases.
    cosines = torch.linalg.svdvals(aa.T @ bb).clamp(0.0, 1.0)
    return float(torch.sort(cosines)[0][: min(j, cosines.numel())].mean())


def paper_subspace_hierarchical_alignment(
    layer_scores: dict[str, list[float] | np.ndarray], n_bottom: int = 3
) -> dict:
    """Reduce per-layer subspace scores exactly as Alg. A2."""
    if n_bottom < 1:
        raise ValueError("n_bottom must be positive")
    per = {}
    for component, values in sorted(layer_scores.items()):
        scores = np.asarray(values, dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size:
            per[component] = float(np.sort(scores)[: min(n_bottom, scores.size)].mean())
    return {
        "subspace_similarity": float(np.mean(list(per.values()))) if per else math.nan,
        "subspace_n_components": len(per),
        "subspace_per_component": per,
    }


def principal_basis(
    matrix: torch.Tensor,
    *,
    rank: int,
    relative_tolerance: float | None = None,
    device: str | None = None,
    max_exact_elements: int | None = None,
) -> torch.Tensor:
    """Return a left-singular basis restricted to detected nonzero singular directions.

    This strict API never fills requested directions with singular vectors of
    the null space. A zero matrix has no nonzero column-space directions and
    produces an empty basis. Historical fixed-rank behavior remains in basis.
    With max_exact_elements, large CUDA matrices use a seeded approximate range finder;
    otherwise the decomposition is exact. The returned basis stays on the chosen device.
    """
    from ..devices import resolve_device
    from ._validation import floating_tensor

    device = resolve_device(device) if device is not None else matrix.device
    matrix = floating_tensor(matrix, name="matrix", device=device)
    if matrix.ndim != 2 or rank < 1:
        raise ValueError("matrix must be 2D and rank must be positive")
    if relative_tolerance is not None and (
        not math.isfinite(relative_tolerance) or relative_tolerance < 0
    ):
        raise ValueError("relative_tolerance must be finite and nonnegative")
    if max_exact_elements is not None and max_exact_elements < 1:
        raise ValueError("max_exact_elements must be positive")
    if (
        matrix.is_cuda
        and max_exact_elements is not None
        and matrix.numel() > max_exact_elements
        and rank < min(matrix.shape)
    ):
        from .decomposition import truncated_left_basis

        left, singular = truncated_left_basis(matrix, rank)
    else:
        left, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = relative_tolerance
    if tolerance is None:
        tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps
    retained = min(rank, int((singular > singular[0] * tolerance).sum()))

    return left[:, :retained].detach()


def principal_angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return principal angles in radians between orthonormal column bases.

    The bases must share an ambient coordinate space; their ranks may differ.
    Empty bases and nonorthonormal inputs are rejected as undefined.
    """
    from ._validation import orthonormal_columns

    a = orthonormal_columns(a, name="a")
    b = orthonormal_columns(b, name="b")
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must share the same ambient coordinate dimension")
    cosines = torch.linalg.svdvals(a.T @ b).clamp(0, 1)

    return torch.acos(cosines)
