"""Coordinate overlap diagnostics with deterministic magnitude ties.

The historical functions retain the numerical definitions of WeightAnalysis 2.0.0.
See docs/mathematical_audit.md for assumptions and limitations.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ._validation import EPS


def topk_overlap(a: torch.Tensor, b: torch.Tensor, fraction: float) -> dict:
    """Compare stable nonzero magnitude supports and absolute-weight Jaccard."""
    if a.shape != b.shape or not 0 < fraction <= 1:
        raise ValueError("invalid top-k inputs")
    aa = a.detach().cpu().reshape(-1).abs().numpy()
    bb = b.detach().cpu().reshape(-1).abs().numpy()
    n = aa.size
    if not n:
        raise ValueError("empty tensor")
    if not (np.isfinite(aa).all() and np.isfinite(bb).all()):
        raise ValueError("nonfinite top-k inputs")
    k = max(1, math.ceil(n * fraction))
    # Stable ties: descending magnitude, then increasing coordinate index.
    # Do not count unchanged zero-valued coordinates as selected parameters.
    # For sparse updates, the effective top-k set can be smaller than k.
    ia = np.lexsort((np.arange(n), -aa))
    ib = np.lexsort((np.arange(n), -bb))
    ia = ia[aa[ia] > 0][:k]
    ib = ib[bb[ib] > 0][:k]
    common = int(np.intersect1d(ia, ib, assume_unique=True).size)
    union = len(ia) + len(ib) - common
    # Weighted Ruzicka/Jaccard on absolute updates is a classical generalized
    # Jaccard coefficient (intersection=min, union=max), applied here to Δθ.
    small = float(np.minimum(aa, bb).sum(dtype=np.float64))
    large = float(np.maximum(aa, bb).sum(dtype=np.float64))
    shared_sign = np.sign(a.detach().cpu().reshape(-1).numpy()) == np.sign(
        b.detach().cpu().reshape(-1).numpy()
    )
    intersect_mask = np.zeros(n, dtype=bool)
    intersect_mask[np.intersect1d(ia, ib, assume_unique=True)] = True
    sign_consistent = int(np.sum(shared_sign & intersect_mask))
    expected = len(ia) * len(ib) / n
    return dict(
        topk_fraction=fraction,
        topk_k=k,
        topk_k_a=len(ia),
        topk_k_b=len(ib),
        topk_intersection=common,
        topk_jaccard=common / union if union else math.nan,
        topk_overlap_coefficient=common / min(len(ia), len(ib))
        if len(ia) and len(ib)
        else math.nan,
        topk_expected_intersection=expected,
        topk_excess_intersection=common - expected,
        weighted_jaccard_abs=small / large if large > EPS else math.nan,
        topk_sign_agreement=sign_consistent / common if common else math.nan,
    )
