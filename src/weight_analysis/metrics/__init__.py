"""Public independent mathematical primitives and historical metric contracts."""

from __future__ import annotations

from scipy.sparse.linalg import svds

from ._validation import EPS
from .overlap import topk_overlap
from .profiles import paper_trace_similarity, quantile_w1
from .spectral import SPECTRAL_METRICS, spectral, spectral_metrics
from .subspace import (
    basis,
    paper_subspace_hierarchical_alignment,
    paper_subspace_layer_alignment,
    principal_angles,
    principal_basis,
    subspaces,
)
from .vector import (
    DRIFT_METRICS,
    PAIR_METRICS,
    VectorAccumulator,
    drift,
    drift_metrics,
    pair_metrics,
    relative_frobenius,
)

__all__ = [
    "DRIFT_METRICS",
    "EPS",
    "PAIR_METRICS",
    "SPECTRAL_METRICS",
    "VectorAccumulator",
    "basis",
    "drift",
    "drift_metrics",
    "pair_metrics",
    "paper_subspace_hierarchical_alignment",
    "paper_subspace_layer_alignment",
    "paper_trace_similarity",
    "principal_angles",
    "principal_basis",
    "quantile_w1",
    "relative_frobenius",
    "spectral",
    "spectral_metrics",
    "subspaces",
    "topk_overlap",
]
