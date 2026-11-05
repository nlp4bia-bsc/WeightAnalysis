"""Compatibility exports for the historical weightspace mathematical API."""

from weight_analysis.metrics import (
    EPS,
    VectorAccumulator,
    basis,
    drift,
    paper_subspace_hierarchical_alignment,
    paper_subspace_layer_alignment,
    paper_trace_similarity,
    quantile_w1,
    spectral,
    subspaces,
    svds,
    topk_overlap,
)

__all__ = [
    "EPS",
    "VectorAccumulator",
    "basis",
    "drift",
    "paper_subspace_hierarchical_alignment",
    "paper_subspace_layer_alignment",
    "paper_trace_similarity",
    "quantile_w1",
    "spectral",
    "subspaces",
    "svds",
    "topk_overlap",
]
