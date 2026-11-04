"""Aligned task-update and final-weight comparisons within one backbone."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..core import (
    SPECTRAL_COMPONENTS,
    Checkpoint,
    WeightReader,
    audit_alignment,
    config_metadata,
    get_delta,
    head_views,
    identify,
)
from ..metrics import (
    VectorAccumulator,
    basis,
    paper_subspace_hierarchical_alignment,
    paper_subspace_layer_alignment,
    subspaces,
    topk_overlap,
)


def compare_related(
    a: Checkpoint, b: Checkpoint, base_path: Path, settings: Any, subspace_cache: dict | None = None
) -> list[dict[str, Any]]:
    """A2-A4: pair aligned deltas + raw final weights (separate comparison_kind)."""
    from ..devices import resolve_device

    compute_device = resolve_device(getattr(settings, "device", "cpu"))
    rows = []
    aggregators = defaultdict(VectorAccumulator)
    # Scores per layer/component for Chen et al. (COLM 2026), Eq. 8, Alg A2.
    paper_layer_scores = defaultdict(list)
    head_config = config_metadata(base_path)
    with (
        WeightReader(a.path) as ra,
        WeightReader(b.path) as rb,
        WeightReader(base_path) as original,
    ):
        ka, sa = audit_alignment(ra, original, settings.min_aligned_fraction)
        kb, sb = audit_alignment(rb, original, settings.min_aligned_fraction)
        matching = sorted(set(ka) & set(kb))
        a_els = sum(ra.get(k).numel() for k in ka)
        b_els = sum(rb.get(k).numel() for k in kb)
        covered = sum(ra.get(k).numel() for k in matching)
        fraction = covered / max(a_els, b_els)
        if fraction < settings.min_aligned_fraction:
            raise ValueError(
                f"Pair has poor matched-element coverage: {a.uid} {b.uid} {fraction:.3f}"
            )
        shared = dict(
            checkpoint_a=a.uid,
            checkpoint_b=b.uid,
            original_path=str(base_path),
            trained_path_a=str(a.path),
            trained_path_b=str(b.path),
            backbone=a.backbone,
            language_a=a.language,
            language_b=b.language,
            entity_a=a.entity,
            entity_b=b.entity,
            fold_a=a.fold,
            fold_b=b.fold,
            pair_element_coverage=fraction,
        )
        for key in matching:
            fa, ini, da = get_delta(ra, original, key, device=compute_device)
            fb, ini2, db = get_delta(rb, original, key, device=compute_device)
            if not torch.equal(ini, ini2):
                raise ValueError("Base tensor changed between pair reads")
            layer, comp, _ = identify(key, tuple(fa.shape))
            for kind, ma, mb in [("delta", da, db), ("final", fa, fb)]:
                for scope, lay, section, tensor in [
                    ("tensor", layer, comp, key),
                    ("layer", layer, None, None),
                    ("component", None, comp, None),
                    ("layer_component", layer, comp, None),
                    ("encoder", None, None, None),
                ]:
                    aggregators[kind, scope, lay, section, tensor].update(
                        ma, mb, settings.sign_floor
                    )
                for (hi, ah), (hj, bh) in zip(
                    head_views(ma, comp, head_config), head_views(mb, comp, head_config)
                ):
                    if hi != hj:
                        raise ValueError("Attention head index mismatch")
                    aggregators[kind, "head", layer, comp, f"{key}::head{hi}"].update(
                        ah, bh, settings.sign_floor
                    )
            if da.ndim == 2 and comp in SPECTRAL_COMPONENTS and settings.subspace_rank > 0:
                # Cache singular vectors for this backbone's checkpoints only.
                # Correct shared coordinate system is guaranteed by alignment above.
                if subspace_cache is None:
                    left = basis(
                        da, settings.subspace_rank, settings.device, settings.max_svd_elements
                    )
                    right = basis(
                        db, settings.subspace_rank, settings.device, settings.max_svd_elements
                    )
                else:
                    for ck, tensor in ((a, da), (b, db)):
                        ident = (ck.uid, key)
                        if ident not in subspace_cache:
                            subspace_cache[ident] = basis(
                                tensor,
                                settings.subspace_rank,
                                settings.device,
                                settings.max_svd_elements,
                            )
                    left = subspace_cache[a.uid, key]
                    right = subspace_cache[b.uid, key]
                subspace_stats = subspaces(left, right)
                if left[0] is not None and right[0] is not None and left[0].shape == right[0].shape:
                    paper_score = paper_subspace_layer_alignment(left[0], right[0], j=3)
                    subspace_stats["paper_colm26_worst3_cosines"] = paper_score
                    if layer is not None:
                        paper_layer_scores[(comp, layer)].append(paper_score)
                rows.append(
                    dict(
                        **shared,
                        level="tensor",
                        layer=layer,
                        component=comp,
                        tensor=key,
                        comparison_kind="delta",
                        metric_family="subspace",
                        **subspace_stats,
                    )
                )
            if settings.topk_fraction > 0 and da.numel() <= settings.max_topk_elements:
                rows.append(
                    dict(
                        **shared,
                        level="tensor",
                        layer=layer,
                        component=comp,
                        tensor=key,
                        comparison_kind="delta",
                        metric_family="overlap",
                        **topk_overlap(da, db, settings.topk_fraction),
                    )
                )
        for (kind, scope, layer, comp, key), metric in aggregators.items():
            rows.append(
                dict(
                    **shared,
                    level=scope,
                    layer=layer,
                    component=comp,
                    tensor=key,
                    comparison_kind=kind,
                    metric_family="vector",
                    **metric.result(),
                )
            )
        if paper_layer_scores:
            # Aggregate duplicate matrices in the same layer/component (if any)
            # before selecting least-aligned layers, as in Chen et al. Alg A2.
            per_component = defaultdict(list)
            for (comp, layer), vals in sorted(paper_layer_scores.items()):
                per_component[comp].append(float(np.mean(vals)))
            summary = paper_subspace_hierarchical_alignment(per_component, n_bottom=3)
            rows.append(
                dict(
                    **shared,
                    level="encoder",
                    layer=None,
                    component=None,
                    tensor=None,
                    comparison_kind="delta",
                    metric_family="subspace_paper_colm26",
                    paper_colm26_hierarchical_similarity=summary["subspace_similarity"],
                    paper_colm26_n_components=summary["subspace_n_components"],
                    paper_colm26_component_scores=str(summary["subspace_per_component"]),
                )
            )
    return rows


def rank_related_encoder_pairs(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Rank A2/A3 pairs only within a common backbone and entity or language."""
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["rank_delta_cosine"] = float("nan")
    mask = (
        (frame.level == "encoder")
        & (frame.comparison_kind == "delta")
        & (frame.metric_family == "vector")
    )
    grouping = ["backbone", "entity_a"] if mode == "cross_language" else ["backbone", "language_a"]
    if mask.any() and set(grouping) <= set(frame.columns):
        frame.loc[mask, "rank_delta_cosine"] = (
            frame[mask].groupby(grouping)["cosine"].rank(method="min", ascending=False)
        )
    return frame
