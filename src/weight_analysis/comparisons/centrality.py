"""Descriptive centrality from explicit pairwise model distances."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def model_distance_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    rows = []
    dist_metrics = (
        "trace_distance_delta",
        "trace_distance_final",
        "curve_l1_delta",
        "quantile_w1_delta",
        "adaptation_profile_l1",
        "curve_l1_final",
        "quantile_w1_final",
    )
    for (language, entity), group in pair_df.groupby(["language", "entity"]):
        models = sorted(set(group.backbone_a) | set(group.backbone_b))
        for metric in dist_metrics:
            if metric not in group:
                continue
            for model in models:
                sub = group[(group.backbone_a == model) | (group.backbone_b == model)]
                vals = []
                for q in sub.itertuples(index=False):
                    v = getattr(q, metric)
                    if np.isfinite(v):
                        other = q.backbone_b if q.backbone_a == model else q.backbone_a
                        vals.append((other, float(v)))
                n_expected = len(models) - 1
                values = np.array([v for _, v in vals], float)
                if values.size:
                    nearest = min(vals, key=lambda p: (p[1], p[0]))
                    furthest = max(vals, key=lambda p: (p[1], p[0]))
                else:
                    nearest = (None, math.nan)
                    furthest = (None, math.nan)
                rows.append(
                    dict(
                        language=language,
                        entity=entity,
                        backbone=model,
                        metric=metric,
                        n_neighbors=len(vals),
                        n_possible_neighbors=n_expected,
                        complete_comparison=len(vals) == n_expected,
                        mean_distance=float(values.mean()) if len(vals) else math.nan,
                        median_distance=float(np.median(values)) if len(vals) else math.nan,
                        nearest_backbone=nearest[0],
                        nearest_distance=nearest[1],
                        furthest_backbone=furthest[0],
                        furthest_distance=furthest[1],
                        interpretation="intrinsic_fingerprint_distance_not_functional_similarity",
                    )
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rank_most_central"] = float("nan")
    valid = df.complete_comparison
    if valid.any():
        df.loc[valid, "rank_most_central"] = (
            df[valid].groupby(["language", "entity", "metric"])["mean_distance"].rank(method="min")
        )
    return df


def model_centrality(
    pair_df: pd.DataFrame, *, metrics: Sequence[str] | None = None
) -> pd.DataFrame:
    """Summarize unique undirected distances within each explicit study stratum.

    Unlike the historical ``model_distance_summary``, duplicate or self edges
    are rejected. Optional fold/seed columns define separate strata. Missing
    distances remain missing neighbors and exclude that model from ranking.
    No shortest-path or functional-similarity interpretation is implied.
    """
    required = {"language", "entity", "backbone_a", "backbone_b"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(f"pair_df is missing columns: {sorted(missing)}")
    if pair_df.empty:
        return pd.DataFrame()
    if pair_df[list(required)].isna().any().any():
        raise ValueError("pair_df model and study identities must not be missing")

    available = (
        "trace_distance_delta",
        "trace_distance_final",
        "curve_l1_delta",
        "quantile_w1_delta",
        "adaptation_profile_l1",
        "curve_l1_final",
        "quantile_w1_final",
    )
    selected = (
        tuple(metrics)
        if metrics is not None
        else tuple(metric for metric in available if metric in pair_df)
    )
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("metrics must contain unique distance columns")
    missing_metrics = set(selected) - set(pair_df.columns)
    if missing_metrics:
        raise ValueError(f"pair_df is missing selected distance columns: {sorted(missing_metrics)}")

    frame = pair_df.copy()
    for column in ("backbone_a", "backbone_b"):
        frame[column] = frame[column].astype(str)
    if (frame.backbone_a == frame.backbone_b).any():
        raise ValueError("pair_df must not contain self comparisons")
    ordered = np.sort(frame[["backbone_a", "backbone_b"]].to_numpy(), axis=1)
    frame["backbone_a"] = ordered[:, 0]
    frame["backbone_b"] = ordered[:, 1]
    strata = ["language", "entity"] + [column for column in ("fold", "seed") if column in frame]
    identities = strata + ["backbone_a", "backbone_b"]
    if frame.duplicated(identities).any():
        raise ValueError("pair_df contains duplicate undirected model pairs within a stratum")
    for metric in selected:
        values = pd.to_numeric(frame[metric], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
            raise ValueError(
                f"pair_df distance {metric!r} must be nonnegative and finite or missing"
            )
        frame[metric] = values
    frame = frame.sort_values(identities, kind="stable").reset_index(drop=True)

    rows = []
    for identity, group in frame.groupby(strata, dropna=False, sort=True):
        identity = identity if isinstance(identity, tuple) else (identity,)
        study = dict(zip(strata, identity))
        models = sorted(set(group.backbone_a) | set(group.backbone_b))
        for metric in selected:
            for model in models:
                edges = group[(group.backbone_a == model) | (group.backbone_b == model)]
                neighbors = []
                for edge in edges.itertuples(index=False):
                    value = getattr(edge, metric)
                    if math.isfinite(value):
                        other = edge.backbone_b if edge.backbone_a == model else edge.backbone_a
                        neighbors.append((other, float(value)))
                values = np.asarray([value for _, value in neighbors], dtype=float)
                complete = {neighbor for neighbor, _ in neighbors} == set(models) - {model}
                nearest = (
                    min(neighbors, key=lambda item: (item[1], item[0]))
                    if neighbors
                    else (None, math.nan)
                )
                furthest = (
                    max(neighbors, key=lambda item: (item[1], item[0]))
                    if neighbors
                    else (None, math.nan)
                )
                rows.append(
                    {
                        **study,
                        "backbone": model,
                        "metric": metric,
                        "n_neighbors": len(neighbors),
                        "n_possible_neighbors": len(models) - 1,
                        "complete_comparison": complete,
                        "mean_distance": float(values.mean()) if len(values) else math.nan,
                        "median_distance": float(np.median(values)) if len(values) else math.nan,
                        "nearest_backbone": nearest[0],
                        "nearest_distance": nearest[1],
                        "furthest_backbone": furthest[0],
                        "furthest_distance": furthest[1],
                        "interpretation": "intrinsic_fingerprint_distance_not_functional_similarity",
                    }
                )
    result = pd.DataFrame(rows)
    result["rank_most_central"] = math.nan
    complete = result.complete_comparison
    result.loc[complete, "rank_most_central"] = (
        result[complete]
        .groupby(strata + ["metric"], dropna=False)["mean_distance"]
        .rank(method="min")
    )

    return result
