"""Intrinsic profiles for comparisons across heterogeneous backbones."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..core import SPECTRAL_COMPONENTS, Checkpoint
from ..metrics import paper_trace_similarity, quantile_w1


def _profile(df: pd.DataFrame, source: str, comp: str, column: str, grid_size: int = 16):
    """Interpolate depth-aligned intrinsic profiles; not neural alignment."""
    if df.empty or column not in df.columns or "status" not in df.columns:
        return None
    sub = df[(df.component == comp) & (df.source == source) & df.relative_depth.notna()]
    if column == "spectral_energy_curve":
        sub = sub[(sub.status == "exact") & sub[column].notna()]
    else:
        sub = sub[sub[column].notna()]
    if sub.empty:
        return None
    pairs = []
    for r in sub.itertuples(index=False):
        arr = np.asarray(getattr(r, column), dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if np.isfinite(arr).all():
            pairs.append((float(r.relative_depth), arr))
    if not pairs:
        return None
    sizes = {v.size for _, v in pairs}
    if len(sizes) != 1:
        return None
    xs = np.array([p[0] for p in pairs])
    ys = np.stack([p[1] for p in pairs])
    ux = np.unique(xs)
    ys = np.stack([ys[xs == x].mean(axis=0) for x in ux])
    if len(ux) == 1:
        return np.repeat(ys, grid_size, axis=0)
    return np.stack(
        [np.interp(np.linspace(0, 1, grid_size), ux, ys[:, j]) for j in range(ys.shape[1])], axis=1
    )


def compare_backbones(
    a: Checkpoint,
    b: Checkpoint,
    spectral_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    min_components: int = 3,
) -> dict[str, Any]:
    """B: exact-paper TRACE plus independent intrinsic fingerprint distances."""
    sa = spectral_df[spectral_df.checkpoint == a.uid]
    sb = spectral_df[spectral_df.checkpoint == b.uid]
    da = drift_df[(drift_df.checkpoint == a.uid) & (drift_df.level == "tensor")]
    db = drift_df[(drift_df.checkpoint == b.uid) & (drift_df.level == "tensor")]
    row = dict(
        checkpoint_a=a.uid,
        checkpoint_b=b.uid,
        trained_path_a=str(a.path),
        trained_path_b=str(b.path),
        backbone_a=a.backbone,
        backbone_b=b.backbone,
        language=a.language,
        entity=a.entity,
        fold=1,
        comparison_kind="heterogeneous_fingerprint",
    )
    for source in ("base", "final", "delta"):
        # Exact TRACE Eq. (5) on per-layer Frobenius norm fingerprints.
        # Chen, Shang & Liu (COLM 2026), official implementation:
        # https://github.com/OPTML-Group/LLM-Biometrics/blob/main/llm_biometrics/trace.py
        # This is a different measure from our normalized spectral-CDF L1.
        def trace_fingerprint(frame):
            if frame.empty or "matrix_fro" not in frame.columns:
                return {}
            matching = frame[
                (frame.source == source)
                & (frame.layer.notna())
                & frame.component.isin(SPECTRAL_COMPONENTS)
                & (frame.matrix_fro.notna())
                & (frame.level == "tensor")
            ]
            result = {}
            for comp, group in matching.groupby("component"):
                series = group.groupby("layer")["matrix_fro"].mean().sort_index()
                result[comp] = series.to_numpy(dtype=float)
            return result

        trace = paper_trace_similarity(trace_fingerprint(sa), trace_fingerprint(sb))
        row[f"trace_similarity_{source}"] = trace["trace_similarity"]
        row[f"trace_distance_{source}"] = (
            1.0 - trace["trace_similarity"]
            if math.isfinite(trace["trace_similarity"])
            else math.nan
        )
        row[f"n_components_trace_{source}"] = trace["trace_n_components"]
        for comp, score in trace["trace_per_component"].items():
            row[f"trace_{comp}_{source}"] = score
        for metric, col in [
            ("curve_l1", "spectral_energy_curve"),
            ("quantile_w1", "normalized_weight_quantiles"),
        ]:
            measures = []
            for comp in SPECTRAL_COMPONENTS:
                x = _profile(sa, source, comp, col)
                y = _profile(sb, source, comp, col)
                if x is None or y is None or x.shape != y.shape:
                    continue
                if metric == "curve_l1":
                    measures.append(float(np.abs(x - y).mean()))
                else:
                    measures.append(float(np.mean([quantile_w1(u, v) for u, v in zip(x, y)])))
            row[f"{metric}_{source}"] = (
                float(np.mean(measures)) if len(measures) >= min_components else math.nan
            )
            row[f"n_components_{metric}_{source}"] = len(measures)
        # Norm profile distance explicitly scale-sensitive, use normalized delta profiles below.
    measures = []
    for comp in SPECTRAL_COMPONENTS:

        def xprofile(frame):
            t = frame[(frame.component == comp) & frame.layer.notna()]
            if t.empty:
                return None
            x = (
                t.groupby("layer", as_index=False)["relative_l2"]
                .mean()
                .sort_values("layer")
                .relative_l2.to_numpy()
            )
            if not np.isfinite(x).all():
                return None
            return np.interp(np.linspace(0, 1, 16), np.linspace(0, 1, len(x)), np.log1p(x))

        x = xprofile(da)
        y = xprofile(db)
        if x is not None and y is not None:
            measures.append(float(np.mean(np.abs(x - y))))
    row["adaptation_profile_l1"] = (
        float(np.mean(measures)) if len(measures) >= min_components else math.nan
    )
    row["n_components_adaptation"] = len(measures)
    row["comparability"] = (
        "intrinsic_profiles_only_not_functional_equivalence"
        if len(measures) >= min_components
        else "insufficient_common_components"
    )
    return row


def ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Independent per-metric ranks; never average incomparable distances."""
    if df.empty:
        return df
    for metric in [
        "trace_distance_base",
        "trace_distance_final",
        "trace_distance_delta",
        "curve_l1_base",
        "curve_l1_final",
        "curve_l1_delta",
        "quantile_w1_base",
        "quantile_w1_final",
        "quantile_w1_delta",
        "adaptation_profile_l1",
    ]:
        if metric in df:
            df["rank_" + metric] = df.groupby(["language", "entity"])[metric].rank(
                method="min", ascending=True
            )
            df["n_pairs_" + metric] = (
                df.groupby(["language", "entity"])[metric].transform("count").astype("int64")
            )
    return df
