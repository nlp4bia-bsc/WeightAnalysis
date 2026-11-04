"""Selective comparisons for independent scientific computations."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..core import WeightReader, audit_alignment, identify, selected_keys
from ..metrics import PAIR_METRICS, pair_metrics, topk_overlap
from .catalog import OVERLAP_METRICS, SUBSPACE_METRICS
from .descriptions import _base, _describe
from .selection import _pairs


def _related(pairs, originals, *, layers, components, tensors, metrics, sources, options):
    from ..metrics import principal_angles, principal_basis

    rows = []
    for a, b in pairs:
        base_path = _base(a, originals)
        _base(b, originals)
        with (
            WeightReader(a.path) as left,
            WeightReader(b.path) as right,
            WeightReader(base_path) as base,
        ):
            ka, _ = audit_alignment(left, base, options.min_aligned_fraction)
            kb, _ = audit_alignment(right, base, options.min_aligned_fraction)
            common = set(ka) & set(kb)
            covered = sum(np.prod(left.shape_dtype(key)[0]) for key in common)
            expected = max(
                sum(np.prod(left.shape_dtype(key)[0]) for key in ka),
                sum(np.prod(right.shape_dtype(key)[0]) for key in kb),
            )
            if covered / expected < options.min_aligned_fraction:
                raise ValueError("Pair lacks common encoder coordinate coverage")
            keys = sorted(
                common
                & set(selected_keys(left, layers=layers, components=components, tensors=tensors))
            )
            for key in keys:
                layer, component, _ = identify(key, left.shape_dtype(key)[0])
                fa = left.get(key)
                fb = right.get(key)
                if "delta" in sources:
                    import torch

                    initial = base.get(key).to(torch.float64)
                    da, db = fa.to(torch.float64) - initial, fb.to(torch.float64) - initial
                else:
                    da = db = None
                for source in sources:
                    x, y = (da, db) if source == "delta" else (fa, fb)
                    row = {
                        "checkpoint_a": a.uid,
                        "checkpoint_b": b.uid,
                        "backbone": a.backbone,
                        "language_a": a.language,
                        "language_b": b.language,
                        "entity_a": a.entity,
                        "entity_b": b.entity,
                        "fold": a.fold,
                        "seed": a.seed,
                        "tensor": key,
                        "layer": layer,
                        "component": component,
                        "level": "tensor",
                        "source": source,
                    }
                    vector_names = [name for name in metrics if name in PAIR_METRICS]
                    if vector_names:
                        row.update(
                            pair_metrics(x, y, metrics=vector_names, sign_floor=options.sign_floor)
                        )
                    overlap_names = set(metrics) & set(OVERLAP_METRICS)
                    if overlap_names:
                        overlap = topk_overlap(x, y, options.topk_fraction)
                        row.update({name: overlap[name] for name in overlap_names})
                    subspace_names = set(metrics) & set(SUBSPACE_METRICS)
                    if subspace_names:
                        if x.ndim != 2 or min(x.shape) < 2:
                            row["subspace_status"] = "not_matrix"
                        else:
                            u = principal_basis(
                                x, rank=options.subspace_rank, device=options.device
                            )
                            v = principal_basis(
                                y, rank=options.subspace_rank, device=options.device
                            )
                            if u.shape[1] == 0 or v.shape[1] == 0 or u.shape[1] != v.shape[1]:
                                row["subspace_status"] = "undefined_or_different_numerical_rank"
                            else:
                                angles = principal_angles(u, v)
                                values = {
                                    "subspace_mean_cos2": float(angles.cos().square().mean()),
                                    "subspace_chordal_normalized": float(
                                        angles.sin().square().mean().sqrt()
                                    ),
                                    "subspace_angle_mean_rad": float(angles.mean()),
                                }
                                row.update({name: values[name] for name in subspace_names})
                                row["subspace_status"] = "exact_numerical_rank"
                    rows.append(row)
    return pd.DataFrame(rows)


def _cross_backbone(
    checkpoints,
    originals,
    *,
    metrics,
    pairs,
    layers,
    components,
    tensors,
    options,
    spectral_df,
    drift_df,
):
    from ..comparisons import compare_backbones

    selected_pairs = _pairs(checkpoints, "cross_backbone", pairs)
    if not selected_pairs:
        return pd.DataFrame()
    endpoints = {checkpoint.uid: checkpoint for pair in selected_pairs for checkpoint in pair}
    selected_checkpoints = [endpoints[uid] for uid in sorted(endpoints)]
    selected_originals = {
        name: path
        for name, path in originals.items()
        if name in {checkpoint.backbone for checkpoint in selected_checkpoints}
    }
    dependencies = defaultdict(set)
    for name in metrics:
        if name == "adaptation_profile_l1":
            continue
        source = name.rsplit("_", 1)[1]
        if name.startswith("trace_"):
            dependencies[source].add("matrix_fro")
        elif name.startswith("curve_l1"):
            dependencies[source].add("spectral_energy_curve")
        elif name.startswith("quantile_w1"):
            dependencies[source].add("normalized_weight_quantiles")
    need_drift = "adaptation_profile_l1" in metrics
    drift_missing = need_drift and drift_df is None
    if dependencies and spectral_df is None:
        spectral_parts = []
        for source, names in sorted(dependencies.items()):
            new_drift, new_spectra = _describe(
                selected_checkpoints,
                selected_originals,
                layers=layers,
                components=components,
                tensors=tensors,
                drift_names=("relative_l2",) if drift_missing else (),
                spectral_names=sorted(names),
                sources=(source,),
                options=options,
            )
            spectral_parts.append(new_spectra)
            if drift_missing:
                drift_df = new_drift
                drift_missing = False
        spectral_df = pd.concat(spectral_parts, ignore_index=True)
    if drift_missing:
        drift_df, _ = _describe(
            selected_checkpoints,
            selected_originals,
            layers=layers,
            components=components,
            tensors=tensors,
            drift_names=("relative_l2",),
            spectral_names=(),
            sources=(),
            options=options,
        )
    if drift_df is None:
        drift_df = pd.DataFrame(
            columns=["checkpoint", "level", "component", "layer", "relative_l2"]
        )
    if spectral_df is None:
        spectral_df = pd.DataFrame(
            columns=["checkpoint", "level", "component", "layer", "source", "status"]
        )

    if dependencies:
        required = {"checkpoint", "level", "component", "layer", "source", "status", "tensor"}
        required |= set.union(*dependencies.values())
        if any(
            name in {"spectral_energy_curve", "normalized_weight_quantiles"}
            for names in dependencies.values()
            for name in names
        ):
            required.add("relative_depth")
        absent = required - set(spectral_df.columns)
        if absent:
            raise ValueError(f"Spectral results missing columns: {sorted(absent)}")
        spectral_df = spectral_df[
            spectral_df.checkpoint.isin(endpoints)
            & spectral_df.source.isin(dependencies)
            & (spectral_df.level == "tensor")
        ].copy()
        if spectral_df[["checkpoint", "source", "tensor", "component"]].isna().any().any():
            raise ValueError("Spectral results contain missing parameter/source identities")
        if spectral_df.duplicated(["checkpoint", "source", "tensor"]).any():
            raise ValueError(
                "Spectral results contain duplicate checkpoint/source/tensor observations"
            )
        observed = set(zip(spectral_df.checkpoint, spectral_df.source))
        missing = {(uid, source) for uid in endpoints for source in dependencies} - observed
        if missing:
            raise ValueError("Spectral results are missing selected checkpoint/source observations")
        for source, names in dependencies.items():
            subset = spectral_df[spectral_df.source == source]
            if "matrix_fro" in names:
                values = pd.to_numeric(subset.matrix_fro, errors="raise").to_numpy(dtype=float)
                if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
                    raise ValueError(
                        "Spectral matrix_fro must be nonnegative and finite or missing"
                    )
            for name in names & {"spectral_energy_curve", "normalized_weight_quantiles"}:
                for value in subset[name]:
                    if value is None or (np.isscalar(value) and pd.isna(value)):
                        continue
                    array = np.asarray(value, dtype=float)
                    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
                        raise ValueError(
                            f"Spectral {name} requires finite one-dimensional profiles"
                        )
                    if (np.diff(array) < -1e-12).any():
                        raise ValueError(f"Spectral {name} profiles must be nondecreasing")
            if "relative_depth" in subset:
                depth = pd.to_numeric(subset.relative_depth, errors="raise").to_numpy(dtype=float)
                finite = depth[np.isfinite(depth)]
                if np.isinf(depth).any() or ((finite < 0) | (finite > 1)).any():
                    raise ValueError("Spectral relative_depth must lie in [0, 1] or be missing")
    if need_drift:
        required = {"checkpoint", "level", "component", "layer", "tensor", "relative_l2"}
        absent = required - set(drift_df.columns)
        if absent:
            raise ValueError(f"Drift results missing columns: {sorted(absent)}")
        drift_df = drift_df[
            drift_df.checkpoint.isin(endpoints) & (drift_df.level == "tensor")
        ].copy()
        if drift_df[["checkpoint", "tensor", "component"]].isna().any().any():
            raise ValueError("Drift results contain missing parameter identities")
        if drift_df.duplicated(["checkpoint", "tensor"]).any():
            raise ValueError("Drift results contain duplicate checkpoint/tensor observations")
        if set(endpoints) - set(drift_df.checkpoint):
            raise ValueError("Drift results are missing selected checkpoint observations")
        values = pd.to_numeric(drift_df.relative_l2, errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
            raise ValueError("Drift relative_l2 must be nonnegative and finite or missing")
    rows = []
    for a, b in selected_pairs:
        values = compare_backbones(a, b, spectral_df, drift_df, options.min_fingerprint_components)
        metadata = {
            name: value
            for name, value in values.items()
            if name.startswith(("checkpoint", "trained_path", "backbone", "n_components"))
            or name in ("language", "entity", "comparability", "comparison_kind")
        }
        rows.append(
            {**metadata, "fold": a.fold, "seed": a.seed, **{name: values[name] for name in metrics}}
        )
    return pd.DataFrame(rows)
