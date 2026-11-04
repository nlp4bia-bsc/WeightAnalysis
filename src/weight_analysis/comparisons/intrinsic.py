"""Reuse intrinsic fingerprints across incompatible backbones and performance selections."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from ..core import WeightReader
from ..core.catalog import geometry_dimensions, stable_id
from ..core.geometry_schema import observation
from ..metrics import quantile_w1
from ..results.models import data_digest
from .backbones import _profile, compare_backbones


def fingerprint_requirements(names):
    """Declare spectral dependencies without loading weights or executing metrics."""
    dependencies = {}
    for name in names:
        if name == "adaptation_profile_l1":
            continue
        source = name.rsplit("_", 1)[1]
        metric = (
            "matrix_fro"
            if name.startswith("trace_")
            else (
                "spectral_energy_curve"
                if name.startswith("curve_l1")
                else "normalized_weight_quantiles"
            )
        )
        dependencies.setdefault(source, set()).add(metric)
    return dependencies


def intrinsic_geometry(ctx, a, b, comparison_id, names):
    if not names:
        return {}
    dependencies = fingerprint_requirements(names)
    wide, drift, detailed, points, stats = [], [], [], [], []
    for entry in (a, b):
        for key in ctx.keys(entry):
            with WeightReader(entry.checkpoint.path) as reader:
                matrix_parameter = len(reader.shape_dtype(key)[0]) == 2
            for source, metrics in sorted(dependencies.items()) if matrix_parameter else []:
                result = ctx.reporter.attempt(
                    stable_id([entry.artifact_id, key, source, "intrinsic"]),
                    lambda: ctx.fingerprint(entry, key, source, tuple(sorted(metrics))),
                    on_error=ctx.on_error,
                )
                if result:
                    wide.extend(result.get("fingerprints", []))
                    detailed.extend(result.get("geometry_detailed", []))
                    points.extend(result.get("profile_points", []))
            if "adaptation_profile_l1" in names:
                result = ctx.vector(
                    entry,
                    None,
                    key,
                    "delta",
                    ("relative_l2",),
                    subject_id=entry.artifact_id,
                    kind="drift",
                )
                for row in result.get("geometry_detailed", []):
                    drift.append(
                        {
                            **row,
                            "checkpoint": entry.artifact_id,
                            "level": "tensor",
                            "relative_l2": row["value"],
                        }
                    )
                detailed.extend(result.get("geometry_detailed", []))
                stats.extend(result.get("sufficient_statistics", []))
    spectra = (
        pd.DataFrame(wide).drop_duplicates(["checkpoint", "tensor", "source"])
        if wide
        else pd.DataFrame()
    )
    drift_df = (
        pd.DataFrame(drift)
        if drift
        else pd.DataFrame(columns=["checkpoint", "level", "component", "layer", "relative_l2"])
    )
    if spectra.empty and dependencies:
        raise ValueError("No available intrinsic fingerprints for selected parameters")
    if spectra.empty:
        spectra = pd.DataFrame(
            columns=["checkpoint", "level", "component", "layer", "source", "status"]
        )
    descriptors = [SimpleNamespace(uid=e.artifact_id, **e.checkpoint.__dict__) for e in (a, b)]
    references = {}
    if any(name.endswith(("_base", "_delta")) or name == "adaptation_profile_l1" for name in names):
        references = {side: ctx.reference(entry)[1] for side, entry in (("a", a), ("b", b))}

    def compare(uid):
        rows = []
        for component in [None] + sorted(
            (set(spectra.component) | set(drift_df.component)) - {"embeddings"}
        ):
            frame = spectra if component is None else spectra[spectra.component == component]
            base = {
                **geometry_dimensions(a, b),
                "comparison_type": "intrinsic",
                "execution_id": stable_id(ctx.definition),
                "subject_id": comparison_id,
                "subject_kind": "pair",
                "tensor": None,
                "layer": None,
                "component": component,
                "embedding_type": None,
                "domain": "intrinsic",
                "reference_id": stable_id(references) if references else None,
            }
            if frame.empty:
                frame = pd.DataFrame(
                    columns=["checkpoint", "level", "component", "layer", "source", "status"]
                )
            values = compare_backbones(
                *descriptors,
                frame,
                drift_df if component is None else drift_df[drift_df.component == component],
                ctx.options.min_fingerprint_components if component is None else 1,
                device=ctx.device,
            )
            for name in names:
                source = name.rsplit("_", 1)[1] if name != "adaptation_profile_l1" else "delta"
                rows.append(
                    observation(
                        uid,
                        {
                            **base,
                            "source": source,
                            "reference_id": None if source == "final" else base["reference_id"],
                        },
                        name,
                        values[name],
                        scope="encoder" if component is None else "component",
                    )
                )
            if component is not None:
                for name in names:
                    if not name.startswith(("curve_l1_", "quantile_w1_")):
                        continue
                    source = name.rsplit("_", 1)[1]
                    column = (
                        "spectral_energy_curve"
                        if name.startswith("curve_l1")
                        else "normalized_weight_quantiles"
                    )
                    left = _profile(
                        frame[frame.checkpoint == a.artifact_id],
                        source,
                        component,
                        column,
                        device=ctx.device,
                    )
                    right = _profile(
                        frame[frame.checkpoint == b.artifact_id],
                        source,
                        component,
                        column,
                        device=ctx.device,
                    )
                    if left is None or right is None:
                        continue
                    for index, (x, y) in enumerate(zip(left, right, strict=True)):
                        value = (
                            float((x - y).abs().mean())
                            if column == "spectral_energy_curve"
                            else quantile_w1(x, y)
                        )
                        rows.append(
                            observation(
                                uid,
                                {
                                    **base,
                                    "source": source,
                                    "reference_id": None
                                    if source == "final"
                                    else base["reference_id"],
                                },
                                name,
                                value,
                                scope="relative_depth",
                                relative_depth=index / (len(left) - 1),
                                depth_alignment="relative_linear_16",
                            )
                        )
        if not spectra.empty:
            embeddings = spectra[spectra.component == "embeddings"]
            for kind, group in embeddings.groupby("embedding_type"):
                for name in names:
                    if not name.startswith(("curve_l1_", "quantile_w1_")):
                        continue
                    source = name.rsplit("_", 1)[1]
                    column = (
                        "spectral_energy_curve"
                        if name.startswith("curve_l1")
                        else "normalized_weight_quantiles"
                    )
                    chunks = [
                        group[(group.checkpoint == entry.artifact_id) & (group.source == source)]
                        for entry in (a, b)
                    ]
                    if any(len(part) != 1 for part in chunks):
                        ctx.reporter.emit(
                            "skipped",
                            stable_id([comparison_id, kind, name]),
                            "Embedding fingerprints need one matrix per kind and source",
                            comparison_id=comparison_id,
                            embedding_type=kind,
                            metric=name,
                        )
                    x, y = [part.iloc[0][column] if len(part) == 1 else None for part in chunks]
                    value = (
                        None
                        if x is None or y is None
                        else (
                            float(
                                (
                                    torch.tensor(x, device=ctx.device, dtype=torch.float64)
                                    - torch.tensor(y, device=ctx.device, dtype=torch.float64)
                                )
                                .abs()
                                .mean()
                            )
                            if column == "spectral_energy_curve"
                            else quantile_w1(x, y, device=ctx.device)
                        )
                    )
                    rows.append(
                        observation(
                            uid,
                            {
                                **geometry_dimensions(a, b),
                                "comparison_type": "intrinsic",
                                "execution_id": stable_id(ctx.definition),
                                "subject_id": comparison_id,
                                "subject_kind": "pair",
                                "source": source,
                                "reference_id": stable_id(references)
                                if references and source != "final"
                                else None,
                                "layer": None,
                                "component": "embeddings",
                                "embedding_type": kind,
                                "tensor": None,
                                "domain": "intrinsic",
                            },
                            name,
                            value,
                            scope="embedding",
                        )
                    )
        return {"geometry_detailed": rows}

    _, comparison = ctx.unit(
        {
            "kind": "intrinsic_pair",
            "pair": comparison_id,
            "metrics": sorted(names),
            "parameters": ctx.parameters.__dict__,
            "references": references,
            "fingerprint_digest": data_digest(spectra),
            "drift_digest": data_digest(drift_df),
        },
        compare,
        stage="intrinsic_comparison",
    )
    return {
        "geometry_detailed": detailed + comparison.get("geometry_detailed", []),
        "profile_points": points,
        "sufficient_statistics": stats,
    }
