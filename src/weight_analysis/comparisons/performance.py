"""Performance-selected direct and intrinsic geometry with independent failure boundaries."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..configuration import MetricOptions
from ..core import WeightReader
from ..core.blocks import parameter_location
from ..core.catalog import ReferenceSpec, geometry_dimensions, stable_id
from ..core.geometry_schema import DEFINITION, SHIFT_INVARIANT, ParameterSelection, observation
from ..core.reporting import RunReporter
from ..execution.comparative_units import ComparativeContext
from ..metrics import (
    DRIFT_METRICS,
    PAIR_METRICS,
    SPECTRAL_METRICS,
    principal_angles,
    principal_basis,
    topk_overlap,
)
from ..results.comparative import ComparativeResult, event_table, issue_table, merge_comparative
from .intrinsic import fingerprint_requirements, intrinsic_geometry
from .selection import SelectionResult

SPECIAL_METRICS = (
    "topk_jaccard",
    "weighted_jaccard_abs",
    "topk_sign_agreement",
    "subspace_mean_cos2",
    "subspace_chordal_normalized",
    "subspace_angle_mean_rad",
)


def _copy_units(target, unit):
    for name, rows in unit.items():
        if name != "fingerprints":
            target.setdefault(name, []).extend(rows)


def _base_tables(selection):
    rankings = selection.rankings.copy()
    if not rankings.empty:
        rankings["ranking_id"] = [
            stable_id([r.selection_id, r.artifact_id]) for r in rankings.itertuples()
        ]
    exclusions = selection.exclusions.copy()
    if not exclusions.empty:
        exclusions["exclusion_id"] = [
            stable_id([selection.provenance["selection_id"], r])
            for r in exclusions.to_dict("records")
        ]
    return {
        "checkpoints": selection.catalog.table,
        "performance": selection.performance,
        "rankings": rankings,
        "selection_pairs": selection.pairs,
        "exclusions": exclusions,
    }


def _compatible(a, b, ctx):
    if (a.parameter_space_id, a.evidence.initialization_id) != (
        b.parameter_space_id,
        b.evidence.initialization_id,
    ):
        raise ValueError("Direct comparison requires common coordinates and initialization")
    if not ctx.allow_unverified and (not a.evidence.provenance_id or not b.evidence.provenance_id):
        raise ValueError("Direct comparison requires explicit training provenance")
    if ctx.parameters.embeddings:
        if not a.evidence.tokenizer_id or a.evidence.tokenizer_id != b.evidence.tokenizer_id:
            raise ValueError(
                "Direct embedding comparison requires identical tokenizer/vocabulary evidence"
            )


def _special_pair(ctx, a, b, key, source, metrics, subject_id):
    ref, rid = ctx.reference(a) if source == "delta" else (None, None)
    if ref and ctx.reference(b)[1] != rid:
        raise ValueError("Updates use different reference origins")

    def compute(uid):
        with WeightReader(a.checkpoint.path) as ra, WeightReader(b.checkpoint.path) as rb:
            x, y = (
                ra.get(key).to(ctx.device, dtype=torch.float64),
                rb.get(key).to(ctx.device, dtype=torch.float64),
            )
            location = parameter_location(ra, key)
            if ref:
                with WeightReader(ref.path) as base:
                    ctx.aligned(ra, base)
                    ctx.aligned(rb, base)
                    initial = base.get(key).to(ctx.device, dtype=torch.float64)
                x, y = x - initial, y - initial
            values = {}
            overlaps = [m for m in metrics if "jaccard" in m or "sign" in m]
            if overlaps:
                result = topk_overlap(x, y, ctx.options.topk_fraction)
                values.update({m: result[m] for m in overlaps})
            subspaces = [m for m in metrics if m.startswith("subspace")]
            if subspaces:
                u, v = [
                    principal_basis(
                        t,
                        rank=ctx.options.subspace_rank,
                        max_exact_elements=ctx.options.max_svd_elements,
                    )
                    for t in (x, y)
                ]
                if not u.shape[1] or u.shape[1] != v.shape[1]:
                    values.update({m: float("nan") for m in subspaces})
                else:
                    angles = principal_angles(u, v)
                    all_values = {
                        "subspace_mean_cos2": float(angles.cos().square().mean()),
                        "subspace_chordal_normalized": float(angles.sin().square().mean().sqrt()),
                        "subspace_angle_mean_rad": float(angles.mean()),
                    }
                    values.update({m: all_values[m] for m in subspaces})
        approximate = (
            x.is_cuda
            and x.numel() > ctx.options.max_svd_elements
            and ctx.options.subspace_rank < min(x.shape)
        )
        return {
            "geometry_detailed": [
                observation(
                    uid,
                    {
                        **geometry_dimensions(a, b),
                        "execution_id": stable_id(ctx.definition),
                        **location,
                        "subject_id": subject_id,
                        "subject_kind": "pair",
                        "source": source,
                        "reference_id": rid,
                    },
                    m,
                    values[m],
                    method_status="topk_approx" if approximate else "exact",
                )
                for m in metrics
            ]
        }

    return ctx.unit(
        {
            "family": "special_pair",
            "pair": subject_id,
            "tensor": key,
            "source": source,
            "reference": rid,
            "metrics": sorted(metrics),
        },
        compute,
        stage="alignment",
        tensor=key,
    )[1]


def compare_selected(
    *,
    selection: SelectionResult,
    references: Mapping[str, ReferenceSpec] | None = None,
    parameters: ParameterSelection | None = None,
    metrics: Sequence[str] = ("l2_distance", "cosine"),
    drift_metrics: Sequence[str] = (),
    spectral_metrics: Sequence[str] = (),
    intrinsic_metrics: Sequence[str] = ("trace_distance_delta",),
    sources: Sequence[str] = ("delta",),
    options: MetricOptions | None = None,
    device: str | torch.device | None = None,
    pair_ids: Sequence[str] | None = None,
    on_error: str = "continue",
    reporter: RunReporter | None = None,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    block_elements: int = 100_000,
    allow_unverified_origin: bool = False,
    histogram_edges: Sequence[float] | None = None,
) -> ComparativeResult:
    """Execute frozen selected pairs, continuing independent units after recorded errors.

    No F1 or selection identity enters a geometry unit's cache key. Shift-invariant
    update distances are computed once from final weights and never duplicated by origin.
    """
    metrics, drift_metrics, spectral_metrics = (
        tuple(metrics),
        tuple(drift_metrics),
        tuple(spectral_metrics),
    )
    if (
        set(metrics) - set(PAIR_METRICS + SPECIAL_METRICS)
        or set(drift_metrics) - set(DRIFT_METRICS)
        or set(spectral_metrics) - set(SPECTRAL_METRICS)
    ):
        raise ValueError("Unknown selected geometric metric")
    if not sources or set(sources) - {"final", "delta"}:
        raise ValueError("Pair sources must select final and/or delta")
    if histogram_edges is not None and (
        len(histogram_edges) < 2
        or not np.isfinite(histogram_edges).all()
        or not np.all(np.diff(histogram_edges) > 0)
    ):
        raise ValueError("Histogram edges must be finite and strictly increasing")
    from ..execution.catalog import CROSS_METRICS

    if set(intrinsic_metrics) - set(CROSS_METRICS):
        raise ValueError("Unknown intrinsic comparison metric")
    reporter = reporter or RunReporter()
    tables = _base_tables(selection)
    collected = {}
    options = options or MetricOptions()
    ctx = reporter.attempt(
        "preflight",
        lambda: ComparativeContext(
            options=options,
            device=device,
            parameters=parameters or ParameterSelection(),
            references=references or {},
            reporter=reporter,
            on_error=on_error,
            cache_dir=cache_dir,
            resume=resume,
            block_elements=block_elements,
            allow_unverified=allow_unverified_origin,
        ),
        on_error=on_error,
    )
    if ctx is not None:
        # Union dependencies before fragmentation: one spectral decomposition per
        # endpoint/tensor/source even when descriptions and intrinsic pairs overlap.
        ctx.fingerprint_plan = {source: set(spectral_metrics) for source in sources}
        if not selection.pairs.empty and selection.pairs.domain.eq("intrinsic").any():
            for source, names in fingerprint_requirements(intrinsic_metrics).items():
                ctx.fingerprint_plan.setdefault(source, set()).update(names)
        entries = selection.catalog.by_id
        comparisons = []
        pairs = selection.pairs
        if pair_ids is not None and not pairs.empty:
            if set(pair_ids) - set(pairs.comparison_id):
                raise ValueError("pair_ids contains comparisons outside the frozen selection")
            pairs = pairs[pairs.comparison_id.isin(pair_ids)]
        for index, pair in enumerate(pairs.to_dict("records"), 1):
            reporter.emit(
                "progress",
                pair["comparison_id"],
                f"Pair {index}/{len(pairs)}",
                domain=pair["domain"],
            )
            a, b = entries[pair["artifact_a"]], entries[pair["artifact_b"]]
            cid = pair["comparison_id"]
            comparison = {
                key: pair[key] for key in ("comparison_id", "artifact_a", "artifact_b", "domain")
            }
            comparison["comparison_type"] = pair["domain"]
            for field in ("language", "entity", "backbone", "fold", "seed"):
                comparison[field] = (
                    pair[f"{field}_a"]
                    if pair[f"{field}_a"] == pair[f"{field}_b"]
                    else f"__cross_{field}__"
                )
            comparisons.append(comparison)

            def run_pair():
                ctx.verify(a)
                ctx.verify(b)
                if pair["domain"] == "intrinsic":
                    _copy_units(collected, intrinsic_geometry(ctx, a, b, cid, intrinsic_metrics))
                    return True
                _compatible(a, b, ctx)
                keys_a, keys_b = ctx.keys(a), ctx.keys(b)
                if set(keys_a) != set(keys_b):
                    raise ValueError("Pair selected parameter coordinates differ")
                requests = {}
                for source in sources:
                    invariant = tuple(sorted(set(metrics) & SHIFT_INVARIANT))
                    if invariant:
                        requests.setdefault("final", set()).update(invariant)
                    sensitive = tuple(sorted(set(metrics) - SHIFT_INVARIANT - set(SPECIAL_METRICS)))
                    if sensitive:
                        requests.setdefault(source, set()).update(sensitive)
                for key in keys_a:
                    for source, requested in sorted(requests.items()):
                        names = tuple(sorted(requested))

                        def vector():
                            return ctx.vector(
                                a,
                                b,
                                key,
                                source,
                                names,
                                subject_id=cid,
                                histogram_edges=histogram_edges,
                            )

                        result = reporter.attempt(
                            stable_id([cid, key, source, "prepare"]),
                            vector,
                            on_error=on_error,
                            stage="parameter",
                            tensor=key,
                        )
                        if result:
                            _copy_units(collected, result)
                    special = tuple(sorted(set(metrics) & set(SPECIAL_METRICS)))
                    if special:
                        for source in sources:
                            result = reporter.attempt(
                                stable_id([cid, key, source, "special"]),
                                lambda: _special_pair(ctx, a, b, key, source, special, cid),
                                on_error=on_error,
                            )
                            if result:
                                _copy_units(collected, result)
                return True

            reporter.attempt(cid, run_pair, on_error=on_error, stage="comparison")
            for entry in (a, b):

                def describe():
                    for key in ctx.keys(entry):
                        if drift_metrics:
                            result = reporter.attempt(
                                stable_id([entry.artifact_id, key, "drift"]),
                                lambda: ctx.vector(
                                    entry,
                                    None,
                                    key,
                                    "delta",
                                    drift_metrics,
                                    subject_id=entry.artifact_id,
                                    kind="drift",
                                    histogram_edges=histogram_edges,
                                ),
                                on_error=on_error,
                            )
                            if result:
                                _copy_units(collected, result)
                        with WeightReader(entry.checkpoint.path) as reader:
                            matrix_parameter = len(reader.shape_dtype(key)[0]) == 2
                        if spectral_metrics and matrix_parameter:
                            for source in sources:
                                result = reporter.attempt(
                                    stable_id([entry.artifact_id, key, source, "spectra"]),
                                    lambda: ctx.fingerprint(entry, key, source, spectral_metrics),
                                    on_error=on_error,
                                )
                                if result:
                                    _copy_units(collected, result)
                    return True

                if drift_metrics or spectral_metrics:
                    reporter.attempt(entry.artifact_id + ":describe", describe, on_error=on_error)
        tables["comparisons"] = pd.DataFrame(comparisons)
        tables["references"] = pd.DataFrame(ctx.reference_rows.values())
        tables.update({name: pd.DataFrame(rows) for name, rows in collected.items()})
    reporter.summary()
    tables["issues"] = issue_table(reporter)
    tables["events"] = event_table(reporter)
    for issue in selection.catalog.issues:
        row = {"issue_id": stable_id(issue), **issue}
        tables["issues"] = pd.concat([tables["issues"], pd.DataFrame([row])], ignore_index=True)
    result = ComparativeResult(
        tables,
        {
            "schema_version": "comparative-1",
            "definition": DEFINITION,
            "selection": selection.provenance,
            "parameters": asdict(parameters or ParameterSelection()),
            "computation": ctx.definition if ctx else None,
            "requested_metrics": {
                "pair": list(metrics),
                "drift": list(drift_metrics),
                "spectral": list(spectral_metrics),
                "intrinsic": list(intrinsic_metrics),
            },
            "sources": list(sources),
            "histogram_edges": list(histogram_edges) if histogram_edges is not None else None,
            "allow_unverified_origin": allow_unverified_origin,
            "requested_pairs": sorted(pair_ids) if pair_ids is not None else "all_selected",
            "complete": tables["issues"].empty,
            "input_locations": {
                e.artifact_id: str(e.checkpoint.path) for e in selection.catalog.entries
            },
        },
    )
    return merge_comparative([result])
