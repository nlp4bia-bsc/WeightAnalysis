"""Public orchestration of selective checkpoint analyses with explicit persistence."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .configuration import MetricOptions
from .core import CheckpointSpec
from .execution.catalog import (
    ANALYSES,
    CROSS_METRICS,
    FACTORIAL_METRICS,
    OVERLAP_METRICS,
    SUBSPACE_METRICS,
)
from .execution.comparisons import _cross_backbone, _related
from .execution.descriptions import _base, _describe
from .execution.factorial import _factorial
from .execution.intermediates import _frame, _intermediate_lineage, _select_distances, _select_table
from .execution.selection import _names, _pairs
from .metrics import DRIFT_METRICS, PAIR_METRICS, SPECTRAL_METRICS
from .provenance import canonical_json, computation_provenance
from .results import AnalysisResult, read_result, write_result


def analyze(
    *,
    analysis: str,
    checkpoints: Sequence[CheckpointSpec] = (),
    originals: Mapping[str, str | Path] | None = None,
    layers: Sequence[int] | None = None,
    components: Sequence[str] | None = None,
    tensors: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    backbones: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    entities: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
    seeds: Sequence[int | None] | None = None,
    checkpoint_ids: Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    options: MetricOptions | None = None,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    spectral_results: pd.DataFrame | AnalysisResult | None = None,
    drift_results: pd.DataFrame | AnalysisResult | None = None,
    pair_results: pd.DataFrame | AnalysisResult | None = None,
) -> AnalysisResult:
    """Compute one analysis, selecting parameters before materializing their tensors.

    Results stay in memory unless cache_dir is explicitly supplied. Resume verifies input
    content hashes and configuration before using a committed result. Fold and seed strata
    are never silently pooled. Legacy aggregate tables remain available through analyses
    and comparisons; this selective interface emits parameter-level observations.
    GPU/CUDA is the default. Pass device="cpu" on hosts without CUDA. The direct
    argument overrides options.device; unavailability never triggers CPU fallback.
    """
    if analysis not in ANALYSES:
        raise ValueError(f"Unknown analysis {analysis!r}; choose one of {ANALYSES}")
    options = options or MetricOptions()
    explicit_sources = sources
    layers, components, tensors = _names(layers), _names(components), _names(tensors)
    filters = {
        "backbone": _names(backbones),
        "language": _names(languages),
        "entity": _names(entities),
        "fold": _names(folds),
        "seed": _names(seeds),
        "checkpoint_id": _names(checkpoint_ids),
    }
    if not all(isinstance(checkpoint, CheckpointSpec) for checkpoint in checkpoints):
        raise TypeError("checkpoints must contain explicit CheckpointSpec objects")
    chosen = sorted(
        (
            checkpoint
            for checkpoint in checkpoints
            if all(
                values is None or getattr(checkpoint, key) in values
                for key, values in filters.items()
            )
        ),
        key=lambda checkpoint: checkpoint.uid,
    )
    if len({checkpoint.uid for checkpoint in chosen}) != len(chosen):
        raise ValueError("Duplicate checkpoint declarations")
    if pairs is not None and analysis in (
        "cross_language",
        "cross_entity",
        "language_entity",
        "cross_backbone",
    ):
        endpoints = {
            checkpoint.uid: checkpoint
            for pair in _pairs(chosen, analysis, pairs)
            for checkpoint in pair
        }
        chosen = sorted(endpoints.values(), key=lambda checkpoint: checkpoint.uid)
    defaults = {
        "pretraining_drift": ("relative_frobenius",),
        "spectral_fingerprints": ("matrix_fro", "effective_rank"),
        "cross_backbone": ("trace_distance_delta",),
        "language_entity_decomposition": FACTORIAL_METRICS,
        "model_centrality": ("trace_distance_delta",),
    }
    metrics = _names(metrics if metrics is not None else defaults.get(analysis, ("cosine",)))
    supported = (
        DRIFT_METRICS
        if analysis == "pretraining_drift"
        else SPECTRAL_METRICS
        if analysis == "spectral_fingerprints"
        else CROSS_METRICS
        if analysis == "cross_backbone"
        else tuple(name for name in CROSS_METRICS if "similarity" not in name)
        if analysis == "model_centrality"
        else FACTORIAL_METRICS
        if analysis == "language_entity_decomposition"
        else PAIR_METRICS + OVERLAP_METRICS + SUBSPACE_METRICS
    )
    if not metrics or set(metrics) - set(supported):
        raise ValueError(f"metrics must select nonempty supported names: {supported}")
    sources = _names(
        sources
        if sources is not None
        else ("base", "final", "delta")
        if analysis == "spectral_fingerprints"
        else ("delta",)
    )
    allowed_sources = (
        {"base", "final", "delta"} if analysis == "spectral_fingerprints" else {"delta", "final"}
    )
    if not sources or set(sources) - allowed_sources:
        raise ValueError(f"sources must select from {sorted(allowed_sources)}")
    if analysis not in (
        "spectral_fingerprints",
        "cross_language",
        "cross_entity",
        "language_entity",
    ):
        if explicit_sources is not None:
            raise ValueError("sources only applies to spectral fingerprints and direct comparisons")
    if pairs is not None and analysis not in (
        "cross_language",
        "cross_entity",
        "language_entity",
        "cross_backbone",
    ):
        raise ValueError("pairs only applies to pairwise comparisons")
    if analysis != "model_centrality" and not chosen:
        raise ValueError("No checkpoints match the requested selection")
    from .devices import resolve_device

    requested_device = device if device is not None else options.device
    resolved_device = resolve_device(requested_device)
    options = replace(options, device=str(resolved_device))
    base_paths = {
        key: Path(value).expanduser().resolve()
        for key, value in (originals or {}).items()
        if key in {checkpoint.backbone for checkpoint in chosen}
    }
    spectral_df, drift_df, pair_df = map(_frame, (spectral_results, drift_results, pair_results))
    if analysis == "cross_backbone":
        spectral_df = _select_table(
            spectral_df, checkpoints=chosen, layers=layers, components=components, tensors=tensors
        )
        drift_df = _select_table(
            drift_df, checkpoints=chosen, layers=layers, components=components, tensors=tensors
        )
    elif spectral_df is not None or drift_df is not None:
        raise ValueError("spectral_results/drift_results only apply to cross_backbone")
    if analysis == "model_centrality":
        pair_df = _select_distances(
            pair_df,
            filters,
            layers=layers,
            components=components,
            tensors=tensors,
            checkpoint_ids=checkpoint_ids,
        )
    elif pair_df is not None:
        raise ValueError("pair_results only applies to model_centrality")
    configuration = {
        "analysis": analysis,
        "metrics": metrics,
        "sources": sources,
        "options": asdict(options),
    }
    selected_ids = {checkpoint.uid for checkpoint in chosen}
    intermediates = {}
    for name, value, expected in (
        ("spectral", spectral_results, "spectral_fingerprints"),
        ("drift", drift_results, "pretraining_drift"),
        ("pairs", pair_results, "cross_backbone"),
    ):
        lineage = _intermediate_lineage(
            value, expected_analysis=expected, selected_ids=selected_ids, originals=base_paths
        )
        if lineage is not None:
            intermediates[name] = lineage
    provenance = computation_provenance(
        configuration=configuration,
        checkpoints=chosen,
        originals=base_paths,
        tables={"spectral": spectral_df, "drift": drift_df, "pairs": pair_df},
        selection={"layers": layers, "components": components, "tensors": tensors, **filters},
        intermediates=intermediates,
    )
    if pairs is not None:
        import hashlib

        provenance["pairs"] = sorted({tuple(sorted(pair)) for pair in pairs})
        provenance["unit_id"] = hashlib.sha256(canonical_json(provenance).encode()).hexdigest()
    cached_path = (
        Path(cache_dir) / (provenance["unit_id"] + ".parquet") if cache_dir is not None else None
    )
    if cached_path is not None and resume and cached_path.is_file():
        cached = read_result(cached_path)
        if (
            cached.unit_id != provenance["unit_id"]
            or cached.analysis != analysis
            or canonical_json(cached.provenance) != canonical_json(provenance)
        ):
            raise ValueError("Cached result does not match the requested computation")
        return cached
    if analysis in ("pretraining_drift", "spectral_fingerprints"):
        if analysis == "pretraining_drift" or set(sources) & {"base", "delta"}:
            for checkpoint in chosen:
                _base(checkpoint, base_paths)
        drift_df, spectral_df = _describe(
            chosen,
            base_paths,
            layers=layers,
            components=components,
            tensors=tensors,
            drift_names=metrics if analysis == "pretraining_drift" else (),
            spectral_names=metrics if analysis == "spectral_fingerprints" else (),
            sources=sources if analysis == "spectral_fingerprints" else (),
            options=options,
        )
        data_df = drift_df if analysis == "pretraining_drift" else spectral_df
    elif analysis in ("cross_language", "cross_entity", "language_entity"):
        data_df = _related(
            _pairs(chosen, analysis, pairs),
            base_paths,
            layers=layers,
            components=components,
            tensors=tensors,
            metrics=metrics,
            sources=sources,
            options=options,
        )
    elif analysis == "cross_backbone":
        data_df = _cross_backbone(
            chosen,
            base_paths,
            metrics=metrics,
            pairs=pairs,
            layers=layers,
            components=components,
            tensors=tensors,
            options=options,
            spectral_df=spectral_df,
            drift_df=drift_df,
        )
    elif analysis == "language_entity_decomposition":
        data_df = _factorial(
            chosen,
            base_paths,
            layers=layers,
            components=components,
            tensors=tensors,
            metrics=metrics,
            options=options,
        )
    else:
        from .comparisons import model_centrality

        if pair_df is None:
            raise ValueError("model_centrality requires explicit pair_results")
        data_df = model_centrality(pair_df, metrics=metrics)
    result = AnalysisResult(analysis=analysis, data=data_df, provenance=provenance)
    if cached_path is not None:
        write_result(result, cached_path)
    return result
