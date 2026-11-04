"""Selective descriptions for independent scientific computations."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..configuration import MetricOptions
from ..core import (
    SPECTRAL_COMPONENTS,
    CheckpointSpec,
    WeightReader,
    audit_alignment,
    config_metadata,
    get_delta,
    identify,
    selected_keys,
)
from ..metrics import drift_metrics, spectral_metrics


def _base(checkpoint: CheckpointSpec, originals: Mapping[str, Path]) -> Path:
    if checkpoint.backbone not in originals:
        raise ValueError(f"Pass originals[{checkpoint.backbone!r}] for this analysis")
    path = originals[checkpoint.backbone]
    final_config = config_metadata(checkpoint.path, strict=True)
    base_config = config_metadata(path, strict=True)
    for name in (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    ):
        if (
            final_config.get(name) is not None
            and base_config.get(name) is not None
            and final_config[name] != base_config[name]
        ):
            raise ValueError(f"Original/checkpoint {name} mismatch for {checkpoint.uid}")
    return path


def _identity(checkpoint: CheckpointSpec) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint.uid,
        "checkpoint_id": checkpoint.checkpoint_id,
        "backbone": checkpoint.backbone,
        "language": checkpoint.language,
        "entity": checkpoint.entity,
        "fold": checkpoint.fold,
        "seed": checkpoint.seed,
        "trained_path": str(checkpoint.path),
    }


def _describe(
    checkpoints: Sequence[CheckpointSpec],
    originals: dict[str, Path],
    *,
    layers: Sequence[int] | None,
    components: Sequence[str] | None,
    tensors: Sequence[str] | None,
    drift_names: Sequence[str],
    spectral_names: Sequence[str],
    sources: Sequence[str],
    options: MetricOptions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drift_rows, spectral_rows = [], []
    base_cache: dict[tuple[str, str], dict[str, Any]] = {}
    with ExitStack() as stack:
        bases = {name: stack.enter_context(WeightReader(path)) for name, path in originals.items()}
        for checkpoint in checkpoints:
            if checkpoint.backbone in originals:
                _base(checkpoint, originals)
            with WeightReader(checkpoint.path) as reader:
                original = bases.get(checkpoint.backbone)
                if original is not None:
                    aligned, status = audit_alignment(
                        reader, original, options.min_aligned_fraction
                    )
                else:
                    aligned = selected_keys(reader)
                    status = {"element_coverage": 1.0}
                keys = sorted(
                    set(aligned)
                    & set(
                        selected_keys(reader, layers=layers, components=components, tensors=tensors)
                    )
                )
                depths = [identify(key, reader.shape_dtype(key)[0])[0] for key in aligned]
                maximum = max((layer for layer in depths if layer is not None), default=0)
                for key in keys:
                    layer, component, _ = identify(key, reader.shape_dtype(key)[0])
                    location = {
                        **_identity(checkpoint),
                        "tensor": key,
                        "layer": layer,
                        "component": component,
                        "level": "tensor",
                        "element_coverage": status["element_coverage"],
                    }
                    if drift_names or "delta" in sources:
                        if original is None:
                            raise ValueError("Drift/delta requires originals")
                        final, initial, delta = get_delta(reader, original, key)
                    else:
                        final = reader.get(key) if "final" in sources else None
                        initial = (
                            original.get(key)
                            if original is not None and "base" in sources
                            else None
                        )
                        delta = None
                    if drift_names:
                        drift_rows.append(
                            {**location, **drift_metrics(final, initial, metrics=drift_names)}
                        )
                    if spectral_names and component in SPECTRAL_COMPONENTS:
                        weights = {"final": final, "base": initial, "delta": delta}
                        for source in sources:
                            matrix = weights[source]
                            if matrix is None:
                                raise ValueError(f"Source {source!r} requires originals")
                            if matrix.ndim != 2:
                                continue
                            cache_key = (str(originals.get(checkpoint.backbone)), key)
                            if source == "base" and cache_key in base_cache:
                                values = base_cache[cache_key]
                            else:
                                values = spectral_metrics(
                                    matrix,
                                    metrics=spectral_names,
                                    mode=options.spectral_mode,
                                    max_exact_elements=options.max_svd_elements,
                                    topk=options.topk,
                                    device=options.device,
                                )
                                if source == "base":
                                    base_cache[cache_key] = values
                            spectral_rows.append(
                                {
                                    **location,
                                    "source": source,
                                    "relative_depth": layer / maximum
                                    if layer is not None and maximum
                                    else 0.0,
                                    **values,
                                }
                            )
    drift_columns = ["checkpoint", "level", "component", "layer", "relative_l2"]
    spectra_columns = ["checkpoint", "level", "component", "layer", "source", "status"]
    return (
        pd.DataFrame(drift_rows) if drift_rows else pd.DataFrame(columns=drift_columns),
        pd.DataFrame(spectral_rows) if spectral_rows else pd.DataFrame(columns=spectra_columns),
    )
