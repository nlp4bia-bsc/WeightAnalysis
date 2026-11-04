"""Cross-analysis dependency planning and strict intermediate scientific contracts."""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from weight_analysis import CheckpointSpec, analyze
from weight_analysis.configuration import MetricOptions
from weight_analysis.core import WeightReader


@pytest.fixture
def selective_study(tmp_path):
    checkpoints, originals = [], {}
    config = {"model_type": "bert", "hidden_size": 2, "num_hidden_layers": 3}
    for backbone in ("a", "b", "unused"):
        base_path = tmp_path / "models" / backbone
        base_path.mkdir(parents=True)
        tensors = {
            f"bert.encoder.layer.{layer}.attention.self.query.weight": torch.eye(2) * (layer + 1)
            for layer in range(3)
        }
        save_file(tensors, base_path / "model.safetensors")
        (base_path / "config.json").write_text(json.dumps(config))
        originals[backbone] = base_path
        for language in ("es", "it"):
            for entity in ("disease", "symptom"):
                final_path = tmp_path / backbone / language / entity
                final_path.mkdir(parents=True)
                update = (0.1 if language == "es" else 0.3) * (1 if entity == "disease" else -1)
                save_file(
                    {name: value + update for name, value in tensors.items()},
                    final_path / "model.safetensors",
                )
                (final_path / "config.json").write_text(json.dumps(config))
                checkpoints.append(CheckpointSpec(backbone, language, entity, final_path))
    return checkpoints, originals


def selected_pair(checkpoints):
    return [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.backbone in ("a", "b")
        and checkpoint.language == "es"
        and checkpoint.entity == "disease"
    ]


def test_cross_backbone_only_reads_explicit_pair_endpoints(selective_study, monkeypatch):
    checkpoints, originals = selective_study
    selected = selected_pair(checkpoints)
    allowed = {checkpoint.path for checkpoint in selected} | {
        originals[checkpoint.backbone] for checkpoint in selected
    }
    enter = WeightReader.__enter__
    observed = []

    def record_enter(reader):
        observed.append(reader.path)
        assert reader.path in allowed, "Unrequested pair endpoint was opened"
        return enter(reader)

    monkeypatch.setattr(WeightReader, "__enter__", record_enter)
    result = analyze(
        device="cpu",
        analysis="cross_backbone",
        checkpoints=checkpoints,
        originals=originals,
        pairs=[tuple(checkpoint.uid for checkpoint in selected)],
        metrics=["trace_distance_delta"],
    )
    assert len(result.data) == 1
    assert observed


def test_cross_backbone_does_not_cross_product_metric_source_dependencies(
    selective_study, monkeypatch
):
    api = importlib.import_module("weight_analysis.api")
    checkpoints, originals = selective_study
    selected = selected_pair(checkpoints)
    namespace = api._cross_backbone.__globals__
    original_describe = namespace["_describe"]
    observed = []

    def record_describe(*args, **kwargs):
        observed.append((tuple(kwargs["sources"]), tuple(kwargs["spectral_names"])))
        return original_describe(*args, **kwargs)

    monkeypatch.setitem(namespace, "_describe", record_describe)

    def forbidden_svd(*args, **kwargs):
        raise AssertionError("Frobenius and quantile dependencies require no SVD")

    monkeypatch.setattr(torch.linalg, "svdvals", forbidden_svd)
    result = analyze(
        device="cpu",
        analysis="cross_backbone",
        checkpoints=selected,
        originals=originals,
        metrics=["trace_distance_delta", "quantile_w1_final"],
        options=MetricOptions(min_fingerprint_components=1),
    )
    assert len(result.data) == 1
    assert observed == [
        (("delta",), ("matrix_fro",)),
        (("final",), ("normalized_weight_quantiles",)),
    ]


def test_factorial_checks_configuration_of_every_cell(selective_study):
    checkpoints, originals = selective_study
    group = sorted(
        (checkpoint for checkpoint in checkpoints if checkpoint.backbone == "a"),
        key=lambda checkpoint: checkpoint.uid,
    )
    path = group[-1].path / "config.json"
    config = json.loads(path.read_text())
    config["num_hidden_layers"] = 99
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="num_hidden_layers mismatch"):
        analyze(
            device="cpu",
            analysis="language_entity_decomposition",
            checkpoints=group,
            originals=originals,
        )


@pytest.mark.parametrize("invalid", ["missing_column", "duplicate", "missing_source", "negative"])
def test_cross_backbone_rejects_invalid_spectral_intermediates(selective_study, invalid):
    checkpoints, originals = selective_study
    selected = selected_pair(checkpoints)
    table = analyze(
        device="cpu",
        analysis="spectral_fingerprints",
        checkpoints=selected,
        metrics=["matrix_fro"],
        sources=["final"],
    ).data
    if invalid == "missing_column":
        table = table.drop(columns="tensor")
        match = "missing columns"
    elif invalid == "duplicate":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
        match = "duplicate"
    elif invalid == "missing_source":
        table["source"] = "base"
        match = "missing selected checkpoint/source"
    else:
        table.loc[0, "matrix_fro"] = -1
        match = "nonnegative"
    with pytest.raises(ValueError, match=match):
        analyze(
            device="cpu",
            analysis="cross_backbone",
            checkpoints=selected,
            originals=originals,
            metrics=["trace_distance_final"],
            spectral_results=table,
        )


def test_cross_backbone_rejects_invalid_quantile_profile(selective_study):
    checkpoints, _ = selective_study
    selected = selected_pair(checkpoints)
    table = analyze(
        device="cpu",
        analysis="spectral_fingerprints",
        checkpoints=selected,
        metrics=["normalized_weight_quantiles"],
        sources=["final"],
    ).data
    table.at[0, "normalized_weight_quantiles"] = [2.0, 1.0]
    with pytest.raises(ValueError, match="nondecreasing"):
        analyze(
            device="cpu",
            analysis="cross_backbone",
            checkpoints=selected,
            metrics=["quantile_w1_final"],
            spectral_results=table,
        )
