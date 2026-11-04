"""Execution boundaries: unused inputs, historical artifacts, and explicit devices."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from weight_analysis import AnalysisResult, CheckpointSpec, analyze
from weight_analysis.core import WeightReader
from weight_analysis.metrics import principal_basis


@pytest.fixture
def pair_study(tmp_path):
    key = "bert.encoder.layer.0.attention.self.query.weight"
    base = tmp_path / "base"
    base.mkdir()
    save_file({key: torch.eye(3)}, base / "model.safetensors")
    (base / "config.json").write_text(json.dumps({"model_type": "bert"}))
    checkpoints = []
    for language, multiplier in (("es", 2), ("en", 3)):
        path = tmp_path / language
        path.mkdir()
        save_file({key: torch.eye(3) * multiplier}, path / "model.safetensors")
        (path / "config.json").write_text(json.dumps({"model_type": "bert"}))
        checkpoints.append(CheckpointSpec("bert", language, "disease", path))
    return checkpoints, {"bert": base}


def test_final_comparison_never_materializes_original(pair_study, monkeypatch):
    checkpoints, originals = pair_study
    get = WeightReader.get

    def final_only(reader, key):
        if reader.path == originals["bert"]:
            raise AssertionError("Final-only comparison must not load original tensors")
        return get(reader, key)

    monkeypatch.setattr(WeightReader, "get", final_only)
    result = analyze(
        analysis="cross_language",
        checkpoints=checkpoints,
        originals=originals,
        metrics=["cosine"],
        sources=["final"],
    )
    assert result.data.cosine.tolist() == pytest.approx([1.0])


def test_pair_selection_excludes_unmounted_checkpoint_before_fingerprinting(pair_study, tmp_path):
    checkpoints, originals = pair_study
    unused = CheckpointSpec("bert", "fr", "disease", tmp_path / "not-mounted")
    result = analyze(
        analysis="cross_language",
        checkpoints=[*checkpoints, unused],
        originals=originals,
        metrics=["cosine"],
        pairs=[(checkpoints[0].uid, checkpoints[1].uid)],
    )
    assert len(result.data) == 1
    assert len(result.provenance["inputs"]) == 2


def test_historical_centrality_does_not_revalidate_live_checkpoints(monkeypatch):
    import weight_analysis.execution.intermediates as intermediates

    def unavailable(*args, **kwargs):
        raise AssertionError("Historical pair centrality must not read original weights")

    monkeypatch.setattr(intermediates, "checkpoint_fingerprint", unavailable)
    table = pd.DataFrame(
        [
            {
                "backbone_a": "a",
                "backbone_b": "b",
                "language": "es",
                "entity": "disease",
                "trace_distance_delta": 0.5,
            }
        ]
    )
    historical = AnalysisResult(
        "cross_backbone",
        table,
        {
            "inputs": [{"checkpoint": "old", "backbone": "a", "path": "/not-mounted"}],
            "originals": {"a": {"path": "/not-mounted", "files": {}}},
        },
    )
    result = analyze(analysis="model_centrality", pair_results=historical)
    assert result.data.mean_distance.tolist() == [0.5, 0.5]
    assert result.provenance["intermediates"]["pairs"]["provenance"] == historical.provenance


def test_explicit_cuda_subspace_request_never_falls_back():
    if torch.cuda.is_available():
        pytest.skip("This regression targets CPU-only environments")
    with pytest.raises(RuntimeError, match="CUDA"):
        principal_basis(torch.eye(3), rank=1, device="cuda")
