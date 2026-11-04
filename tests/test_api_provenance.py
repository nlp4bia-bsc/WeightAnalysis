"""Regression contracts for selective API provenance and reusable intermediates."""

from __future__ import annotations

import importlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file, save_file

from weight_analysis import CheckpointSpec, analyze, merge_results
from weight_analysis.configuration import MetricOptions
from weight_analysis.provenance import computation_provenance
from weight_analysis.results import AnalysisResult, write_result
from weight_analysis.results.models import data_digest


@pytest.fixture
def provenance_study(tmp_path):
    checkpoints = []
    originals = {}
    config = {"model_type": "bert", "hidden_size": 2, "num_hidden_layers": 3}
    for backbone, profile in (("a", [1.0, 2.0, 3.0]), ("b", [1.0, 3.0, 2.0])):
        base = tmp_path / "models" / backbone
        final = tmp_path / "weights" / backbone
        base.mkdir(parents=True)
        final.mkdir(parents=True)
        original = {
            f"bert.encoder.layer.{layer}.attention.self.query.weight": torch.eye(2) * (layer + 1)
            for layer in range(3)
        }
        updated = {
            f"bert.encoder.layer.{layer}.attention.self.query.weight": torch.eye(2) * magnitude
            for layer, magnitude in enumerate(profile)
        }
        save_file(original, base / "model.safetensors")
        save_file(updated, final / "model.safetensors")
        for directory in (base, final):
            (directory / "config.json").write_text(json.dumps(config))
        originals[backbone] = base
        checkpoints.append(CheckpointSpec(backbone, "es", "disease", final))
    return checkpoints, originals


def pair_table():
    return pd.DataFrame(
        [
            {
                "backbone_a": "a",
                "backbone_b": "b",
                "language": "es",
                "entity": "disease",
                "fold": 1,
                "seed": 13,
                "trace_distance_delta": 0.25,
            },
            {
                "backbone_a": "a",
                "backbone_b": "b",
                "language": "it",
                "entity": "disease",
                "fold": 2,
                "seed": 42,
                "trace_distance_delta": 0.75,
            },
        ]
    )


def test_table_fingerprint_distinguishes_adjacent_float64_values():
    arguments = dict(configuration={"analysis": "test"}, checkpoints=[], originals={})
    left = pd.DataFrame({"value": [1.0]})
    right = pd.DataFrame({"value": [np.nextafter(1.0, 2.0)]})
    a = computation_provenance(tables={"pairs": left}, **arguments)
    b = computation_provenance(tables={"pairs": right}, **arguments)
    assert a["unit_id"] != b["unit_id"], "Float64 changes must invalidate cached computations"


def test_table_fingerprint_is_independent_of_observation_and_column_order():
    arguments = dict(configuration={"analysis": "test"}, checkpoints=[], originals={})
    table = pair_table()
    reversed_table = table.iloc[::-1].loc[:, list(reversed(table.columns))]
    a = computation_provenance(tables={"pairs": table}, **arguments)
    b = computation_provenance(tables={"pairs": reversed_table}, **arguments)
    assert a["unit_id"] == b["unit_id"]


def test_numerical_definition_version_invalidates_unit(provenance_study, monkeypatch):
    checkpoints, originals = provenance_study
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["delta_l2"],
    )
    first = analyze(device="cpu", **arguments)
    provenance_module = importlib.import_module("weight_analysis.provenance")
    monkeypatch.setattr(provenance_module, "__version__", "999.0.0-test")
    refreshed = analyze(device="cpu", **arguments)
    assert first.unit_id != refreshed.unit_id


def test_modified_original_content_invalidates_cache(provenance_study, tmp_path):
    checkpoints, originals = provenance_study
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["delta_l2"],
        cache_dir=tmp_path / "cache",
    )
    first = analyze(device="cpu", **arguments)
    path = originals["a"] / "model.safetensors"
    tensors = load_file(path)
    tensors[next(iter(tensors))] += 0.125
    save_file(tensors, path)
    refreshed = analyze(device="cpu", **arguments)
    assert first.unit_id != refreshed.unit_id
    assert refreshed.data.delta_l2.sum() > first.data.delta_l2.sum()


def test_model_configuration_invalidates_unit(provenance_study):
    checkpoints, originals = provenance_study
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["delta_l2"],
    )
    first = analyze(device="cpu", **arguments)
    path = checkpoints[0].path / "config.json"
    config = json.loads(path.read_text())
    config["architectures"] = ["BertForTokenClassification"]
    path.write_text(json.dumps(config))
    assert analyze(device="cpu", **arguments).unit_id != first.unit_id


def test_interrupted_cache_write_can_be_retried(provenance_study, tmp_path, monkeypatch):
    import weight_analysis.results.storage as storage

    checkpoints, originals = provenance_study
    cache_dir = tmp_path / "cache"
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["delta_l2"],
        cache_dir=cache_dir,
    )
    original_replace = storage.os.replace

    def interrupt(*args):
        raise OSError("simulated interrupted cache commit")

    monkeypatch.setattr(storage.os, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted cache commit"):
        analyze(device="cpu", **arguments)
    assert not list(cache_dir.iterdir())
    monkeypatch.setattr(storage.os, "replace", original_replace)
    completed = analyze(device="cpu", **arguments)
    assert (cache_dir / f"{completed.unit_id}.parquet").is_file()
    assert analyze(device="cpu", **arguments).unit_id == completed.unit_id


def test_corrupt_committed_cache_is_rejected(provenance_study, tmp_path):
    checkpoints, originals = provenance_study
    cache_dir = tmp_path / "cache"
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["delta_l2"],
        cache_dir=cache_dir,
    )
    completed = analyze(device="cpu", **arguments)
    path = cache_dir / f"{completed.unit_id}.parquet"
    path.write_bytes(b"interrupted or corrupt parquet")
    with pytest.raises((ValueError, OSError)):
        analyze(device="cpu", **arguments)


def test_explicit_pair_order_does_not_change_identity(provenance_study):
    checkpoints, _ = provenance_study
    pair = tuple(checkpoint.uid for checkpoint in checkpoints)
    arguments = dict(
        analysis="cross_backbone", checkpoints=checkpoints, metrics=["trace_distance_final"]
    )
    forward = analyze(device="cpu", pairs=[pair], **arguments)
    reverse = analyze(device="cpu", pairs=[tuple(reversed(pair))], **arguments)
    assert forward.unit_id == reverse.unit_id
    pd.testing.assert_frame_equal(forward.data, reverse.data)


def test_precomputed_spectra_respect_layer_selection(provenance_study):
    checkpoints, _ = provenance_study
    full = analyze(
        device="cpu",
        analysis="spectral_fingerprints",
        checkpoints=checkpoints,
        metrics=["matrix_fro"],
        sources=["final"],
    )
    arguments = dict(
        analysis="cross_backbone",
        checkpoints=checkpoints,
        metrics=["trace_distance_final"],
        layers=[0, 1],
    )
    direct = analyze(device="cpu", **arguments)
    precomputed = analyze(device="cpu", spectral_results=full, **arguments)
    assert direct.data.iloc[0].trace_distance_final == pytest.approx(0.0, abs=1e-12)
    assert precomputed.data.iloc[0].trace_distance_final == pytest.approx(
        direct.data.iloc[0].trace_distance_final, abs=1e-12
    )


def test_centrality_filters_explicit_intermediate_table_by_study_dimensions():
    arguments = dict(
        analysis="model_centrality", pair_results=pair_table(), metrics=["trace_distance_delta"]
    )
    spanish = analyze(device="cpu", languages=["es"], folds=[1], seeds=[13], **arguments)
    italian = analyze(device="cpu", languages=["it"], folds=[2], seeds=[42], **arguments)
    assert set(spanish.data.language) == {"es"}
    assert set(italian.data.language) == {"it"}
    assert spanish.unit_id != italian.unit_id
    assert spanish.data.mean_distance.tolist() == [0.25, 0.25]
    assert italian.data.mean_distance.tolist() == [0.75, 0.75]


def test_precomputed_results_reject_changed_checkpoint_content(provenance_study):
    checkpoints, _ = provenance_study
    spectra = analyze(
        device="cpu",
        analysis="spectral_fingerprints",
        checkpoints=checkpoints,
        metrics=["matrix_fro"],
        sources=["final"],
    )
    path = checkpoints[1].path / "model.safetensors"
    tensors = load_file(path)
    tensors["bert.encoder.layer.1.attention.self.query.weight"] *= 2
    save_file(tensors, path)
    with pytest.raises(ValueError, match="(?i)(stale|provenance|fingerprint|content)"):
        analyze(
            device="cpu",
            analysis="cross_backbone",
            checkpoints=checkpoints,
            metrics=["trace_distance_final"],
            spectral_results=spectra,
        )


@pytest.mark.parametrize("mismatch", ["analysis", "configuration"])
def test_cache_metadata_must_match_requested_computation(provenance_study, tmp_path, mismatch):
    checkpoints, originals = provenance_study
    cache_dir = tmp_path / "cache"
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["delta_l2"],
        cache_dir=cache_dir,
    )
    completed = analyze(device="cpu", **arguments)
    analysis = "other_analysis" if mismatch == "analysis" else completed.analysis
    provenance = dict(completed.provenance)
    if mismatch == "configuration":
        provenance["configuration"] = {"analysis": "different", "metrics": ["cosine"]}
    altered = AnalysisResult(analysis, completed.data, provenance)
    write_result(altered, cache_dir / f"{completed.unit_id}.parquet")
    with pytest.raises(ValueError, match="(?i)(cache|match|provenance)"):
        analyze(device="cpu", **arguments)


def test_independent_layer_shards_preserve_definition_and_merge_joint_result(provenance_study):
    checkpoints, originals = provenance_study
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints,
        originals=originals,
        metrics=["delta_l2"],
    )
    combined = analyze(device="cpu", **arguments)
    first = analyze(device="cpu", layers=[0], **arguments)
    remaining = analyze(device="cpu", layers=[1, 2], **arguments)
    assert first.provenance["configuration"] == remaining.provenance["configuration"]
    assert first.provenance["selection"] != remaining.provenance["selection"]
    assert first.unit_id != remaining.unit_id
    merged = merge_results([remaining, first], observation_keys=["checkpoint", "tensor", "level"])
    assert data_digest(merged.data) == data_digest(combined.data)


def test_overlapping_layer_shards_can_be_detected_before_aggregation(provenance_study):
    checkpoints, originals = provenance_study
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints,
        originals=originals,
        metrics=["delta_l2"],
    )
    first = analyze(device="cpu", layers=[0, 1], **arguments)
    overlap = analyze(device="cpu", layers=[1, 2], **arguments)
    with pytest.raises(ValueError, match="Overlapping observations"):
        merge_results([first, overlap], observation_keys=["checkpoint", "tensor", "level"])


def alternate_originals(originals, tmp_path, *, change_content):
    replacement = tmp_path / "replacement-original"
    shutil.copytree(originals["a"], replacement)
    if change_content:
        weights_path = replacement / "model.safetensors"
        weights = load_file(weights_path)
        weights[next(iter(weights))] += 0.25
        save_file(weights, weights_path)
    return {**originals, "a": replacement}


@pytest.mark.parametrize("source", ["base", "delta"])
def test_precomputed_spectra_reject_conflicting_declared_original(
    provenance_study, tmp_path, source
):
    checkpoints, originals = provenance_study
    spectra = analyze(
        device="cpu",
        analysis="spectral_fingerprints",
        checkpoints=checkpoints,
        originals=originals,
        metrics=["matrix_fro"],
        sources=[source],
    )
    replacements = alternate_originals(originals, tmp_path, change_content=True)
    with pytest.raises(ValueError, match="(?i)(original|provenance|fingerprint|content)"):
        analyze(
            device="cpu",
            analysis="cross_backbone",
            checkpoints=checkpoints,
            originals=replacements,
            metrics=[f"trace_distance_{source}"],
            spectral_results=spectra,
        )


def test_precomputed_drift_rejects_conflicting_declared_original(provenance_study, tmp_path):
    checkpoints, originals = provenance_study
    drift = analyze(
        device="cpu",
        analysis="pretraining_drift",
        checkpoints=checkpoints,
        originals=originals,
        metrics=["relative_l2"],
    )
    replacements = alternate_originals(originals, tmp_path, change_content=True)
    with pytest.raises(ValueError, match="(?i)(original|provenance|fingerprint|content)"):
        analyze(
            device="cpu",
            analysis="cross_backbone",
            checkpoints=checkpoints,
            originals=replacements,
            metrics=["adaptation_profile_l1"],
            drift_results=drift,
        )


@pytest.mark.parametrize("source", ["base", "delta"])
def test_precomputed_spectra_allow_relocated_identical_original(provenance_study, tmp_path, source):
    checkpoints, originals = provenance_study
    spectra = analyze(
        device="cpu",
        analysis="spectral_fingerprints",
        checkpoints=checkpoints,
        originals=originals,
        metrics=["matrix_fro"],
        sources=[source],
    )
    replacements = alternate_originals(originals, tmp_path, change_content=False)
    arguments = dict(
        analysis="cross_backbone",
        checkpoints=checkpoints,
        originals=replacements,
        metrics=[f"trace_distance_{source}"],
        options=MetricOptions(min_fingerprint_components=1),
    )
    reused = analyze(device="cpu", spectral_results=spectra, **arguments)
    direct = analyze(device="cpu", **arguments)
    pd.testing.assert_frame_equal(reused.data, direct.data)
