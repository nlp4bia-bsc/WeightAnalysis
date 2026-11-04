"""Independent scientific execution, selection, resumption, and unit assembly."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file, save_file

from weight_analysis import CheckpointSpec, analyze, merge_results
from weight_analysis.core import WeightReader

QUERY = "bert.encoder.layer.0.attention.self.query.weight"
OTHER_QUERY = "bert.encoder.layer.1.attention.self.query.weight"


@pytest.fixture
def independent_study(tmp_path):
    original_paths = {}
    checkpoints = []
    components = (
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    )
    for backbone in ("encoder_a", "encoder_b"):
        tensors = {}
        for layer in range(3):
            for component in components:
                key = f"bert.encoder.layer.{layer}.{component}.weight"
                tensors[key] = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])) + layer * 0.1
        tensors["bert.embeddings.word_embeddings.weight"] = torch.ones(7, 4)
        original_path = tmp_path / "models" / backbone
        original_path.mkdir(parents=True)
        save_file(tensors, original_path / "model.safetensors")
        config = {"model_type": "bert", "hidden_size": 4, "num_hidden_layers": 3}
        (original_path / "config.json").write_text(json.dumps(config))
        original_paths[backbone] = original_path
        for language in ("es", "it"):
            for entity in ("disease", "symptom"):
                bump = (0.03 if language == "es" else 0.07) * (1 if entity == "disease" else -1)
                directory = tmp_path / "weights" / language / entity / backbone
                directory.mkdir(parents=True)
                updated = {key: value + bump for key, value in tensors.items()}
                updated["classifier.weight"] = torch.ones(2, 4)
                save_file(updated, directory / "model.safetensors")
                (directory / "config.json").write_text(json.dumps(config))
                checkpoints.append(CheckpointSpec(backbone, language, entity, directory))
    return checkpoints, original_paths


def reject_decomposition(*args, **kwargs):
    raise AssertionError("This requested metric must not run a decomposition")


def canonical_table(data_df):
    columns = sorted(data_df.columns)
    order = [name for name in ("checkpoint", "source", "tensor") if name in data_df]
    return data_df.loc[:, columns].sort_values(order, kind="stable").reset_index(drop=True)


def test_relative_frobenius_is_independent_and_does_not_run_svd(independent_study, monkeypatch):
    checkpoints, originals = independent_study
    monkeypatch.setattr(torch.linalg, "svd", reject_decomposition)
    monkeypatch.setattr(torch.linalg, "svdvals", reject_decomposition)
    spectral_module = importlib.import_module("weight_analysis.metrics.spectral")
    monkeypatch.setattr(spectral_module, "svds", reject_decomposition)

    result = analyze(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["relative_frobenius"],
        tensors=[QUERY],
    )

    assert result.analysis == "pretraining_drift"
    assert result.unit_id
    assert len(result.data) == 1
    row = result.data.iloc[0]
    expected = 0.03 * np.sqrt(16) / np.linalg.norm(np.diag([4.0, 3.0, 2.0, 1.0]))
    assert row.relative_frobenius == pytest.approx(expected, rel=1e-5)
    assert "spectral_norm" not in result.data


def test_final_matrix_frobenius_needs_neither_svd_nor_original_weights(
    independent_study, monkeypatch
):
    checkpoints, _ = independent_study
    monkeypatch.setattr(torch.linalg, "svd", reject_decomposition)
    monkeypatch.setattr(torch.linalg, "svdvals", reject_decomposition)
    spectral_module = importlib.import_module("weight_analysis.metrics.spectral")
    monkeypatch.setattr(spectral_module, "svds", reject_decomposition)

    result = analyze(
        analysis="spectral_fingerprints",
        checkpoints=checkpoints[:1],
        metrics=["matrix_fro"],
        sources=["final"],
        tensors=[QUERY],
    )

    expected = np.linalg.norm(np.diag([4.0, 3.0, 2.0, 1.0]) + 0.03)
    assert len(result.data) == 1
    assert result.data.iloc[0].matrix_fro == pytest.approx(expected, rel=1e-6)
    assert result.data.iloc[0].source == "final"


def test_layer_component_and_tensor_selection_happens_before_values_are_read(
    independent_study, monkeypatch
):
    checkpoints, originals = independent_study
    observed = []
    original_get = WeightReader.get

    def selected_only(self, key):
        observed.append(key)
        assert key == QUERY, f"Unrequested tensor materialized: {key}"
        return original_get(self, key)

    monkeypatch.setattr(WeightReader, "get", selected_only)
    result = analyze(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        layers=[0],
        components=["q"],
        metrics=["relative_frobenius"],
    )

    assert observed
    assert set(result.data.tensor) == {QUERY}
    assert set(result.data.layer) == {0}
    assert set(result.data.component) == {"q"}


def test_checkpoint_dimension_filters_do_not_read_unselected_checkpoints(
    independent_study, monkeypatch
):
    checkpoints, originals = independent_study
    selected = replace(checkpoints[0], fold=2, seed=13, checkpoint_id="epoch-7")
    candidates = [*checkpoints, selected]
    original_get = WeightReader.get
    allowed_paths = {selected.path, originals[selected.backbone]}

    def selected_only(self, key):
        assert self.path in allowed_paths
        return original_get(self, key)

    monkeypatch.setattr(WeightReader, "get", selected_only)
    result = analyze(
        analysis="pretraining_drift",
        checkpoints=candidates,
        originals=originals,
        backbones=[selected.backbone],
        languages=["es"],
        entities=["disease"],
        folds=[2],
        seeds=[13],
        checkpoint_ids=["epoch-7"],
        tensors=[QUERY],
        metrics=["relative_frobenius"],
    )

    assert len(result.data) == 1
    assert set(result.data.checkpoint) == {selected.uid}
    assert set(result.data.fold) == {2}
    assert set(result.data.seed) == {13}


def test_default_execution_has_no_disk_outputs(independent_study, tmp_path, monkeypatch):
    checkpoints, originals = independent_study
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    before = set(tmp_path.rglob("*"))

    analyze(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["relative_frobenius"],
        tensors=[QUERY],
    )

    assert set(tmp_path.rglob("*")) == before


def test_explicit_cache_resumes_without_materializing_weights(
    independent_study, tmp_path, monkeypatch
):
    checkpoints, originals = independent_study
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["relative_frobenius"],
        tensors=[QUERY],
        cache_dir=tmp_path / "cache",
    )
    first = analyze(**arguments)

    def no_values(*args, **kwargs):
        raise AssertionError("A valid completed cache unit must not be recomputed")

    monkeypatch.setattr(WeightReader, "get", no_values)
    resumed = analyze(**arguments)

    assert resumed.unit_id == first.unit_id
    pd.testing.assert_frame_equal(
        canonical_table(first.data), canonical_table(resumed.data), check_dtype=False
    )


def test_modified_checkpoint_invalidates_cached_result(independent_study, tmp_path):
    checkpoints, originals = independent_study
    checkpoint = checkpoints[0]
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=[checkpoint],
        originals=originals,
        metrics=["relative_frobenius"],
        tensors=[QUERY],
        cache_dir=tmp_path / "cache",
    )
    first = analyze(**arguments)
    path = checkpoint.path / "model.safetensors"
    tensors = load_file(path)
    tensors[QUERY] = tensors[QUERY] + 0.25
    save_file(tensors, path)

    refreshed = analyze(**arguments)

    assert refreshed.unit_id != first.unit_id
    assert refreshed.data.iloc[0].relative_frobenius > first.data.iloc[0].relative_frobenius


def test_reversed_input_order_preserves_unit_and_results(independent_study):
    checkpoints, originals = independent_study
    arguments = dict(
        analysis="pretraining_drift",
        originals=originals,
        metrics=["relative_frobenius"],
        tensors=[QUERY, OTHER_QUERY],
    )
    forward = analyze(checkpoints=checkpoints, **arguments)
    reverse = analyze(checkpoints=list(reversed(checkpoints)), **arguments)

    assert forward.unit_id == reverse.unit_id
    pd.testing.assert_frame_equal(canonical_table(forward.data), canonical_table(reverse.data))


def test_independent_subsets_merge_to_same_values_as_joint_execution(independent_study):
    checkpoints, originals = independent_study
    arguments = dict(
        analysis="pretraining_drift",
        originals=originals,
        metrics=["relative_frobenius"],
        tensors=[QUERY],
    )
    combined = analyze(checkpoints=checkpoints, **arguments)
    left = analyze(checkpoints=checkpoints[:4], **arguments)
    right = analyze(checkpoints=checkpoints[4:], **arguments)
    merged = merge_results([left, right])
    reverse = merge_results([right, left, left])

    pd.testing.assert_frame_equal(
        canonical_table(combined.data), canonical_table(merged.data), check_dtype=False
    )
    pd.testing.assert_frame_equal(merged.data, reverse.data)
    assert merged.unit_id == reverse.unit_id


@pytest.mark.parametrize("analysis", ["cross_language", "cross_entity", "language_entity"])
def test_related_comparisons_can_run_independently(independent_study, analysis):
    checkpoints, originals = independent_study
    result = analyze(
        analysis=analysis,
        checkpoints=checkpoints,
        originals=originals,
        metrics=["cosine"],
        layers=[0],
        components=["q"],
    )

    assert result.analysis == analysis
    assert not result.data.empty
    assert "cosine" in result.data
    assert set(result.data.layer) == {0}
    assert set(result.data.component) == {"q"}
    assert np.isfinite(result.data.cosine).all()


def test_factorial_analysis_runs_independently(independent_study):
    checkpoints, originals = independent_study
    selected = [checkpoint for checkpoint in checkpoints if checkpoint.backbone == "encoder_a"]
    result = analyze(
        analysis="language_entity_decomposition",
        checkpoints=selected,
        originals=originals,
        layers=[0],
        components=["q"],
    )

    assert not result.data.empty
    fractions = result.data[["fraction_language", "fraction_entity", "fraction_interaction"]]
    assert np.allclose(fractions.sum(axis=1), 1, atol=1e-8)


def test_unknown_metric_or_analysis_fails_before_loading_values(independent_study, monkeypatch):
    checkpoints, originals = independent_study
    monkeypatch.setattr(WeightReader, "get", reject_decomposition)
    for arguments in (
        {"analysis": "not_an_analysis"},
        {"analysis": "pretraining_drift", "metrics": ["not_a_metric"]},
    ):
        with pytest.raises(ValueError):
            analyze(checkpoints=checkpoints, originals=originals, **arguments)


def test_cross_backbone_reuses_explicit_spectra_without_reading_tensor_values(
    independent_study, monkeypatch
):
    checkpoints, _ = independent_study
    spectra = analyze(
        analysis="spectral_fingerprints",
        checkpoints=checkpoints,
        sources=["final"],
        metrics=["matrix_fro"],
    )
    direct = analyze(
        analysis="cross_backbone",
        checkpoints=checkpoints,
        metrics=["trace_distance_final"],
    )

    def no_values(*args, **kwargs):
        raise AssertionError("Explicit spectral intermediates must not reread tensor values")

    monkeypatch.setattr(WeightReader, "get", no_values)
    reused = analyze(
        analysis="cross_backbone",
        checkpoints=checkpoints,
        metrics=["trace_distance_final"],
        spectral_results=spectra,
    )
    reused_frame = analyze(
        analysis="cross_backbone",
        checkpoints=checkpoints,
        metrics=["trace_distance_final"],
        spectral_results=spectra.data,
    )

    assert len(reused.data) == 4
    assert np.allclose(reused.data.trace_distance_final, 0, atol=1e-12)
    ordering = ["language", "entity", "backbone_a", "backbone_b"]
    pd.testing.assert_frame_equal(
        direct.data.sort_values(ordering).reset_index(drop=True),
        reused.data.sort_values(ordering).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(reused.data, reused_frame.data)


def test_model_centrality_consumes_explicit_pairs_and_preserves_fold_seed_strata(
    independent_study, monkeypatch
):
    checkpoints, _ = independent_study
    pairs = analyze(
        analysis="cross_backbone",
        checkpoints=checkpoints,
        metrics=["trace_distance_final"],
    )

    def no_values(*args, **kwargs):
        raise AssertionError("Centrality must consume pair results without loading checkpoints")

    monkeypatch.setattr(WeightReader, "get", no_values)
    result = analyze(
        analysis="model_centrality",
        pair_results=pairs,
        metrics=["trace_distance_final"],
    )
    reordered = analyze(
        analysis="model_centrality",
        pair_results=pairs.data.iloc[::-1],
        metrics=["trace_distance_final"],
    )

    assert len(result.data) == 8
    assert result.data.complete_comparison.all()
    assert set(result.data.fold) == {1}
    assert result.data.seed.isna().all()
    assert np.allclose(result.data.mean_distance, 0, atol=1e-12)
    pd.testing.assert_frame_equal(result.data, reordered.data)
    with pytest.raises(ValueError, match="pair_results"):
        analyze(analysis="model_centrality", metrics=["trace_distance_final"])


def test_partial_cache_artifacts_are_never_treated_as_completed_units(independent_study, tmp_path):
    checkpoints, originals = independent_study
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    partial = cache_path / ".interrupted.parquet.partial"
    partial.write_bytes(b"incomplete parquet write")

    result = analyze(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        metrics=["relative_frobenius"],
        tensors=[QUERY],
        cache_dir=cache_path,
    )

    assert len(result.data) == 1
    assert partial.read_bytes() == b"incomplete parquet write"
    assert len(list(cache_path.glob("*.parquet"))) == 1


def test_selected_checkpoints_do_not_require_excluded_paths(independent_study, tmp_path):
    checkpoints, originals = independent_study
    missing = replace(checkpoints[1], path=tmp_path / "not-mounted", fold=99)
    selected = checkpoints[0]

    result = analyze(
        analysis="pretraining_drift",
        checkpoints=[selected, missing],
        originals=originals,
        folds=[1],
        metrics=["relative_frobenius"],
        tensors=[QUERY],
    )

    assert list(result.data.checkpoint) == [selected.uid]
