"""Independent mathematical and data-contract checks for descriptive analyses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from weight_analysis.analyses import factorial, validate_factorial_design
from weight_analysis.comparisons import model_centrality, model_distance_summary
from weight_analysis.core import Checkpoint


@pytest.fixture
def factorial_study(tmp_path: Path):
    base_path = tmp_path / "base"
    base_path.mkdir()
    key = "bert.embeddings.word_embeddings.weight"
    base = torch.ones(2, 2, dtype=torch.float64)
    save_file({key: base}, base_path / "model.safetensors")
    checkpoints = []
    for language, language_effect in (("es", -1), ("it", 1)):
        for entity, entity_effect in (("disease", -1), ("symptom", 1)):
            update = torch.tensor(
                [language_effect, 2 * entity_effect, 3 * language_effect * entity_effect, 0],
                dtype=torch.float64,
            ).reshape(2, 2)
            path = tmp_path / language / entity
            path.mkdir(parents=True)
            save_file({key: base + update}, path / "model.safetensors")
            checkpoints.append(Checkpoint("b1", language, entity, path))
    settings = SimpleNamespace(min_aligned_fraction=0.99, factorial_chunk_elements=1)
    return checkpoints, base_path, settings


def test_factorial_sums_match_orthogonal_known_effects(factorial_study):
    checkpoints, base_path, settings = factorial_study
    validate_factorial_design(checkpoints)
    rows = factorial(checkpoints, base_path, settings)
    tensor = next(row for row in rows if row["level"] == "tensor")
    assert tensor["ss_language"] == pytest.approx(4)
    assert tensor["ss_entity"] == pytest.approx(16)
    assert tensor["ss_interaction"] == pytest.approx(36)
    assert tensor["ss_total"] == pytest.approx(56)
    assert tensor["fraction_language"] + tensor["fraction_entity"] + tensor[
        "fraction_interaction"
    ] == pytest.approx(1)
    assert tensor["interpretation"] == "descriptive_one_checkpoint_per_cell_no_inference"


def test_factorial_is_order_and_chunk_independent(factorial_study):
    checkpoints, base_path, settings = factorial_study
    first = pd.DataFrame(factorial(checkpoints, base_path, settings))
    settings.factorial_chunk_elements = 100
    second = pd.DataFrame(factorial(list(reversed(checkpoints)), base_path, settings))
    pd.testing.assert_frame_equal(first, second)


def test_factorial_rejects_mixed_backbones_folds_and_duplicate_cells(factorial_study):
    checkpoints, _, _ = factorial_study
    with pytest.raises(ValueError, match="backbone, fold and seed"):
        validate_factorial_design([replace(checkpoints[0], fold=2), *checkpoints[1:]])
    with pytest.raises(ValueError, match="backbone, fold and seed"):
        validate_factorial_design([replace(checkpoints[0], backbone="b2"), *checkpoints[1:]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_factorial_design([*checkpoints, checkpoints[0]])
    with pytest.raises(ValueError, match="missing"):
        validate_factorial_design(checkpoints[:-1])


def test_factorial_rejects_seed_replicates():
    checkpoints = [
        SimpleNamespace(backbone="b", language=language, entity=entity, fold=1, seed=seed)
        for language, entity, seed in (
            ("es", "disease", 1),
            ("es", "symptom", 1),
            ("it", "disease", 1),
            ("it", "symptom", 2),
        )
    ]
    with pytest.raises(ValueError, match="seed"):
        validate_factorial_design(checkpoints)


def test_factorial_filters_before_tensor_materialization(factorial_study, monkeypatch):
    from importlib import import_module

    module = import_module("weight_analysis.analyses.factorial")
    checkpoints, base_path, settings = factorial_study

    def unexpected_read(*args, **kwargs):
        raise AssertionError("excluded tensors were materialized")

    monkeypatch.setattr(module.WeightReader, "get", unexpected_read)
    assert factorial(checkpoints, base_path, settings, components=("q",)) == []


@pytest.fixture
def pair_table():
    return pd.DataFrame(
        [
            {
                "language": "es",
                "entity": "disease",
                "backbone_a": "a",
                "backbone_b": "b",
                "curve_l1_delta": 1.0,
            },
            {
                "language": "es",
                "entity": "disease",
                "backbone_a": "a",
                "backbone_b": "c",
                "curve_l1_delta": 3.0,
            },
            {
                "language": "es",
                "entity": "disease",
                "backbone_a": "b",
                "backbone_b": "c",
                "curve_l1_delta": 2.0,
            },
        ]
    )


def test_centrality_matches_arithmetic_mean_and_historical_values(pair_table):
    before = pair_table.copy(deep=True)
    strict = model_centrality(pair_table, metrics=("curve_l1_delta",))
    historical = model_distance_summary(pair_table)
    pd.testing.assert_frame_equal(strict, historical)
    result = strict.set_index("backbone")
    assert result.loc["a", "mean_distance"] == 2
    assert result.loc["b", "mean_distance"] == 1.5
    assert result.loc["c", "mean_distance"] == 2.5
    assert result.loc["b", "rank_most_central"] == 1
    assert result.complete_comparison.all()
    pd.testing.assert_frame_equal(pair_table, before)


def test_centrality_is_independent_of_edge_order_and_orientation(pair_table):
    first = model_centrality(pair_table)
    shuffled = pair_table.sample(frac=1, random_state=4).reset_index(drop=True)
    shuffled[["backbone_a", "backbone_b"]] = shuffled[["backbone_b", "backbone_a"]].to_numpy()
    pd.testing.assert_frame_equal(first, model_centrality(shuffled))


def test_centrality_rejects_duplicate_reverse_edges(pair_table):
    reverse = pair_table.iloc[[0]].rename(
        columns={"backbone_a": "backbone_b", "backbone_b": "backbone_a"}
    )
    with pytest.raises(ValueError, match="duplicate undirected"):
        model_centrality(pd.concat([pair_table, reverse], ignore_index=True))


def test_centrality_missing_distances_do_not_count_as_neighbors(pair_table):
    pair_table.loc[0, "curve_l1_delta"] = np.nan
    result = model_centrality(pair_table).set_index("backbone")
    assert not result.loc["a", "complete_comparison"]
    assert not result.loc["b", "complete_comparison"]
    assert result.loc["c", "complete_comparison"]
    assert np.isnan(result.loc["a", "rank_most_central"])
    assert result.loc["a", "n_neighbors"] == 1


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf])
def test_centrality_rejects_invalid_distances(pair_table, value):
    pair_table.loc[0, "curve_l1_delta"] = value
    with pytest.raises(ValueError, match="nonnegative and finite"):
        model_centrality(pair_table)


def test_centrality_fold_and_seed_are_separate_strata(pair_table):
    first = pair_table.assign(fold=1, seed=42)
    second = pair_table.assign(fold=2, seed=7, curve_l1_delta=5.0)
    result = model_centrality(pd.concat([first, second], ignore_index=True))
    assert len(result) == 6
    assert set(result.fold) == {1, 2}
    assert (result[result.fold == 2].mean_distance == 5).all()


def test_centrality_computes_only_selected_metric(pair_table):
    pair_table["trace_distance_delta"] = -100.0
    result = model_centrality(pair_table, metrics=("curve_l1_delta",))
    assert set(result.metric) == {"curve_l1_delta"}
    with pytest.raises(ValueError, match="missing selected"):
        model_centrality(pair_table, metrics=("missing_distance",))


def test_checkpoint_description_reads_weights_once_per_tensor(factorial_study, monkeypatch):
    from importlib import import_module

    from weight_analysis.analyses import describe_checkpoint

    module = import_module("weight_analysis.analyses.drift")
    checkpoints, base_path, settings = factorial_study
    reads = []
    original_get = module.WeightReader.get

    def record_get(reader, key):
        reads.append((reader.path, key))
        return original_get(reader, key)

    monkeypatch.setattr(module.WeightReader, "get", record_get)
    drift_rows, spectra_rows = describe_checkpoint(
        checkpoints[0],
        base_path,
        settings,
        compute_spectra=False,
    )
    assert drift_rows and not spectra_rows
    assert len(reads) == 2
    assert len(set(reads)) == 2
