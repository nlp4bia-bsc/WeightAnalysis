"""Explicit checkpoint identities, safe local loading and metadata-only selection."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from weight_analysis.core import (
    Checkpoint,
    CheckpointSpec,
    WeightReader,
    audit_alignment,
    config_metadata,
    get_delta,
    normalize,
    selected_keys,
)

QUERY = "bert.encoder.layer.0.attention.self.query.weight"
KEY = "bert.encoder.layer.1.attention.self.key.weight"
EMBEDDINGS = "bert.embeddings.word_embeddings.weight"


def write_safe(directory: Path, tensors: dict[str, torch.Tensor], *, model_type="bert"):
    directory.mkdir(parents=True)
    save_file(tensors, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))

    return directory


def test_identity_covers_dimensions_without_parsing_paths(tmp_path):
    checkpoint = CheckpointSpec("bert", "xx", "custom_entity", str(tmp_path / "fold-9_seed-7"))

    assert isinstance(checkpoint.path, Path)
    assert checkpoint.fold == 1
    assert checkpoint.seed is None
    assert checkpoint.uid == replace(checkpoint).uid
    alternatives = [
        replace(checkpoint, backbone="roberta"),
        replace(checkpoint, language="yy"),
        replace(checkpoint, entity="another_entity"),
        replace(checkpoint, path=tmp_path / "another"),
        replace(checkpoint, fold=2),
        replace(checkpoint, seed=0),
        replace(checkpoint, checkpoint_id="epoch-2"),
    ]
    assert len({checkpoint.uid, *(item.uid for item in alternatives)}) == 8

    with pytest.raises(FrozenInstanceError):
        checkpoint.fold = 2


def test_explicit_identity_avoids_separator_collisions(tmp_path):
    left = CheckpointSpec("bert::es", "it", "disease", tmp_path)
    right = CheckpointSpec("bert", "es::it", "disease", tmp_path)

    assert left.uid != right.uid
    assert replace(left, checkpoint_id="same").uid != replace(right, checkpoint_id="same").uid


def test_legacy_checkpoint_identity_remains_unchanged(tmp_path):
    checkpoint = Checkpoint("bert", "es", "disease", tmp_path, fold=2)

    assert checkpoint.uid == "bert::es::disease::2"


@pytest.mark.parametrize(
    "changes",
    [{"backbone": ""}, {"language": ""}, {"entity": ""}, {"fold": 0}, {"fold": -1}, {"fold": True}],
)
def test_explicit_identity_rejects_invalid_metadata(tmp_path, changes):
    values = dict(backbone="bert", language="es", entity="disease", path=tmp_path)
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        CheckpointSpec(**values)


def test_safe_reader_supports_string_paths_and_canonical_keys(tmp_path):
    directory = write_safe(
        tmp_path / "safe",
        {
            "module.encoder.layer.0.attention.self.query.weight": torch.arange(6)
            .reshape(2, 3)
            .float(),
            "embeddings.position_ids": torch.arange(4),
        },
    )

    with WeightReader(str(directory)) as reader:
        assert QUERY in reader.keys
        assert reader.shape_dtype(QUERY) == ((2, 3), True)
        assert reader.shape_dtype("bert.embeddings.position_ids") == ((4,), False)
        torch.testing.assert_close(reader.get(QUERY), torch.arange(6).reshape(2, 3).float())

    assert config_metadata(str(directory))["model_type"] == "bert"
    assert normalize("module.model.encoder.layer.0.weight", "bert") == "bert.encoder.layer.0.weight"


@pytest.mark.parametrize("legacy_serialization", [False, True])
def test_torch_reader_accepts_nested_state_dict_without_non_tensor_values(
    tmp_path, legacy_serialization
):
    directory = tmp_path / "pytorch"
    directory.mkdir()
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.save(
        {"state_dict": {QUERY: tensor, "epoch": 5}},
        directory / "pytorch_model.bin",
        _use_new_zipfile_serialization=not legacy_serialization,
    )

    with WeightReader(directory) as reader:
        assert reader.keys == {QUERY}
        torch.testing.assert_close(reader.get(QUERY), tensor)


def test_indexed_shards_read_selected_tensors(tmp_path):
    directory = tmp_path / "sharded"
    directory.mkdir()
    save_file({QUERY: torch.eye(2)}, directory / "part-1.safetensors")
    save_file({KEY: torch.eye(3)}, directory / "part-2.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {QUERY: "part-1.safetensors", KEY: "part-2.safetensors"},
            }
        )
    )

    with WeightReader(directory) as reader:
        assert reader.keys == {QUERY, KEY}
        torch.testing.assert_close(reader.get(KEY), torch.eye(3))


@pytest.mark.parametrize("mapping", [None, {}, [], "not-a-map"])
def test_reader_rejects_malformed_shard_indexes(tmp_path, mapping):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))

    with pytest.raises(ValueError, match="index"):
        with WeightReader(tmp_path):
            pass


def test_shard_paths_cannot_escape_the_checkpoint(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    save_file({QUERY: torch.eye(2)}, tmp_path / "outside.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {QUERY: "../outside.safetensors"},
            }
        )
    )

    with pytest.raises(ValueError, match="shard"):
        with WeightReader(directory):
            pass


def test_reader_rejects_missing_shards_and_missing_indexed_keys(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {QUERY: "missing.safetensors"}}))

    with pytest.raises(FileNotFoundError):
        with WeightReader(tmp_path):
            pass

    save_file({KEY: torch.eye(2)}, tmp_path / "missing.safetensors")
    with pytest.raises(ValueError, match="mismatch"):
        with WeightReader(tmp_path):
            pass


def test_reader_rejects_colliding_canonical_names(tmp_path):
    directory = write_safe(
        tmp_path / "collision",
        {
            "encoder.layer.0.attention.self.query.weight": torch.eye(2),
            QUERY: torch.ones(2, 2),
        },
    )

    with pytest.raises(ValueError, match="collision"):
        with WeightReader(directory):
            pass


def test_selection_filters_metadata_without_materializing_parameters(tmp_path, monkeypatch):
    directory = write_safe(
        tmp_path / "selected",
        {
            QUERY: torch.eye(2),
            KEY: torch.eye(2),
            EMBEDDINGS: torch.ones(4, 2),
            "bert.embeddings.position_ids": torch.arange(4),
            "classifier.weight": torch.ones(3, 2),
        },
    )

    def forbidden_get(*args, **kwargs):
        raise AssertionError("selection must inspect metadata only")

    with WeightReader(directory) as reader:
        monkeypatch.setattr(reader, "get", forbidden_get)
        assert selected_keys(reader) == sorted([QUERY, KEY, EMBEDDINGS])
        assert selected_keys(reader, layers=[0]) == [QUERY]
        assert selected_keys(reader, components=["k"]) == [KEY]
        assert selected_keys(reader, tensors=[EMBEDDINGS]) == [EMBEDDINGS]
        assert selected_keys(reader, layers=[0], components=["k"]) == []


def test_alignment_coverage_uses_elements_and_excludes_buffers_and_task_heads(tmp_path):
    original = write_safe(tmp_path / "original", {QUERY: torch.ones(10, 10)})
    final = write_safe(
        tmp_path / "final",
        {
            QUERY: torch.ones(10, 10),
            "bert.encoder.layer.0.missing.bias": torch.ones(1),
            "classifier.weight": torch.ones(1000, 2),
            "bert.embeddings.position_ids": torch.arange(500),
        },
    )

    with WeightReader(final) as reader, WeightReader(original) as base:
        keys, status = audit_alignment(reader, base, min_fraction=0.99)
        assert keys == [QUERY]
        assert status["element_coverage"] == pytest.approx(100 / 101)
        assert status["candidate_elements"] == 101
        with pytest.raises(ValueError, match="alignment"):
            audit_alignment(reader, base, min_fraction=1)


def test_delta_is_float64_and_preserves_exact_reference_subtraction(tmp_path):
    original = write_safe(tmp_path / "original", {QUERY: torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
    final = write_safe(tmp_path / "final", {QUERY: torch.tensor([[2.0, 2.0], [5.0, 1.0]])})

    with WeightReader(final) as reader, WeightReader(original) as base:
        trained, initial, delta = get_delta(reader, base, QUERY)

    assert trained.dtype == initial.dtype == delta.dtype == torch.float64
    torch.testing.assert_close(delta, torch.tensor([[1.0, 0.0], [2.0, -3.0]], dtype=torch.float64))


@pytest.mark.parametrize(
    "final_tensor",
    [
        torch.ones(3, 2),
        torch.full((2, 2), float("nan")),
        torch.full((2, 2), float("inf")),
        torch.ones(2, 2).long(),
    ],
)
def test_delta_rejects_incompatible_or_nonfinite_weights(tmp_path, final_tensor):
    original = write_safe(tmp_path / "original", {QUERY: torch.eye(2)})
    final = write_safe(tmp_path / "final", {QUERY: final_tensor})

    with WeightReader(final) as reader, WeightReader(original) as base:
        with pytest.raises(ValueError):
            get_delta(reader, base, QUERY)


def test_failed_context_entry_releases_handles_and_mapping(tmp_path):
    directory = write_safe(
        tmp_path / "collision",
        {
            "encoder.layer.0.attention.self.query.weight": torch.eye(2),
            QUERY: torch.ones(2, 2),
        },
    )
    reader = WeightReader(directory)
    with pytest.raises(ValueError, match="collision"):
        reader.__enter__()
    assert not reader.handles
    assert not reader.mapping
    assert not reader.raw_names


def test_configuration_tolerates_legacy_but_exposes_strict_validation(tmp_path):
    (tmp_path / "config.json").write_text("[")
    assert config_metadata(tmp_path) == {}
    with pytest.raises(ValueError):
        config_metadata(tmp_path, strict=True)
    (tmp_path / "config.json").write_text("[]")
    assert config_metadata(tmp_path) == {}
    with pytest.raises(ValueError, match="object"):
        config_metadata(tmp_path, strict=True)


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan"), float("inf")])
def test_alignment_rejects_invalid_coverage_thresholds(tmp_path, threshold):
    directory = write_safe(tmp_path / "valid", {QUERY: torch.eye(2)})
    with WeightReader(directory) as reader, WeightReader(directory) as base:
        with pytest.raises(ValueError, match="min_fraction"):
            audit_alignment(reader, base, threshold)
