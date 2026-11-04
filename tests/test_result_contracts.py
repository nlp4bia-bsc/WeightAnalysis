"""Result integrity, atomic persistence and order-independent unit assembly."""

from __future__ import annotations

import dataclasses
import itertools
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from weight_analysis.results import AnalysisResult, merge_results, read_result, write_result
from weight_analysis.results.models import data_digest


def result(unit_id="a", values=(2.0, 1.0), *, configuration=None):
    return AnalysisResult(
        "pretraining_drift",
        pd.DataFrame({"checkpoint": [unit_id] * len(values), "value": values}),
        {
            "unit_id": unit_id,
            "configuration": configuration or {"metrics": ["delta_l2"]},
            "inputs": [{"sha256": unit_id * 64}],
        },
    )


def test_result_contract_and_copied_inputs():
    data_df = pd.DataFrame({"value": [1.0]})
    provenance = {"configuration": {"layers": [1]}}
    output = AnalysisResult("test", data_df, provenance)
    data_df.loc[0, "value"] = 9
    provenance["configuration"]["layers"].append(2)
    assert output.data.value.tolist() == [1]
    assert output.provenance == {"configuration": {"layers": [1]}}
    assert output.unit_id is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        output.analysis = "other"


@pytest.mark.parametrize("analysis", ["", " ", None])
def test_invalid_analysis(analysis):
    with pytest.raises(ValueError, match="analysis"):
        AnalysisResult(analysis, pd.DataFrame(), {})


@pytest.mark.parametrize("provenance", [{"unit_id": 3}, {"unit_id": ""}, {"bad": np.nan}])
def test_invalid_provenance(provenance):
    with pytest.raises((TypeError, ValueError)):
        AnalysisResult("test", pd.DataFrame(), provenance)


def test_roundtrip_arrays_nulls_and_metadata(tmp_path):
    frame = pd.DataFrame(
        {
            "value": [np.nan, 3.0],
            "curve": [[0.0, 1.0], [0.0, 0.5]],
            "layer": pd.Series([None, 2], dtype="Int64"),
        }
    )
    original = AnalysisResult("spectral_fingerprints", frame, {"unit_id": "spectra", "seed": 42})
    path = write_result(original, tmp_path / "nested" / "result.parquet")
    loaded = read_result(path)
    assert loaded.analysis == original.analysis
    assert loaded.provenance == original.provenance
    assert loaded.unit_id == "spectra"
    assert data_digest(loaded.data) == data_digest(original.data)
    assert loaded.data["layer"].dtype == original.data["layer"].dtype
    metadata = pq.read_schema(path).metadata
    assert metadata[b"weight_analysis.schema_version"] == b"1.0"
    assert json.loads(metadata[b"weight_analysis.provenance"]) == original.provenance
    assert not list(path.parent.glob("*.partial"))


def test_empty_result_roundtrip(tmp_path):
    output = AnalysisResult("empty", pd.DataFrame(), {"unit_id": "none"})
    loaded = read_result(write_result(output, tmp_path / "empty.parquet"))
    assert loaded.data.empty and not len(loaded.data.columns)
    assert loaded.unit_id == "none"


def test_tampered_data_rejected(tmp_path):
    path = write_result(result(), tmp_path / "result.parquet")
    table = pq.read_table(path)
    column = pa.array([55.0, 66.0])
    pq.write_table(table.set_column(table.column_names.index("value"), "value", column), path)
    with pytest.raises(ValueError, match="digest mismatch"):
        read_result(path)


def test_corrupt_or_unversioned_files_rejected(tmp_path):
    path = tmp_path / "result.parquet"
    path.write_bytes(b"not a parquet")
    with pytest.raises((ValueError, OSError)):
        read_result(path)
    pq.write_table(pa.table({"value": [1]}), path)
    with pytest.raises(ValueError, match="metadata"):
        read_result(path)


def test_unsupported_schema_rejected(tmp_path):
    path = write_result(result(), tmp_path / "result.parquet")
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    metadata[b"weight_analysis.schema_version"] = b"999"
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match="Unsupported result schema"):
        read_result(path)


def test_interrupted_replace_preserves_completed_result(tmp_path, monkeypatch):
    import weight_analysis.results.storage as storage

    path = write_result(result(), tmp_path / "result.parquet")
    previous_bytes = path.read_bytes()

    def interrupted_replace(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr(storage.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated"):
        write_result(result(values=(10.0, 20.0)), path)
    assert path.read_bytes() == previous_bytes
    assert not list(tmp_path.glob("*.partial"))


def test_merge_permutations_identical_and_retry_deduplicated():
    a, b, c = result("a"), result("b", (3.0,)), result("c", ())
    expected = merge_results([a, b, c])
    for permutation in itertools.permutations([a, b, c]):
        actual = merge_results([*permutation, a])
        pd.testing.assert_frame_equal(actual.data, expected.data)
        assert actual.provenance == expected.provenance
    assert len(expected.data) == 3
    assert expected.provenance["unit_ids"] == ["a", "b", "c"]


def test_merge_composes_without_duplicating_original_units(tmp_path):
    a, b, c = result("a"), result("b", (3.0,)), result("c", ())
    intermediate = merge_results([a, b])
    saved = read_result(write_result(intermediate, tmp_path / "intermediate.parquet"))
    actual = merge_results([saved, b, c])
    expected = merge_results([a, b, c])
    pd.testing.assert_frame_equal(actual.data, expected.data)
    assert actual.provenance == expected.provenance


def test_merge_does_not_deduplicate_distinct_units_with_equal_values():
    a = AnalysisResult("test", pd.DataFrame({"value": [1]}), {"unit_id": "a"})
    b = AnalysisResult("test", pd.DataFrame({"value": [1]}), {"unit_id": "b"})
    assert len(merge_results([a, b]).data) == 2


def test_conflicting_repeated_unit_rejected():
    with pytest.raises(ValueError, match="Conflicting repeated"):
        merge_results([result(), result(values=(100.0,))])
    a = result()
    b = AnalysisResult(a.analysis, a.data, {**a.provenance, "extra": True})
    with pytest.raises(ValueError, match="Conflicting repeated"):
        merge_results([a, b])


def test_incompatible_analyses_settings_and_columns_rejected():
    a = result()
    with pytest.raises(ValueError, match="different analyses"):
        merge_results([a, AnalysisResult("other", a.data, a.provenance)])
    with pytest.raises(ValueError, match="configuration"):
        merge_results([a, result("b", configuration={"metrics": ["relative_frobenius"]})])
    b = AnalysisResult(a.analysis, pd.DataFrame({"other": [1]}), {**a.provenance, "unit_id": "b"})
    with pytest.raises(ValueError, match="column schemas"):
        merge_results([a, b])


def test_empty_merge_is_explicit_and_empty_units_are_kept():
    with pytest.raises(ValueError, match="pass analysis explicitly"):
        merge_results([])
    assert merge_results([], analysis="test").data.empty
    a = AnalysisResult("test", pd.DataFrame(), {"unit_id": "a"})
    b = AnalysisResult("test", pd.DataFrame(), {"unit_id": "b"})
    output = merge_results([b, a])
    assert output.data.empty
    assert output.provenance["unit_ids"] == ["a", "b"]
    assert merge_results([output, a]).provenance == output.provenance


def test_value_digest_is_index_row_and_column_order_independent():
    frame = pd.DataFrame({"b": [1.0, np.nan], "a": [[1, 2], [3, 4]]}, index=[9, 8])
    reordered = frame.iloc[::-1].loc[:, ["a", "b"]].reset_index(drop=True)
    assert data_digest(frame) == data_digest(reordered)
    with pytest.raises(ValueError, match="infinite"):
        data_digest(pd.DataFrame({"value": [float("inf")]}))


def test_optional_observation_identity_rejects_overlapping_units():
    a = AnalysisResult(
        "test", pd.DataFrame({"checkpoint": ["x"], "value": [1.0]}), {"unit_id": "a"}
    )
    b = AnalysisResult(
        "test", pd.DataFrame({"checkpoint": ["x"], "value": [1.0]}), {"unit_id": "b"}
    )
    assert len(merge_results([a, b]).data) == 2
    with pytest.raises(ValueError, match="Overlapping observations"):
        merge_results([a, b], observation_keys=["checkpoint"])
    assert len(merge_results([a, a], observation_keys=["checkpoint"]).data) == 1


def test_distinct_observations_with_identical_metric_values_remain_distinct():
    a = AnalysisResult(
        "test", pd.DataFrame({"checkpoint": ["x"], "value": [1.0]}), {"unit_id": "a"}
    )
    b = AnalysisResult(
        "test", pd.DataFrame({"checkpoint": ["y"], "value": [1.0]}), {"unit_id": "b"}
    )
    assert len(merge_results([a, b], observation_keys=["checkpoint"]).data) == 2
    with pytest.raises(ValueError, match="Missing observation_keys"):
        merge_results([a, b], observation_keys=["unknown"])
