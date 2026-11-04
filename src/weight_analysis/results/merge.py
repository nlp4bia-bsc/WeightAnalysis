"""Deterministic assembly of independently computed result units."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from .models import AnalysisResult, canonical_json, data_digest, row_keys


def _unit_key(result: AnalysisResult) -> str:
    if result.unit_id is not None:
        return result.unit_id
    payload = canonical_json({"provenance": result.provenance, "data": data_digest(result.data)})
    return hashlib.sha256(payload.encode()).hexdigest()


def _original_units(result: AnalysisResult) -> list[AnalysisResult]:
    if result.provenance.get("merge_format") != "units-v1":
        return [result]
    units = []
    assigned_rows = []
    for entry in result.provenance["units"]:
        indices = entry["row_indices"]
        assigned_rows.extend(indices)
        frame = result.data.iloc[indices].loc[:, entry["columns"]].reset_index(drop=True)
        original = AnalysisResult(result.analysis, frame, entry["provenance"])
        if _unit_key(original) != entry["unit_id"]:
            raise ValueError("Merged result contains invalid original unit identities")
        units.append(original)
    if sorted(assigned_rows) != list(range(len(result.data))):
        raise ValueError("Merged result must assign each row to exactly one original unit")
    return units


def merge_results(
    results: Iterable[AnalysisResult],
    *,
    analysis: str | None = None,
    observation_keys: Sequence[str] | None = None,
) -> AnalysisResult:
    """Merge compatible units independently of order and deduplicate exact retries.

    Units must have the same analysis, column names and provenance configuration.
    Each unit identity can occur repeatedly only with identical values and
    provenance. Different unit identities keep their rows even when their values
    coincide. Provenance for every original unit is retained under ``units``.
    No metrics, rankings, or group statistics are recomputed by this operation.
    Pass observation_keys to reject overlapping scientific observations across
    distinct units. Without keys, rows from distinct units are preserved and can
    represent repeated observations; callers must avoid double weighting them.
    """
    supplied = list(results)
    if not supplied:
        if analysis is None:
            raise ValueError("Cannot infer analysis from no results; pass analysis explicitly")
        return AnalysisResult(analysis, pd.DataFrame(), {"units": [], "unit_ids": []})
    if not all(isinstance(result, AnalysisResult) for result in supplied):
        raise TypeError("Every result must be an AnalysisResult")
    units = [unit for result in supplied for unit in _original_units(result)]
    if not units:
        return merge_results([], analysis=analysis or supplied[0].analysis)
    expected_analysis = analysis or units[0].analysis
    configuration = units[0].provenance.get("configuration")
    expected_configuration = canonical_json(configuration)
    columns: set[str] | None = None
    unique: dict[str, AnalysisResult] = {}
    for result in units:
        if result.analysis != expected_analysis:
            raise ValueError("Cannot merge different analyses")
        if canonical_json(result.provenance.get("configuration")) != expected_configuration:
            raise ValueError("Cannot merge results with incompatible provenance configuration")
        current_columns = set(result.data.columns)
        if current_columns:
            if columns is not None and current_columns != columns:
                raise ValueError("Cannot merge results with different column schemas")
            columns = current_columns
        key = _unit_key(result)
        previous = unique.get(key)
        if previous is not None:
            if data_digest(previous.data) != data_digest(result.data) or canonical_json(
                previous.provenance
            ) != canonical_json(result.provenance):
                raise ValueError(f"Conflicting repeated result unit: {key}")
            continue
        unique[key] = result
    ordered = sorted(unique.items())
    tables = [result.data for _, result in ordered if len(result.data.columns)]
    row_units = [key for key, result in ordered for _ in range(len(result.data))]
    if tables:
        data_df = pd.concat(tables, ignore_index=True).loc[:, sorted(columns or [])]
        keys = row_keys(data_df)
        row_order = sorted(range(len(keys)), key=lambda index: (keys[index], row_units[index]))
        data_df = data_df.iloc[row_order].reset_index(drop=True)
        row_units = [row_units[index] for index in row_order]
    else:
        data_df = pd.DataFrame()
    if observation_keys is not None:
        if isinstance(observation_keys, (str, bytes)):
            raise TypeError("observation_keys must be a sequence of column names")
        keys = list(observation_keys)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("observation_keys must contain distinct column names")
        missing_keys = set(keys) - set(data_df.columns)
        if missing_keys:
            raise ValueError(f"Missing observation_keys columns: {sorted(missing_keys)}")
        if data_df.duplicated(keys).any():
            sample = data_df.loc[data_df.duplicated(keys, keep=False), keys].head(5)
            raise ValueError(
                f"Overlapping observations for observation_keys={keys}: {sample.to_dict('records')}"
            )
    provenance: dict[str, Any] = {
        "configuration": configuration,
        "merge_format": "units-v1",
        "unit_ids": [key for key, _ in ordered],
        "units": [
            {
                "unit_id": key,
                "provenance": result.provenance,
                "columns": list(result.data.columns),
                "row_indices": [index for index, owner in enumerate(row_units) if owner == key],
            }
            for key, result in ordered
        ],
    }
    payload = canonical_json({"analysis": expected_analysis, "unit_ids": provenance["unit_ids"]})
    provenance["unit_id"] = hashlib.sha256(payload.encode()).hexdigest()
    return AnalysisResult(expected_analysis, data_df, provenance)
