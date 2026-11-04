"""Selective intermediates for independent scientific computations."""

from __future__ import annotations

import pandas as pd

from ..provenance import checkpoint_fingerprint
from ..results import AnalysisResult


def _frame(value: pd.DataFrame | AnalysisResult | None) -> pd.DataFrame | None:
    if isinstance(value, AnalysisResult):
        return value.data
    if value is not None and not isinstance(value, pd.DataFrame):
        raise TypeError("Intermediate results must be DataFrames or AnalysisResult objects")
    return value


def _intermediate_lineage(value, *, expected_analysis, selected_ids, originals=None):
    if value is None:
        return None
    if not isinstance(value, AnalysisResult):
        return {"lineage": "unverified_caller_dataframe"}
    if value.analysis != expected_analysis:
        raise ValueError(f"Intermediate analysis must be {expected_analysis!r}")
    provenance = value.provenance
    units = [entry["provenance"] for entry in provenance.get("units", [])]
    units = units or [provenance]
    declared_originals = originals or {}
    declared_fingerprints = {}
    verified = False
    for unit in units:
        relevant = [
            checkpoint
            for checkpoint in unit.get("inputs", [])
            if checkpoint["checkpoint"] in selected_ids
        ]
        for checkpoint in relevant:
            actual = checkpoint_fingerprint(checkpoint["path"])
            if checkpoint.get("files") != actual["files"]:
                raise ValueError("Stale intermediate provenance: checkpoint content changed")
            verified = True
        backbones = {checkpoint["backbone"] for checkpoint in relevant}
        for backbone, original in unit.get("originals", {}).items():
            if backbone not in backbones:
                continue
            actual = checkpoint_fingerprint(original["path"])
            if original.get("files") != actual["files"]:
                raise ValueError("Stale intermediate provenance: original content changed")
            if backbone in declared_originals:
                if backbone not in declared_fingerprints:
                    declared_path = declared_originals[backbone]
                    declared_fingerprints[backbone] = (
                        actual
                        if str(declared_path) == actual["path"]
                        else checkpoint_fingerprint(declared_path)
                    )
                if original.get("files") != declared_fingerprints[backbone]["files"]:
                    raise ValueError(
                        f"Intermediate provenance conflicts with the declared original "
                        f"content for backbone {backbone!r}; recompute the intermediate"
                    )
    return {
        "lineage": "verified_inputs" if verified else "unverified_caller_result",
        "unit_id": value.unit_id,
        "provenance": provenance,
    }


def _select_table(table_df, *, checkpoints, layers=None, components=None, tensors=None):
    if table_df is None:
        return None
    required = {"checkpoint", "level", "component", "layer"}
    if required - set(table_df.columns):
        raise ValueError(f"Intermediate table requires columns {sorted(required)}")
    selected = table_df[table_df.checkpoint.isin([checkpoint.uid for checkpoint in checkpoints])]
    for column, values in (("layer", layers), ("component", components), ("tensor", tensors)):
        if values is not None:
            if column not in selected:
                raise ValueError(f"Intermediate table requires {column!r} for this selection")
            selected = selected[selected[column].isin(values)]
    return selected.copy()


def _select_distances(table_df, filters, *, layers, components, tensors, checkpoint_ids):
    if table_df is None:
        raise ValueError("model_centrality requires explicit pair_results")
    if any(value is not None for value in (layers, components, tensors, checkpoint_ids)):
        raise ValueError("Centrality selects experiment strata, not parameter/checkpoint labels")
    selected = table_df.copy()
    for column, values in filters.items():
        if values is None:
            continue
        if column == "backbone":
            required = {"backbone_a", "backbone_b"}
            if not required <= set(selected):
                raise ValueError("Centrality requires backbone_a/backbone_b columns")
            selected = selected[selected.backbone_a.isin(values) & selected.backbone_b.isin(values)]
        else:
            if column not in selected:
                raise ValueError(f"Pair table requires {column!r} for this selection")
            selected = selected[selected[column].isin(values)]
    return selected
