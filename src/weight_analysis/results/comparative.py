"""Normalized comparative studies, deterministic assembly and partitioned Parquet datasets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pandas as pd

from ..core.catalog import stable_id
from ..provenance import file_sha256
from .models import AnalysisResult, data_digest, row_keys
from .storage import read_result, write_result

TABLE_KEYS = {
    "checkpoints": "artifact_id",
    "references": "reference_id",
    "performance": "score_id",
    "rankings": "ranking_id",
    "comparisons": "comparison_id",
    "selection_pairs": "selection_pair_id",
    "groups": "group_id",
    "group_members": "membership_id",
    "geometry_detailed": "observation_id",
    "geometry_aggregated": "observation_id",
    "sufficient_statistics": "statistic_id",
    "profile_points": "point_id",
    "issues": "issue_id",
    "associations": "association_id",
    "histograms": "histogram_id",
    "exclusions": "exclusion_id",
    "shared_statistics": "shared_statistic_id",
    "events": "event_id",
}


@dataclass(frozen=True)
class ComparativeResult:
    tables: dict[str, pd.DataFrame]
    provenance: dict

    def __post_init__(self):
        object.__setattr__(
            self, "tables", {name: table.copy(deep=True) for name, table in self.tables.items()}
        )
        object.__setattr__(self, "provenance", json.loads(json.dumps(self.provenance, default=str)))

    @property
    def complete(self):
        return self.tables.get("issues", pd.DataFrame()).empty


def issue_table(reporter):
    rows = []
    for event in reporter.events:
        if event["status"] == "failed":
            row = dict(event)
            row["issue_id"] = stable_id(row)
            rows.append(row)
    return pd.DataFrame(rows)


def event_table(reporter):
    """Preserve progress, cache hits and incident context alongside numerical tables."""
    return pd.DataFrame([{**event, "event_id": stable_id(event)} for event in reporter.events])


def merge_comparative(results):
    """Deduplicate exact observations and reject conflicting retries without averaging."""
    results = list(results)
    tables = {}
    for name in sorted({name for result in results for name in result.tables}):
        frames = [
            result.tables[name]
            for result in results
            if name in result.tables and not result.tables[name].empty
        ]
        if not frames:
            tables[name] = pd.DataFrame()
            continue
        frame = pd.concat(frames, ignore_index=True)
        key = TABLE_KEYS.get(name)
        if name == "sufficient_statistics" and key in frame:
            merged = []
            for _, group in frame.groupby(key, sort=True):
                if len(set(row_keys(group.drop(columns="metric_names")))) > 1:
                    raise ValueError("Conflicting sufficient statistics")
                row = group.iloc[0].to_dict()
                row["metric_names"] = ",".join(
                    sorted({m for names in group.metric_names for m in names.split(",")})
                )
                merged.append(row)
            frame = pd.DataFrame(merged)
        if key and key in frame:
            for _, group in frame.groupby(key, dropna=False, sort=True):
                if len(set(row_keys(group))) > 1:
                    raise ValueError(f"Conflicting retry in {name} for {group.iloc[0][key]}")
            frame = frame.drop_duplicates(key)
        keys = row_keys(frame)
        order = sorted(range(len(frame)), key=lambda i: keys[i])
        tables[name] = frame.iloc[order].reset_index(drop=True)
    provenance = (
        results[0].provenance
        if len(results) == 1
        else {
            "schema_version": "comparative-1",
            "units": [
                result.provenance
                for result in sorted(results, key=lambda r: stable_id(r.provenance))
            ],
        }
    )
    return ComparativeResult(tables, provenance)


def write_dataset(
    result: ComparativeResult,
    output_dir,
    *,
    partition_by=("language", "entity", "backbone", "comparison_type"),
):
    """Write atomic result files and commit an integrity-checked manifest last."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "comparative-1", "provenance": result.provenance, "files": []}
    for name, frame in sorted(result.tables.items()):
        if not name.isidentifier():
            raise ValueError("Dataset table names must be Python identifiers")
        columns = [column for column in partition_by if column in frame]
        groups = (
            frame.groupby(columns, dropna=False, sort=True)
            if columns and len(frame)
            else [((), frame)]
        )
        for keys, part in groups:
            keys = keys if isinstance(keys, tuple) else (keys,)
            partitions = {
                column: None if pd.isna(value) else str(value)
                for column, value in zip(columns, keys)
            }
            directory = root / name
            for column, value in partitions.items():
                directory = (
                    directory
                    / f"{column}={quote(value if value is not None else '__null__', safe='')}"
                )
            identity = stable_id([data_digest(part), result.provenance])
            path = directory / f"part-{identity}.parquet"
            write_result(AnalysisResult(name, part.reset_index(drop=True), result.provenance), path)
            manifest["files"].append(
                {
                    "table": name,
                    "path": str(path.relative_to(root)),
                    "partitions": partitions,
                    "sha256": file_sha256(path),
                    "rows": len(part),
                }
            )
    temporary = root / f".manifest-{uuid4().hex}.partial"
    try:
        temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf8")
        os.replace(temporary, root / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)
    return root / "manifest.json"


def read_dataset(path, *, filters=None):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if manifest["schema_version"] != "comparative-1":
        raise ValueError("Unsupported comparative schema")
    tables = {}
    for entry in manifest["files"]:
        if filters and any(
            key in entry["partitions"] and str(value) != str(entry["partitions"][key])
            for key, value in filters.items()
        ):
            continue
        file = (root / entry["path"]).resolve()
        if not file.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes dataset")
        if file_sha256(file) != entry["sha256"]:
            raise ValueError(f"Dataset file content changed: {entry['path']}")
        restored = read_result(file)
        if len(restored.data) != entry["rows"]:
            raise ValueError("Dataset row count mismatch")
        tables.setdefault(entry["table"], []).append(restored.data)
    return ComparativeResult(
        {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()},
        manifest["provenance"],
    )
