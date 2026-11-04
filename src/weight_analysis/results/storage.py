"""Explicit, atomic Parquet persistence for versioned analysis results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from .models import SCHEMA_VERSION, AnalysisResult, canonical_json, data_digest

METADATA_PREFIX = "weight_analysis."


def write_result(result: AnalysisResult, path: str | Path) -> Path:
    """Write a self-describing result atomically, preserving a prior valid file.

    The temporary file is read and checked before replacement. A failed write or
    interrupted replacement leaves the previous destination untouched and removes
    this call's temporary artifact.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not isinstance(result, AnalysisResult):
        raise TypeError("result must be an AnalysisResult")
    output_path = Path(path)
    fingerprint = data_digest(result.data)
    table = pa.Table.from_pandas(result.data, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    attributes = {
        "schema_version": SCHEMA_VERSION,
        "analysis": result.analysis,
        "provenance": canonical_json(result.provenance),
        "data_sha256": fingerprint,
    }
    metadata.update(
        {(METADATA_PREFIX + key).encode(): value.encode() for key, value in attributes.items()}
    )
    table = table.replace_schema_metadata(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.partial")
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        read_result(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def read_result(path: str | Path) -> AnalysisResult:
    """Read and validate an analysis result, including its content digest."""
    import pyarrow.parquet as pq

    input_path = Path(path)
    try:
        table = pq.read_table(input_path)
        metadata = table.schema.metadata or {}
        attributes = {
            key: metadata[(METADATA_PREFIX + key).encode()].decode()
            for key in ("schema_version", "analysis", "provenance", "data_sha256")
        }
    except (KeyError, UnicodeDecodeError) as error:
        raise ValueError(f"Missing or invalid Weight Analysis metadata in {input_path}") from error
    if attributes["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported result schema version: {attributes['schema_version']!r}")
    try:
        provenance = json.loads(attributes["provenance"])
        result = AnalysisResult(attributes["analysis"], table.to_pandas(), provenance)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid result provenance in {input_path}") from error
    if data_digest(result.data) != attributes["data_sha256"]:
        raise ValueError(f"Result content digest mismatch in {input_path}; recompute this unit")
    return result
