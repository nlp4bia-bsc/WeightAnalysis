"""Typed analysis results and canonical representations of their contents."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"


def canonical_json(value: Any) -> str:
    """Serialize JSON metadata deterministically, rejecting nonfinite values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cell_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_cell_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Result dictionary cells require string keys")
        return {key: _cell_value(item) for key, item in value.items()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ValueError(
                "Results contain infinite values; represent undefined metrics with NaN"
            )
        return int(number) if number.is_integer() else number
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, (timedelta, pd.Timedelta)):
        return {"__timedelta_seconds__": value.total_seconds()}
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported result cell type: {type(value).__name__}")


def row_keys(data_df: pd.DataFrame) -> list[str]:
    """Return value-based row keys in the table's current order."""
    columns = sorted(data_df.columns)
    return [
        canonical_json([_cell_value(value) for value in row])
        for row in data_df.loc[:, columns].itertuples(index=False, name=None)
    ]


def canonical_rows(data_df: pd.DataFrame) -> list[str]:
    """Return sorted canonical rows, preserving repeated observations."""
    return sorted(row_keys(data_df))


def data_digest(data_df: pd.DataFrame) -> str:
    """Hash the table's values independently of row, column and index order."""
    payload = canonical_json({"columns": sorted(data_df.columns), "rows": canonical_rows(data_df)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnalysisResult:
    """A table and explicit provenance, without automatic persistence.

    The dataclass freezes field assignment; the DataFrame remains a mutable pandas
    object. Inputs are copied on construction. Copy before changing a result used
    for persistence or reproducibility checks.
    """

    analysis: str
    data: pd.DataFrame
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.analysis, str) or not self.analysis.strip():
            raise ValueError("analysis must be a nonempty string")
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if self.data.columns.duplicated().any():
            raise ValueError("Result columns must be unique")
        if not all(isinstance(column, str) for column in self.data.columns):
            raise TypeError("Result column names must be strings")
        if len(self.data) and not len(self.data.columns):
            raise ValueError("A populated result requires at least one named column")
        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a JSON-compatible dictionary")
        provenance = json.loads(canonical_json(self.provenance))
        unit_id = provenance.get("unit_id")
        if unit_id is not None and (not isinstance(unit_id, str) or not unit_id):
            raise ValueError("provenance['unit_id'] must be a nonempty string when present")
        object.__setattr__(self, "data", self.data.copy(deep=True))
        object.__setattr__(self, "provenance", provenance)

    @property
    def unit_id(self) -> str | None:
        """Return the calculation identity supplied by the caller, when present."""
        return self.provenance.get("unit_id")
