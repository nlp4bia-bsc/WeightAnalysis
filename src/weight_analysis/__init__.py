"""Independent checkpoint analyses with explicit computation and persistence."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "3.0.0"

_EXPORTS = {
    "analyze": ".api",
    "CheckpointSpec": ".core",
    "AnalysisResult": ".results",
    "read_result": ".results",
    "write_result": ".results",
    "merge_results": ".results",
    "build_catalog": ".core.catalog",
    "CheckpointEvidence": ".core.catalog",
    "CheckpointCatalog": ".core.catalog",
    "CatalogEntry": ".core.catalog",
    "input_digest": ".core.catalog",
    "ReferenceSpec": ".core.catalog",
    "ParameterSelection": ".core.geometry_schema",
    "select_performance_pairs": ".comparisons.selection",
    "select_axis_pairs": ".comparisons.selection",
    "SelectionResult": ".comparisons.selection",
    "compare_selected": ".comparisons.performance",
    "analyze_shared_adaptation": ".analyses.shared_adaptation",
    "aggregate_geometry": ".statistics.aggregation",
    "comparison_view": ".statistics.comparative",
    "member_performance_view": ".statistics.comparative",
    "associate_f1": ".statistics.comparative",
    "associate_member_f1": ".statistics.comparative",
    "RunReporter": ".core.reporting",
    "ComparativeResult": ".results.comparative",
    "merge_comparative": ".results.comparative",
    "write_dataset": ".results.comparative",
    "read_dataset": ".results.comparative",
}
__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
