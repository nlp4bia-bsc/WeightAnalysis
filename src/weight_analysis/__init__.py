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
}
__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
