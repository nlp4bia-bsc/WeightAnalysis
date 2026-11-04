"""Result contracts, explicit persistence, and deterministic unit assembly."""

from .merge import merge_results
from .models import SCHEMA_VERSION, AnalysisResult
from .storage import read_result, write_result

__all__ = ["SCHEMA_VERSION", "AnalysisResult", "merge_results", "read_result", "write_result"]
