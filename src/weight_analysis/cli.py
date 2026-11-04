"""A thin JSON entry point calling the same public Python analysis function."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Run a JSON declaration or list analysis names without importing tensor libraries."""
    parser = argparse.ArgumentParser(prog="weight-analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("analyses", help="List independent analysis names")
    run = commands.add_parser("run", help="Call analyze with named arguments from JSON")
    run.add_argument("configuration", type=Path)
    args = parser.parse_args(argv)
    if args.command == "analyses":
        names = (
            "pretraining_drift",
            "spectral_fingerprints",
            "cross_language",
            "cross_entity",
            "language_entity",
            "cross_backbone",
            "language_entity_decomposition",
            "model_centrality",
        )
        print("\n".join(names))
        return
    from .api import analyze
    from .configuration import MetricOptions
    from .core import CheckpointSpec
    from .results import read_result, write_result

    parameters = json.loads(args.configuration.read_text(encoding="utf8"))
    if not isinstance(parameters, dict):
        parser.error("The JSON document must contain an object of named analyze arguments")
    output = parameters.pop("output", None)
    parameters["checkpoints"] = [
        CheckpointSpec(**item) for item in parameters.get("checkpoints", [])
    ]
    if "options" in parameters:
        parameters["options"] = MetricOptions(**parameters["options"])
    for name in ("spectral_results", "drift_results", "pair_results"):
        if name in parameters:
            parameters[name] = read_result(parameters[name])
    result = analyze(**parameters)
    if output is not None:
        write_result(result, output)
    print(
        json.dumps(
            {
                "analysis": result.analysis,
                "rows": len(result.data),
                "unit_id": result.unit_id,
                "output": output,
            },
            sort_keys=True,
        )
    )
