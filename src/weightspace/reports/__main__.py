"""Run historical report definitions explicitly, with user-selected input and output paths."""

from __future__ import annotations

import argparse
import importlib
import os

REPORTS = {
    "correlate": "correlation_summary",
    "validate": "validation",
    "audit": "data_audit",
    "layers": "layer_correlations",
    "sensitivity": "backbone_sensitivity",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", choices=REPORTS)
    parser.add_argument("--data-dir", default=os.environ.get("WS_DATA"))
    parser.add_argument("--performance", default=os.environ.get("WS_F1"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input", help="Rankings CSV for leave-one-backbone-out sensitivity")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--permutations", type=int)
    args = parser.parse_args(argv)
    if args.report != "sensitivity" and not (args.data_dir and args.performance):
        parser.error("--data-dir and --performance are required (or WS_DATA/WS_F1)")
    options = dict(
        data_dir=args.data_dir,
        performance=args.performance,
        output_dir=args.output_dir,
        input_path=args.input,
    )
    if args.seed is not None:
        options["seed"] = args.seed
    if args.permutations is not None:
        if args.permutations < 1:
            parser.error("--permutations must be positive")
        options["permutations"] = args.permutations
    importlib.import_module("." + REPORTS[args.report], package="weightspace.reports").run(
        **options
    )


if __name__ == "__main__":
    main()
