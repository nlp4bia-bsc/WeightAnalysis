"""Maintenance commands with explicit paths and no import-time execution."""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "check-grid": "grid",
    "export-review": "review",
    "snapshot": "snapshot",
    "benchmark": "benchmark",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        return importlib.import_module(
            "." + COMMANDS[argv[0]], package="weight_analysis.maintenance"
        ).main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    args, remaining = parser.parse_known_args(argv)
    return importlib.import_module(
        "." + COMMANDS[args.command], package="weight_analysis.maintenance"
    ).main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
