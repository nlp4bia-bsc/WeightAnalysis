"""Submit frozen experiments from the login host, without the compute container."""

import argparse
from pathlib import Path

from .submission import submit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--scripts", default=str(Path(__file__).resolve().parents[2] / "scripts/picasso")
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--partition")
    parser.add_argument("--account")
    parser.add_argument("--constraint")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args(argv)
    try:
        submit(**vars(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, "[SUBMISSION ERROR] {}\n".format(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
