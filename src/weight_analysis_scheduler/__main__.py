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
    parser.add_argument("--no-figures", action="store_true", help="Compact runs: do not submit the figure job")
    parser.add_argument(
        "--recover", action="store_true",
        help="Verify quarantined initial submissions with scontrol, then submit missing jobs only",
    )
    args = parser.parse_args(argv)
    with_figures = not vars(args).pop("no_figures")
    recover = vars(args).pop("recover")
    if recover and args.retry:
        parser.error("--recover handles initial submissions; cannot combine with --retry")
    try:
        if (Path(args.run_dir) / "manifest/plan.json").is_file():
            from .compact import submit_compact

            scripts = Path(args.scripts)
            if scripts.name != "compact":
                args.scripts = str(scripts / "compact")
            submit_compact(**vars(args), recover=recover, with_figures=with_figures)
            return 0
        if recover:
            from .recovery import recover_submissions

            recover_submissions(args.run_dir, scripts=args.scripts)
        submit(**vars(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, "[SUBMISSION ERROR] {}\n".format(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
