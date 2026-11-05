"""Portable CLI for audit, preparation, array workers and checkpoint-free postprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plan import prepare_run
from .source import write_source_audit
from .submission import submit as submit

DEFAULT_WEIGHTS = "/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weights"
DEFAULT_OUTPUT = "/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weigh_analysis_final"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--performance", default="results_aggregated.parquet")
    audit.add_argument("--output", default="local_results/ner_source")
    audit.add_argument("--top-k", type=int, default=3)
    audit.add_argument("--protocol-id")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--performance", default="results_aggregated.parquet")
    prepare.add_argument("--trained", default=DEFAULT_WEIGHTS)
    prepare.add_argument("--originals", default=str(Path.home() / "ner_lab/TFG_Aziz/models"))
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--layers-per-task", type=int, default=4)
    prepare.add_argument("--block-elements", type=int, default=1_000_000)
    prepare.add_argument("--checkpoint-fold", type=int, default=1)
    prepare.add_argument("--checkpoint-seed", type=int)
    prepare.add_argument("--evidence")
    prepare.add_argument("--allow-unverified-origin", action="store_true")
    prepare.add_argument("--protocol-id")
    prepare.add_argument("--top-k", type=int, default=3)
    prepare.add_argument(
        "--spectral-mode", choices=["auto", "exact", "topk", "skip"], default="auto"
    )
    prepare.add_argument("--max-svd-elements", type=int, default=3_000_000)
    worker = sub.add_parser("worker")
    worker.add_argument("--run-dir", required=True)
    selection = worker.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--array-map")
    worker.add_argument("--array-index", type=int)
    worker.add_argument("--device", default="gpu")
    worker.add_argument("--budget-seconds", type=int, default=27_000)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--skip-intrinsic", action="store_true")
    figures = sub.add_parser("figures")
    figures.add_argument("--run-dir", required=True)
    submitter = sub.add_parser("submit")
    submitter.add_argument("--run-dir", required=True)
    submitter.add_argument(
        "--scripts", default=str(Path(__file__).resolve().parents[3] / "scripts/picasso")
    )
    submitter.add_argument("--concurrency", type=int, default=1)
    submitter.add_argument("--partition")
    submitter.add_argument("--account")
    submitter.add_argument("--constraint")
    submitter.add_argument("--retry", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "audit":
        scores, report = write_source_audit(
            args.performance, args.output, k=args.top_k, protocol_id=args.protocol_id
        )
        print(
            json.dumps(
                {
                    "rows": len(scores),
                    "hash": report["source_sha256"],
                    "missing_cells": report["missing_cells"],
                },
                indent=2,
            )
        )
    elif args.command == "prepare":
        plan = prepare_run(
            performance=args.performance,
            trained=args.trained,
            originals=args.originals,
            run_dir=args.run_dir,
            layers_per_task=args.layers_per_task,
            block_elements=args.block_elements,
            evidence_path=args.evidence,
            allow_unverified_origin=args.allow_unverified_origin,
            fold=args.checkpoint_fold,
            seed=args.checkpoint_seed,
            protocol_id=args.protocol_id,
            top_k=args.top_k,
            spectral_mode=args.spectral_mode,
            max_svd_elements=args.max_svd_elements,
        )
        print("[PREPARED]", plan["run_id"], "tasks", len(plan["tasks"]))
    elif args.command == "worker":
        from .worker import run_task

        index = args.index
        if args.array_map:
            if args.array_index is None:
                parser.error("--array-map requires --array-index")
            mapping_path = Path(args.array_map).resolve()
            if not mapping_path.is_relative_to(Path(args.run_dir).resolve() / "submissions"):
                parser.error("Array map must belong to this run")
            mapping = json.loads(mapping_path.read_text())
            if not 0 <= args.array_index < len(mapping):
                parser.error("Array ordinal out of bounds")
            index = mapping[args.array_index]
        status = run_task(
            args.run_dir, index, device=args.device, budget_seconds=args.budget_seconds
        )
        return (
            0
            if status["state"] == "complete"
            else 75
            if status["state"] == "interrupted_resumable"
            else 1
        )
    elif args.command == "finalize":
        from .exports import finalize_run

        status = finalize_run(args.run_dir, intrinsic=not args.skip_intrinsic)
        return 0 if status["state"] == "complete" else 1
    elif args.command == "figures":
        from .figures import make_figures

        make_figures(args.run_dir)
    else:
        submit(
            args.run_dir,
            scripts=args.scripts,
            concurrency=args.concurrency,
            partition=args.partition,
            account=args.account,
            constraint=args.constraint,
            retry=args.retry,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
