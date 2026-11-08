"""Portable compact MultiClinNER stages. Scheduler submission is a host-only adapter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from weight_analysis_scheduler.resources import check_storage, quota_snapshot, usage

from ..plan import atomic_json
from .plan import load_plan

WEIGHTS = "/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weights"
OUTPUT = "/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weigh_analysis_final"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--performance", default="results_aggregated.parquet")
    prepare.add_argument("--trained", default=WEIGHTS)
    prepare.add_argument("--originals", default=str(Path.home() / "ner_lab/TFG_Aziz/models"))
    prepare.add_argument("--layers-per-task", type=int, default=4)
    prepare.add_argument("--block-elements", type=int, default=131072)
    prepare.add_argument(
        "--spectral-mode", choices=("auto", "exact", "topk", "skip"), default="exact"
    )
    prepare.add_argument("--max-svd-elements", type=int, default=3_000_000)
    prepare.add_argument("--topk", type=int, default=32)
    prepare.add_argument("--curve-points", type=int, default=64)
    prepare.add_argument("--quantile-points", type=int, default=101)
    prepare.add_argument("--top-k", type=int, default=3)
    prepare.add_argument("--resident-gib", type=float, default=4)
    prepare.add_argument("--part-mib", type=float, default=64)
    prepare.add_argument("--checkpoint-fold", type=int, default=1)
    prepare.add_argument("--checkpoint-seed", type=int)
    prepare.add_argument("--protocol-id")
    prepare.add_argument("--evidence")
    prepare.add_argument("--quota-file")
    prepare.add_argument("--allow-unknown-quota", action="store_true")
    prepare.add_argument("--max-run-files", type=int, default=8000)
    prepare.add_argument("--reserve-gib", type=float, default=8)
    prepare.add_argument("--reserve-inodes", type=int, default=4000)
    worker = sub.add_parser("worker")
    worker.add_argument("--run-dir", required=True)
    choose = worker.add_mutually_exclusive_group(required=True)
    choose.add_argument("--index", type=int)
    choose.add_argument("--array-map")
    worker.add_argument("--array-index", type=int)
    worker.add_argument("--device", default="gpu")
    worker.add_argument("--budget-seconds", type=int, default=27000)
    final = sub.add_parser("finalize")
    final.add_argument("--run-dir", required=True)
    final.add_argument("--skip-intrinsic", action="store_true")
    figures = sub.add_parser("figures")
    figures.add_argument("--run-dir", required=True)
    figures.add_argument(
        "--metrics", nargs="+", default=["l2_distance", "cosine", "trace_similarity"]
    )
    figures.add_argument("--max-figures", type=int, default=2400)
    check = sub.add_parser("check")
    check.add_argument("--run-dir", required=True)
    check.add_argument("--verify", action="store_true")
    compact = sub.add_parser("compact")
    compact.add_argument("--run-dir", required=True)
    quota = sub.add_parser("quota")
    quota.add_argument("--run-dir", required=True)
    quota.add_argument("--output", required=True)
    quota.add_argument("--remaining-gib", type=float)
    quota.add_argument("--remaining-inodes", type=int)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .plan import prepare_run

        values = vars(args).copy()
        values.pop("command")
        values["resident_bytes"] = int(values.pop("resident_gib") * 1024**3)
        values["target_part_bytes"] = int(values.pop("part_mib") * 1024**2)
        values["storage_policy"] = dict(
            require_quota=not values.pop("allow_unknown_quota"),
            quota_file=str(Path(values.pop("quota_file")).resolve()) if args.quota_file else None,
            max_run_files=values.pop("max_run_files"),
            reserve_bytes=int(values.pop("reserve_gib") * 1024**3),
            reserve_inodes=values.pop("reserve_inodes"),
        )
        values.pop("quota_file", None)
        plan = prepare_run(**values)
        print("[PREPARED]", plan["run_id"], "tasks", len(plan["tasks"]), flush=True)
    elif args.command == "worker":
        from .worker import run_task

        index = args.index
        if args.array_map:
            path = Path(args.array_map).resolve()
            if path.parent != Path(args.run_dir).resolve() / "manifest/submissions":
                parser.error("Array map does not belong to this compact run")
            mapping = json.loads(path.read_text())
            if args.array_index is None or not 0 <= args.array_index < len(mapping):
                parser.error("Array ordinal outside submitted map")
            index = mapping[args.array_index]
        result = run_task(
            args.run_dir, index, device=args.device, budget_seconds=args.budget_seconds
        )
        return (
            0
            if result["state"] == "complete"
            else 75
            if result["state"] == "interrupted_resumable"
            else 1
        )
    elif args.command in ("finalize", "compact"):
        from .finalize import finalize_run

        result = finalize_run(args.run_dir, intrinsic=not getattr(args, "skip_intrinsic", False))
        return 0 if result["state"] == "complete" else 1
    elif args.command == "figures":
        from .figures import make_figures

        make_figures(args.run_dir, metrics=args.metrics, max_figures=args.max_figures)
    elif args.command == "quota":
        root = Path(args.run_dir).resolve()
        existing = root
        while not existing.exists():
            existing = existing.parent
        if args.remaining_gib is not None and args.remaining_inodes is not None:
            snapshot = dict(
                remaining_bytes=int(args.remaining_gib * 1024**3),
                remaining_inodes=args.remaining_inodes,
                source="operator_real_quota_reading",
                observed_at=time.time(),
                path=str(existing),
            )
        else:
            snapshot = quota_snapshot(existing)
        if snapshot is None:
            parser.error(
                "Quota command unsupported; provide BOTH real remaining GiB and inode counts"
            )
        atomic_json(args.output, snapshot)
        print("[QUOTA]", snapshot, flush=True)
    else:
        plan = load_plan(args.run_dir)
        report_path = Path(args.run_dir) / "manifest/datasets.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if args.verify and report:
            from ...provenance import file_sha256

            for item in report["files"]:
                if file_sha256(Path(args.run_dir) / item["path"]) != item["sha256"]:
                    raise ValueError("Parquet checksum mismatch: " + item["path"])
        print(
            json.dumps(
                dict(
                    run_id=plan["run_id"],
                    state=report.get("state", "prepared"),
                    retry=report.get("retry_array_indices", []),
                    usage=usage(args.run_dir),
                    resources=check_storage(args.run_dir, plan["config"]["storage_policy"]),
                ),
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
