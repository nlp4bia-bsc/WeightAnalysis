"""One serial GPU worker per array element, resumable at independent scientific units."""

from __future__ import annotations

import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..analyses.shared_adaptation import analyze_shared_adaptation
from ..comparisons.performance import _compatible
from ..configuration import MetricOptions
from ..core import WeightReader
from ..core.catalog import stable_id
from ..core.geometry_schema import SHIFT_INVARIANT, ParameterSelection
from ..core.reporting import RunReporter
from ..devices import resolve_device
from ..execution.comparative_units import ComparativeContext
from ..provenance import file_sha256
from ..results.comparative import (
    ComparativeResult,
    event_table,
    issue_table,
    merge_comparative,
    read_dataset,
    write_dataset,
)
from .inventory import restore_catalog, verify_signature
from .plan import atomic_json, load_plan
from .source import runtime_versions


class TimeBudgetReached(BaseException):
    """A deliberate checkpoint boundary, distinct from an ordinary failed experiment."""


class WorkerReporter(RunReporter):
    def __init__(self, *, deadline, **kwargs):
        super().__init__(**kwargs)
        self.deadline, self.stop_requested = deadline, False

    def emit(self, status, unit_id, message, **context):
        if (
            status == "started"
            and context.get("stage") != "persistence"
            and (self.stop_requested or time.monotonic() >= self.deadline)
        ):
            raise TimeBudgetReached()
        return super().emit(status, unit_id, message, **context)


def _collect(target, unit):
    for name, rows in unit.items():
        if name == "fingerprints":
            name = "intrinsic_inputs"
        target.setdefault(name, []).extend(rows)


def compute_work(work, catalog, ctx, config, selector):
    selected = [catalog.by_id[aid] for aid in work["members"]]
    if work["kind"] == "shared":
        return analyze_shared_adaptation(
            checkpoints=catalog,
            axis=work["axis"],
            member_ids=work["members"],
            references=ctx.references,
            parameters=selector,
            options=ctx.options,
            device=ctx.device,
            reporter=ctx.reporter,
            cache_dir=ctx.cache_dir,
            block_elements=ctx.block_elements,
            allow_unverified_origin=ctx.allow_unverified,
            _context=ctx,
        )
    collected = {}
    if work["kind"] == "reference":
        entry = selected[0]
        for key in ctx.keys(entry):
            _collect(
                collected,
                ctx.vector(
                    entry,
                    None,
                    key,
                    "delta",
                    config["drift_metrics"],
                    subject_id=entry.artifact_id,
                    kind="drift",
                ),
            )
            with WeightReader(entry.checkpoint.path) as reader:
                matrix = len(reader.shape_dtype(key)[0]) == 2
            if matrix:
                for source in config["sources"]:
                    _collect(
                        collected, ctx.fingerprint(entry, key, source, config["spectral_metrics"])
                    )
    elif work["kind"] == "pair":
        a, b = selected
        ctx.verify(a)
        ctx.verify(b)
        _compatible(a, b, ctx)
        keys = ctx.keys(a)
        if set(keys) != set(ctx.keys(b)):
            raise ValueError("Selected pair coordinates differ")
        cid = stable_id({"checkpoints": sorted(work["members"])})
        for key in keys:
            for source in config["sources"]:
                names = [
                    m
                    for m in config["pair_metrics"]
                    if source == "final" or m not in SHIFT_INVARIANT
                ]
                if names:
                    _collect(collected, ctx.vector(a, b, key, source, names, subject_id=cid))
        collected["comparisons"] = [
            dict(
                comparison_id=cid,
                artifact_a=a.artifact_id,
                artifact_b=b.artifact_id,
                comparison_type=work["comparison_type"],
                domain="direct",
            )
        ]
    else:
        raise ValueError("Unknown work kind")
    tables = {name: pd.DataFrame(rows) for name, rows in collected.items()}
    tables.update(
        checkpoints=catalog.table[catalog.table.artifact_id.isin(work["members"])],
        references=pd.DataFrame(ctx.reference_rows.values()),
        issues=issue_table(ctx.reporter),
        events=event_table(ctx.reporter),
    )
    return merge_comparative(
        [
            ComparativeResult(
                tables,
                {"work": work, "computation": ctx.definition, "parameters": asdict(selector)},
            )
        ]
    )


def run_task(run_dir, index, *, device="gpu", budget_seconds=27_000):
    root = Path(run_dir).expanduser().resolve()
    plan = load_plan(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("Array task index outside manifest")
    task = plan["tasks"][index]
    task_root = root / "tasks" / f"{index:05d}-{task['backbone']}"
    reporter = WorkerReporter(
        deadline=time.monotonic() + budget_seconds, log_path=task_root / "events.jsonl"
    )
    previous = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, lambda *_: setattr(reporter, "stop_requested", True))
    status = {
        "run_id": plan["run_id"],
        "task_id": task["task_id"],
        "index": index,
        "state": "running",
        "completed": 0,
        "failed": 0,
        "runtime_versions": runtime_versions(),
    }
    task_root.mkdir(parents=True, exist_ok=True)
    # No simultaneous writers for a retried array index.
    import fcntl

    lock = (task_root / "worker.lock").open("a")
    acquired = False
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        resolve_device(device)  # fail loudly before claiming any GPU result
        records = [row for row in plan["checkpoints"] if row["backbone"] == task["backbone"]]
        catalog, refs = restore_catalog(records)
        selector = ParameterSelection(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in task["selector"].items()}
        )
        ctx = ComparativeContext(
            options=MetricOptions(**plan["config"]["options"]),
            device=device,
            parameters=selector,
            references=refs,
            reporter=reporter,
            cache_dir=task_root / "cache",
            block_elements=plan["config"]["block_elements"],
            allow_unverified=plan["config"]["allow_unverified_origin"],
        )
        reporter.emit(
            "summary",
            str(index),
            "Execution environment",
            device=ctx.device,
            hardware=ctx.definition["hardware"],
            versions=status["runtime_versions"],
        )
        for number, work in enumerate(task["work"], 1):
            reporter.events.clear()
            active = [row for row in records if row["artifact_id"] in work["members"]]

            def guard():
                for row in active:
                    verify_signature(row["input_signature"])
                    verify_signature(row["reference_signature"])

            ctx.input_guard = guard
            directory = task_root / "work" / work["comparison_type"] / work["work_id"]
            marker = directory / "completion.json"
            if marker.is_file():
                try:
                    completed = json.loads(marker.read_text())
                    if (
                        completed["state"] == "complete"
                        and completed["run_id"] == plan["run_id"]
                        and completed["task_id"] == task["task_id"]
                    ):
                        guard()
                        if (
                            file_sha256(directory / "dataset" / "manifest.json")
                            != completed["manifest_sha256"]
                        ):
                            raise ValueError("Completed dataset manifest changed")
                        read_dataset(directory / "dataset")
                        status["completed"] += 1
                        reporter.emit(
                            "cached",
                            work["work_id"],
                            "Validated completed work",
                            progress=f"{number}/{len(task['work'])}",
                        )
                        continue
                except Exception as exc:
                    reporter.emit(
                        "warning", work["work_id"], "Invalid result; recomputing", exception=exc
                    )
            reporter.emit(
                "progress",
                work["work_id"],
                work["comparison_type"],
                progress=f"{number}/{len(task['work'])}",
                device=ctx.device,
            )
            result = reporter.attempt(
                work["work_id"],
                lambda: compute_work(work, catalog, ctx, plan["config"], selector),
                stage=work["comparison_type"],
            )
            if result is not None:
                result = ComparativeResult(
                    result.tables,
                    {
                        **result.provenance,
                        "run_id": plan["run_id"],
                        "task_id": task["task_id"],
                        "work": work,
                    },
                )
                written = reporter.attempt(
                    work["work_id"] + ":write",
                    lambda: write_dataset(result, directory / "dataset"),
                    stage="persistence",
                )
                complete = result.complete and written is not None
                if written is not None:
                    atomic_json(
                        marker,
                        {
                            "state": "complete" if complete else "partial",
                            "run_id": plan["run_id"],
                            "task_id": task["task_id"],
                            "work_id": work["work_id"],
                            "manifest_sha256": file_sha256(written),
                        },
                    )
                status["completed" if complete else "failed"] += 1
            else:
                status["failed"] += 1
            ctx.unit_cache.clear()  # scalar/row cache is on disk; bound host memory per work item
            atomic_json(task_root / "status.json", status)
        status["state"] = "complete" if not status["failed"] else "partial"
    except TimeBudgetReached:
        status["state"] = "interrupted_resumable"
        reporter.emit(
            "warning", str(index), "Time budget/signal reached; rerun this array index to resume"
        )
    except Exception as exc:
        status["state"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
        reporter.emit("failed", str(index), str(exc), exception=exc)
    finally:
        signal.signal(signal.SIGUSR1, previous)
        if acquired:
            atomic_json(task_root / "status.json", status)
        lock.close()
    return status
