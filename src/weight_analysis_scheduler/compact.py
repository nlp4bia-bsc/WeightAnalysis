"""Idempotent compact-run arrays/finalizer/figures, using host Python only."""

import fcntl
import json
from functools import wraps
from pathlib import Path

from .manifest import atomic_json, code_digest, stable_id
from .resources import check_storage
from .submission import _submit_checked


def _exclusive_submission(function):
    @wraps(function)
    def locked(run_dir, **kwargs):
        with (Path(run_dir).resolve() / "state/submission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return function(run_dir, **kwargs)

    return locked


def load_compact_plan(root):
    plan = json.loads((Path(root) / "manifest/plan.json").read_text())
    identity = {key: value for key, value in plan.items() if key != "run_id"}
    if plan.get("schema_version") != "multiclinner-packed-1" or stable_id(identity) != plan.get(
        "run_id"
    ):
        raise ValueError("Invalid compact run manifest")
    if plan["code_digest"] != code_digest():
        raise ValueError("Scientific code differs from frozen compact run")
    return plan


@_exclusive_submission
def submit_compact(
    run_dir,
    *,
    scripts,
    concurrency=1,
    partition=None,
    account=None,
    constraint=None,
    retry=False,
    recover=False,
    with_figures=True,
):
    root, scripts = Path(run_dir).resolve(), Path(scripts).resolve()
    plan = load_compact_plan(root)
    if concurrency < 1:
        raise ValueError("Positive concurrency required")
    check_storage(root, plan["config"]["storage_policy"], launch=True)
    directory = root / "manifest/submissions"
    directory.mkdir(exist_ok=True)
    indices = list(range(len(plan["tasks"])))
    suffix = "initial"
    if retry:
        report = json.loads((root / "manifest/datasets.json").read_text())
        if report["run_id"] != plan["run_id"]:
            raise ValueError("Retry report belongs to another run")
        indices = report["retry_array_indices"]
        suffix = "retry-" + stable_id(report)[:16]
    if not indices:
        print("[COMPLETE] No retry indices; no jobs submitted", flush=True)
        return None
    ledger = directory / (suffix + ".json")
    record = (
        json.loads(ledger.read_text())
        if ledger.exists()
        else dict(
            run_id=plan["run_id"],
            indices=indices,
            gpu_job_ids=[],
            finalize_job_id=None,
            figures_job_id=None,
            pending=None,
            accepted=[],
        )
    )
    if record["run_id"] != plan["run_id"] or record["indices"] != indices:
        raise ValueError("Submission ledger differs from run")
    markers = list(directory.glob("unparsed-*.json"))
    if recover and markers:
        from .recovery import _verify_job
        from .submission import parse_job_id

        if len(markers) != 1:
            raise ValueError(
                "Recovery needs exactly one receipt matching a durable submission intent"
            )
        marker = markers[0]
        receipt = json.loads(marker.read_text())
        pending = record["pending"] or next(
            (item for item in record.get("accepted", []) if item["command"] == receipt["command"]),
            None,
        )
        if pending is None:
            raise ValueError("No durable intent or accepted receipt matches quarantine")
        if receipt["run_id"] != plan["run_id"] or receipt["command"] != pending["command"]:
            raise ValueError("Receipt differs from pending intent")
        job_id = parse_job_id(receipt["stdout"], receipt["stderr"])
        if pending.get("job_id") and pending["job_id"] != job_id:
            raise ValueError("Recovered ID differs from previously accepted receipt")
        command = pending["command"]
        script = scripts / (
            "gpu_array.sbatch"
            if pending["kind"] == "gpu"
            else "finalize.sbatch"
            if pending["kind"] == "finalize"
            else "figures.sbatch"
        )
        evidence = _verify_job(
            job_id,
            command,
            script,
            root,
            array=pending["kind"] == "gpu",
            project_root=scripts.parents[2],
        )
        if pending["kind"] == "gpu" and job_id not in record["gpu_job_ids"]:
            record["gpu_job_ids"].append(job_id)
        elif pending["kind"] != "gpu":
            record[pending["kind"] + "_job_id"] = job_id
        if not pending.get("job_id"):
            record.setdefault("accepted", []).append(dict(pending, job_id=job_id))
        record["pending"] = None
        atomic_json(ledger, record)
        resolved = directory / "resolved" / marker.name
        atomic_json(
            resolved.with_suffix(".verification.json"),
            dict(job_id=job_id, scontrol_stdout=evidence),
        )
        marker.replace(resolved)
        markers = []
    if markers or record["pending"]:
        raise ValueError("Unresolved accepted/uncertain submission; inspect queue before --recover")
    options = []
    for flag, value in (
        ("--partition", partition),
        ("--account", account),
        ("--constraint", constraint),
    ):
        if value:
            if value.startswith("-"):
                raise ValueError("Invalid scheduler option")
            options += [flag, value]

    def send(command, kind):
        # Persist intent BEFORE calling sbatch; a killed host process cannot silently resubmit.
        record["pending"] = dict(command=command, kind=kind)
        atomic_json(ledger, record)
        try:
            job = _submit_checked(command, root / "manifest", plan["run_id"])
        except Exception:
            # Only a known rejection (CalledProcessError cause) proves no accepted job.
            import subprocess
            import sys

            exc = sys.exc_info()[1]
            if isinstance(
                getattr(exc, "__cause__", None), (subprocess.CalledProcessError, FileNotFoundError)
            ):
                record["pending"] = None
                atomic_json(ledger, record)
            raise
        if kind == "gpu":
            record["gpu_job_ids"].append(job)
        else:
            record[kind + "_job_id"] = job
        record.setdefault("accepted", []).append(dict(command=command, kind=kind, job_id=job))
        record["pending"] = None
        atomic_json(ledger, record)
        print("[SUBMITTED]", kind, job, flush=True)

    for batch_index, start in enumerate(range(0, len(indices), 500)):
        if batch_index < len(record["gpu_job_ids"]):
            continue
        batch = indices[start : start + 500]
        mapping = directory / (suffix + "-map-" + str(batch_index) + ".json")
        atomic_json(mapping, batch)
        send(
            [
                "sbatch",
                "--parsable",
                "--chdir",
                str(scripts.parents[2]),
                *options,
                "--array",
                "0-{}%{}".format(len(batch) - 1, concurrency),
                "--output",
                str(root / "logs/gpu-%A_%a.out"),
                "--error",
                str(root / "logs/gpu-%A_%a.err"),
                str(scripts / "gpu_array.sbatch"),
                str(root),
                str(mapping),
            ],
            "gpu",
        )
    cpu = ["--account", account] if account else []
    if not record["finalize_job_id"]:
        send(
            [
                "sbatch",
                "--parsable",
                "--chdir",
                str(scripts.parents[2]),
                *cpu,
                "--dependency",
                "afterany:" + ":".join(record["gpu_job_ids"]),
                "--output",
                str(root / "logs/finalize-%j.out"),
                "--error",
                str(root / "logs/finalize-%j.err"),
                str(scripts / "finalize.sbatch"),
                str(root),
            ],
            "finalize",
        )
    if with_figures and not record["figures_job_id"]:
        send(
            [
                "sbatch",
                "--parsable",
                "--chdir",
                str(scripts.parents[2]),
                *cpu,
                "--dependency",
                "afterany:" + record["finalize_job_id"],
                "--output",
                str(root / "logs/figures-%j.out"),
                "--error",
                str(root / "logs/figures-%j.err"),
                str(scripts / "figures.sbatch"),
                str(root),
            ],
            "figures",
        )
    return record
