"""Checked, resumable scheduler submissions kept outside scientific computation."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from uuid import uuid4

from .plan import atomic_json, load_plan


def parse_job_id(output):
    identifiers = []
    for line in output.splitlines():
        match = re.fullmatch(
            r"\s*(?:Submitted batch job\s+)?([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", line
        )
        if match:
            identifiers.append(match.group(1))
    if len(identifiers) != 1:
        raise ValueError("Successful sbatch output has no unique job ID")
    return identifiers[0]


def _submit_checked(command, root, run_id):
    try:
        response = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"sbatch rejected submission: {exc.stderr or exc.stdout or str(exc)}"
        ) from exc
    try:
        return parse_job_id(response.stdout)
    except ValueError as exc:
        path = root / "submissions" / f"unparsed-{uuid4().hex}.json"
        atomic_json(
            path,
            dict(run_id=run_id, command=command, stdout=response.stdout, stderr=response.stderr),
        )
        raise RuntimeError(
            f"Unrecognized successful sbatch response; recorded in {path}. Check the queue before resolving the marker; do not resubmit blindly."
        ) from exc


def submit(
    run_dir, *, scripts, concurrency=1, partition=None, account=None, constraint=None, retry=False
):
    root = Path(run_dir).expanduser().resolve()
    plan = load_plan(root)
    if concurrency < 1:
        raise ValueError("Positive array concurrency required")
    unresolved = list((root / "submissions").glob("unparsed-*.json"))
    if unresolved:
        raise RuntimeError(
            f"Unrecognized previous sbatch output remains unresolved: {unresolved[0]}"
        )
    indices = list(range(len(plan["tasks"])))
    if retry:
        report = json.loads((root / "derived" / "finalization.json").read_text())
        if report["run_id"] != plan["run_id"]:
            raise ValueError("Retry report from another run")
        indices = report["retry_array_indices"]
    if not indices:
        raise ValueError("No pending tasks; inspect inventory/finalization before submitting")
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ledger = (
        root / "submissions" / (f"retry-{uuid4().hex}.json" if retry else "initial_submission.json")
    )
    record = (
        json.loads(ledger.read_text())
        if ledger.is_file()
        else dict(run_id=plan["run_id"], gpu_job_ids=[], finalize_job_id=None, indices=indices)
    )
    if record["run_id"] != plan["run_id"] or record["indices"] != indices:
        raise ValueError("Submission ledger differs from frozen run")
    if record["finalize_job_id"]:
        print("[SUBMITTED EARLIER]", record, flush=True)
        return record
    extra = []
    for flag, value in (
        ("--partition", partition),
        ("--account", account),
        ("--constraint", constraint),
    ):
        if value:
            if value.startswith("-"):
                raise ValueError("Invalid SLURM option value")
            extra += [flag, value]
    job_ids = record["gpu_job_ids"]
    for batch_index, offset in enumerate(range(0, len(indices), 500)):
        if batch_index < len(job_ids):
            continue
        batch = indices[offset : offset + 500]
        mapping = root / "submissions" / f"map-{uuid4().hex}.json"
        atomic_json(mapping, batch)
        command = [
            "sbatch",
            "--parsable",
            *extra,
            "--array",
            f"0-{len(batch) - 1}%{concurrency}",
            "--output",
            str(logs / "gpu-%A_%a.out"),
            "--error",
            str(logs / "gpu-%A_%a.err"),
            str(Path(scripts) / "gpu_array.sbatch"),
            str(root),
            str(mapping),
        ]
        job = _submit_checked(command, root, plan["run_id"])
        job_ids.append(job)
        atomic_json(ledger, record)
        atomic_json(root / "submissions" / f"jobs-{job_ids[0]}.json", record)
        print("[SUBMITTED] GPU array", job, "tasks", len(batch), flush=True)
    command = [
        "sbatch",
        "--parsable",
        "--dependency",
        "afterany:" + ":".join(job_ids),
        "--output",
        str(logs / "finalize-%j.out"),
        "--error",
        str(logs / "finalize-%j.err"),
        str(Path(scripts) / "finalize.sbatch"),
        str(root),
    ]
    if account:
        command[2:2] = ["--account", account]
    record["finalize_job_id"] = _submit_checked(command, root, plan["run_id"])
    atomic_json(ledger, record)
    atomic_json(root / "submissions" / f"jobs-{job_ids[0]}.json", record)
    print(
        "[SUBMITTED] CPU finalizer",
        record["finalize_job_id"],
        "afterany; partial results are retained",
        flush=True,
    )
    return record
