"""Adopt a quarantined scheduler receipt without submitting its job again."""

import json
import os
import re
import subprocess
from pathlib import Path

from .manifest import atomic_json, load_plan
from .submission import parse_job_id


def _option(command, name):
    if command.count(name) != 1 or command.index(name) + 1 >= len(command):
        raise ValueError("Missing or ambiguous recorded option: " + name)
    return command[command.index(name) + 1]


def _verify_job(job_id, command, script, root, *, array):
    """Check the real owner, script and run-specific log paths using read-only scontrol."""
    try:
        response = subprocess.run(
            ["scontrol", "show", "job", job_id, "-o"],
            check=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        diagnostic = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(
            "Cannot verify accepted job {} with scontrol; marker retained: {}".format(
                job_id, diagnostic
            )
        ) from exc
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", response.stdout)
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", clean))
    fields = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in fields:
            raise ValueError("Ambiguous scontrol job records; marker retained")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        fields[key] = clean[match.end():end].strip()
    owner = re.fullmatch(r".*\(([0-9]+)\)", fields.get("UserId", ""))
    if owner is None or int(owner.group(1)) != os.getuid():
        raise ValueError("Accepted job owner differs from current user; marker retained")
    identity = fields.get("ArrayJobId") if array else fields.get("JobId")
    if identity != job_id or fields.get("Command") != str(script):
        raise ValueError("Accepted job ID or script differs from receipt; marker retained")
    if fields.get("WorkDir") != str(Path(script).parents[2]):
        raise ValueError("Accepted job working directory differs from project; marker retained")
    for field, flag in (("StdOut", "--output"), ("StdErr", "--error")):
        template = _option(command, flag)
        if Path(template).parent != root / "logs":
            raise ValueError("Receipt logs do not belong to this run")
        pattern = re.escape(template).replace("%A", job_id).replace("%j", job_id)
        pattern = pattern.replace("%a", r"[0-9]+")
        if re.fullmatch(pattern, fields.get(field, "")) is None:
            raise ValueError("Accepted job log paths differ from receipt; marker retained")
    return response.stdout


def recover_submissions(run_dir, *, scripts):
    """Recover initial array/finalizer receipts only when IDs are now unambiguous.

    Does not call sbatch. The normal submit path can subsequently create only
    missing jobs, using the recovered initial_submission.json ledger.
    """
    root = Path(run_dir).expanduser().resolve()
    scripts = Path(scripts).expanduser().resolve()
    plan = load_plan(root)
    markers = sorted((root / "submissions").glob("unparsed-*.json"))
    ledger = root / "submissions" / "initial_submission.json"
    indices = list(range(len(plan["tasks"])))
    record = (
        json.loads(ledger.read_text()) if ledger.is_file()
        else dict(run_id=plan["run_id"], gpu_job_ids=[], finalize_job_id=None, indices=indices)
    )
    if record["run_id"] != plan["run_id"] or record["indices"] != indices:
        raise ValueError("Initial submission ledger differs from frozen run")
    for marker in markers:
        receipt = json.loads(marker.read_text())
        if receipt.get("run_id") != plan["run_id"]:
            raise ValueError("Quarantined receipt belongs to another run")
        job_id = parse_job_id(receipt.get("stdout", ""), receipt.get("stderr", ""))
        command = receipt.get("command", [])
        if not command or not all(isinstance(value, str) for value in command):
            raise ValueError("Invalid quarantined command; marker retained")
        array = "--array" in command
        script = scripts / ("gpu_array.sbatch" if array else "finalize.sbatch")
        if array:
            if len(command) < 3 or command[-3:-1] != [str(script), str(root)]:
                raise ValueError("Recorded array script or run differs; marker retained")
            mapping = Path(command[-1])
            if mapping.resolve().parent != root / "submissions":
                raise ValueError("Recorded array map does not belong to this run")
            batch = json.loads(mapping.read_text())
            batches = [indices[start:start + 500] for start in range(0, len(indices), 500)]
            if batch not in batches:
                raise ValueError("Array map differs from initial submission; marker retained")
            position = batches.index(batch)
            if re.fullmatch(r"0-{}%[1-9][0-9]*".format(len(batch) - 1),
                            _option(command, "--array")) is None:
                raise ValueError("Recorded array range differs from map")
            known = record["gpu_job_ids"]
            if position > len(known) or (position < len(known) and known[position] != job_id):
                raise ValueError("Accepted array conflicts with submission ledger")
            if position == len(known) and (record["finalize_job_id"] or job_id in known):
                raise ValueError("Cannot append array to an already finalized/conflicting ledger")
        else:
            if command[-2:] != [str(script), str(root)] or not record["gpu_job_ids"]:
                raise ValueError("Recorded finalizer differs from initial submission")
            dependency = "afterany:" + ":".join(record["gpu_job_ids"])
            if _option(command, "--dependency") != dependency:
                raise ValueError("Recorded finalizer dependencies differ")
            if record["finalize_job_id"] not in (None, job_id):
                raise ValueError("Accepted finalizer conflicts with submission ledger")
        evidence = _verify_job(job_id, command, script, root, array=array)
        if array and position == len(record["gpu_job_ids"]):
            record["gpu_job_ids"].append(job_id)
        elif not array:
            record["finalize_job_id"] = job_id
        # Persist ledger before archiving; recovery after an interruption is idempotent.
        resolved = root / "submissions" / "resolved" / marker.name
        atomic_json(resolved.with_suffix(".verification.json"), dict(
            run_id=plan["run_id"], job_id=job_id, scontrol_stdout=evidence,
        ))
        atomic_json(ledger, record)
        atomic_json(root / "submissions" / ("jobs-" + record["gpu_job_ids"][0] + ".json"), record)
        os.replace(str(marker), str(resolved))
        print("[RECOVERED]", "GPU array" if array else "CPU finalizer", job_id,
              "verified; original receipt archived at", resolved, flush=True)
    return record
