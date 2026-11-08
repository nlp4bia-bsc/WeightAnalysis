"""Exercise the real login shell and host Python against a fake scheduler only."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from weight_analysis.core.catalog import stable_id
from weight_analysis.experiments.plan import atomic_json, load_plan
from weight_analysis.experiments.source import code_digest
from weight_analysis_scheduler.manifest import load_plan as host_load_plan

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen_run(tmp_path):
    root = tmp_path / "run with spaces"
    # A frozen scheduler contract needs no checkpoints to test host submission.
    plan = {
        "schema_version": "picasso-1",
        "code_digest": code_digest(),
        "tasks": [{"index": 0}, {"index": 1}],
        "source": {"label": "Descripción", "mean": 0.75, "seed": None},
    }
    plan["run_id"] = stable_id(plan)
    atomic_json(root / "plan.json", plan)
    return root, plan


def host_env(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    capture = tmp_path / "calls.txt"
    (binaries / "sbatch").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$WA_CAPTURE"\n'
        'case "$*" in *--array*) echo "101;picasso";; *) echo "102";; esac\n'
    )
    # Both would fail if the wrapper reintroduced the compute module/python.
    for name in ("module", "python", "python3"):
        (binaries / name).write_text("#!/bin/sh\nexit 99\n")
    for file in binaries.iterdir():
        file.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(binaries),
        WA_CAPTURE=str(capture),
        WEIGHT_ANALYSIS_PROJECT_DIR=str(PROJECT),
        WEIGHT_ANALYSIS_HOST_PYTHON="/usr/bin/python3",
    )
    return env, capture


def invoke(run, env, *extra):
    return subprocess.run(
        ["/bin/bash", str(PROJECT / "scripts/picasso/submit.sh"),
         "--run-dir", str(run), "--concurrency", "16", *extra],
        cwd=run.parent, env=env, text=True, capture_output=True,
    )


def test_host_uses_stdlib_outside_container_and_does_not_resubmit(frozen_run, tmp_path):
    run, plan = frozen_run
    env, capture = host_env(tmp_path)
    result = invoke(run, env)
    assert result.returncode == 0, result.stderr
    commands = capture.read_text().splitlines()
    assert len(commands) == 2
    assert "0-1%16" in commands[0] and "afterany:101" in commands[1]
    record = json.loads((run / "submissions/initial_submission.json").read_text())
    assert record["run_id"] == plan["run_id"]
    assert record["gpu_job_ids"] == ["101"] and record["finalize_job_id"] == "102"
    assert invoke(run, env).returncode == 0
    assert capture.read_text().splitlines() == commands


def test_host_manifest_validation_matches_scientific_reader(frozen_run):
    run, plan = frozen_run
    assert host_load_plan(run) == load_plan(run) == plan


@pytest.mark.parametrize("mismatch", ["run_id", "code_digest"])
def test_host_rejects_modified_plan_before_scheduler(frozen_run, tmp_path, mismatch):
    run, plan = frozen_run
    if mismatch == "run_id":
        plan["tasks"].append({"index": 2})
    else:
        plan["code_digest"] = "other-science-code"
        plan["run_id"] = stable_id({k: v for k, v in plan.items() if k != "run_id"})
    atomic_json(run / "plan.json", plan)
    env, capture = host_env(tmp_path)
    result = invoke(run, env)
    assert result.returncode != 0
    assert "[SUBMISSION ERROR]" in result.stderr
    assert not capture.exists()


def test_missing_host_scheduler_reports_actionable_error(frozen_run, tmp_path):
    env, capture = host_env(tmp_path)
    (tmp_path / "bin/sbatch").unlink()
    result = invoke(frozen_run[0], env)
    assert result.returncode != 0
    assert "sbatch unavailable" in result.stderr and "login host" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize("stdout,stderr", [
    ("\x1b[31m\x1b[0m3549600\n", ""),  # Exact Picasso response supplied by the user.
    ("", "Submitted batch job 3549600\n"),
    ("3549600\n", "Submitted batch job 3549600\n"),
    ("3549600\n3549600\n", ""),
])
def test_scheduler_colours_and_streams_keep_one_unique_id(stdout, stderr):
    from weight_analysis_scheduler.submission import parse_job_id

    assert parse_job_id(stdout, stderr) == "3549600"


def test_conflicting_scheduler_ids_are_still_rejected():
    from weight_analysis_scheduler.submission import parse_job_id

    with pytest.raises(ValueError, match="no unique job ID"):
        parse_job_id("3549600\n", "Submitted batch job 3549601\n")


def quarantined_array(run, plan):
    scripts = PROJECT / "scripts/picasso"
    mapping = run / "submissions/map-accepted.json"
    atomic_json(mapping, list(range(len(plan["tasks"]))))
    command = [
        "sbatch", "--parsable", "--array", "0-{}%16".format(len(plan["tasks"]) - 1),
        "--output", str(run / "logs/gpu-%A_%a.out"),
        "--error", str(run / "logs/gpu-%A_%a.err"),
        str(scripts / "gpu_array.sbatch"), str(run), str(mapping),
    ]
    marker = run / "submissions/unparsed-accepted.json"
    atomic_json(marker, dict(
        run_id=plan["run_id"], command=command,
        stdout="\x1b[31m\x1b[0m3549600\n", stderr="",
    ))
    # Preserve the relevant shape of the real Picasso scontrol response,
    # including its pending-array task sentinel and unrelated resource fields.
    control = (
        "JobId=3549600 ArrayJobId=3549600 ArrayTaskId=0-96%16 ArrayTaskThrottle=16 "
        "JobName=wa-geometry UserId=fgd({uid}) GroupId=group(3300) JobState=PENDING "
        "Reason=Priority CPUs/Task=8 ReqB:S:C:T=0:0:*:* NodeList= "
        "Command={script} WorkDir={project} "
        "StdErr={run}/logs/gpu-3549600_4294967294.err StdIn=/dev/null "
        "StdOut={run}/logs/gpu-3549600_4294967294.out TresPerNode=gres/gpu:1\n"
    ).format(uid=os.getuid(), script=scripts / "gpu_array.sbatch", project=PROJECT, run=run)
    return marker, control


def install_fake_scontrol(tmp_path, env, text):
    control = tmp_path / "control.txt"
    control.write_text(text)
    env["WA_CONTROL"] = str(control)
    binary = tmp_path / "bin/scontrol"
    binary.write_text('#!/bin/sh\n/bin/cat "$WA_CONTROL"\n')
    binary.chmod(0o755)


def test_recover_accepted_array_sends_only_finalizer_and_keeps_plan(frozen_run, tmp_path):
    run, plan = frozen_run
    original_plan = (run / "plan.json").read_bytes()
    marker, control = quarantined_array(run, plan)
    original_receipt = marker.read_bytes()
    env, capture = host_env(tmp_path)
    install_fake_scontrol(tmp_path, env, control)
    result = invoke(run, env, "--recover")
    assert result.returncode == 0, result.stderr
    assert "[RECOVERED] GPU array 3549600" in result.stdout
    assert (run / "plan.json").read_bytes() == original_plan
    assert not marker.exists()
    assert (run / "submissions/resolved" / marker.name).read_bytes() == original_receipt
    assert list((run / "submissions/resolved").glob("*.verification.json"))
    calls = capture.read_text().splitlines()
    assert len(calls) == 1 and "--array" not in calls[0]
    assert "afterany:3549600" in calls[0]
    assert invoke(run, env, "--recover").returncode == 0
    assert capture.read_text().splitlines() == calls


@pytest.mark.parametrize("mismatch", ["owner", "logs", "script", "run_id", "mapping"])
def test_recovery_rejects_wrong_evidence_without_sending_jobs(frozen_run, tmp_path, mismatch):
    run, plan = frozen_run
    marker, control = quarantined_array(run, plan)
    if mismatch == "owner":
        control = control.replace("fgd({})".format(os.getuid()), "other(999999)")
    elif mismatch == "logs":
        control = control.replace("gpu-3549600_", "gpu-1111111_")
    elif mismatch == "script":
        control = control.replace("gpu_array.sbatch", "another.sbatch")
    elif mismatch == "run_id":
        value = json.loads(marker.read_text())
        value["run_id"] = "another-run"
        atomic_json(marker, value)
    else:
        atomic_json(run / "submissions/map-accepted.json", [0])
    env, capture = host_env(tmp_path)
    install_fake_scontrol(tmp_path, env, control)
    result = invoke(run, env, "--recover")
    assert result.returncode != 0
    assert marker.exists() and not capture.exists()
    assert not (run / "submissions/initial_submission.json").exists()


def test_missing_scontrol_retains_quarantine(frozen_run, tmp_path):
    run, plan = frozen_run
    marker, _ = quarantined_array(run, plan)
    env, capture = host_env(tmp_path)
    result = invoke(run, env, "--recover")
    assert result.returncode != 0
    assert "Cannot verify accepted job 3549600" in result.stderr
    assert marker.exists() and not capture.exists()


def test_recovery_resumes_after_ledger_saved_before_marker_archive(
    frozen_run, monkeypatch
):
    from types import SimpleNamespace

    from weight_analysis_scheduler.recovery import recover_submissions

    run, plan = frozen_run
    marker, control = quarantined_array(run, plan)
    calls = []

    def inspect(command, **kwargs):
        calls.append(command)
        assert command == ["scontrol", "show", "job", "3549600", "-o"]
        return SimpleNamespace(stdout=control, stderr="")

    monkeypatch.setattr(subprocess, "run", inspect)
    replace = os.replace

    def interrupted(source, target):
        if str(source) == str(marker):
            raise OSError("interrupted before archive")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted before archive"):
        recover_submissions(run, scripts=PROJECT / "scripts/picasso")
    assert marker.exists()
    monkeypatch.setattr(os, "replace", replace)
    record = recover_submissions(run, scripts=PROJECT / "scripts/picasso")
    assert record["gpu_job_ids"] == ["3549600"] and len(calls) == 2
    assert not marker.exists()


def test_recover_accepted_finalizer_does_not_submit_it_twice(frozen_run, tmp_path):
    run, plan = frozen_run
    atomic_json(run / "submissions/initial_submission.json", dict(
        run_id=plan["run_id"], gpu_job_ids=["3549600"], finalize_job_id=None,
        indices=[0, 1],
    ))
    script = PROJECT / "scripts/picasso/finalize.sbatch"
    marker = run / "submissions/unparsed-finalizer.json"
    atomic_json(marker, dict(
        run_id=plan["run_id"], stdout="\x1b[31m\x1b[0m3549601\n", stderr="",
        command=["sbatch", "--parsable", "--dependency", "afterany:3549600",
                 "--output", str(run / "logs/finalize-%j.out"),
                 "--error", str(run / "logs/finalize-%j.err"), str(script), str(run)],
    ))
    control = (
        "JobId=3549601 UserId=fgd({}) Command={} WorkDir={} "
        "StdOut={}/logs/finalize-3549601.out StdErr={}/logs/finalize-3549601.err\n"
    ).format(os.getuid(), script, PROJECT, run, run)
    env, capture = host_env(tmp_path)
    install_fake_scontrol(tmp_path, env, control)
    result = invoke(run, env, "--recover")
    assert result.returncode == 0, result.stderr
    assert not capture.exists()
    ledger = json.loads((run / "submissions/initial_submission.json").read_text())
    assert ledger["finalize_job_id"] == "3549601" and not marker.exists()
