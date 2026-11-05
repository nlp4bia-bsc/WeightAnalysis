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


def invoke(run, env):
    return subprocess.run(
        ["/bin/bash", str(PROJECT / "scripts/picasso/submit.sh"),
         "--run-dir", str(run), "--concurrency", "16"],
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
