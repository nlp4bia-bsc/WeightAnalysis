"""Independent numerical oracles and real compact I/O; no real scheduler submissions."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.dataset as ds
import pytest
import torch
from test_picasso_experiments import experiment as experiment

from weight_analysis.experiments.compact.finalize import finalize_run
from weight_analysis.experiments.compact.plan import prepare_run
from weight_analysis.experiments.compact.primitives import TensorStatistics, moments, shared_values
from weight_analysis.experiments.compact.query import joined_geometry, read_geometry
from weight_analysis.experiments.compact.storage import SCHEMA, PackedStore, record
from weight_analysis.experiments.compact.worker import BudgetStop, run_task
from weight_analysis.metrics.spectral import SPECTRAL_METRICS, spectral_metrics
from weight_analysis.statistics.aggregation import VectorMoments
from weight_analysis_scheduler.resources import StoragePressure, check_storage, parse_quota, usage

POLICY = dict(
    require_quota=False,
    reserve_bytes=0,
    reserve_inodes=0,
    min_free_bytes=0,
    min_free_inodes=0,
    max_run_files=8000,
)


def prepare(experiment, **kwargs):
    source, trained, models, run = experiment
    return prepare_run(
        performance=source,
        trained=trained,
        originals=models,
        run_dir=run,
        layers_per_task=1,
        block_elements=4,
        storage_policy=POLICY,
        **kwargs,
    )


def test_block_statistics_match_independent_vectors_and_resume():
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(5, 47, generator=generator, dtype=torch.float64)
    values[1, :5] = 0
    groups = [dict(group_id="group", indices=[1, 2, 3, 4])]
    joint = TensorStatistics(5, groups)
    joint.add(values)
    fragments = TensorStatistics(5, groups)
    fragments.add(values[:, :13])
    fragments = TensorStatistics(5, groups, state=fragments.state())
    for start in range(13, 47, 7):
        fragments.add(values[:, start : start + 7])
    for source in ("final", "delta"):
        tensors = values if source == "final" else values - values[0]
        for i in range(4):
            for j in range(i + 1, 5):
                oracle = VectorMoments.from_tensors(tensors[i], tensors[j])
                left, right = (
                    moments(joint.state(), source, i, j),
                    moments(fragments.state(), source, i, j),
                )
                for field, number in asdict(oracle).items():
                    assert getattr(left, field) == pytest.approx(number, abs=1e-12)
                    assert getattr(right, field) == pytest.approx(number, abs=1e-12)
    delta = (values[1:] - values[0]).numpy()
    centroid = delta.mean(axis=0)
    for i, raw in enumerate(fragments.state()["shared"]["group"]):
        derived = shared_values(raw, 4)
        projection = np.dot(delta[i], centroid) / np.dot(centroid, centroid) * centroid
        assert derived["distance_to_mean"] == pytest.approx(np.linalg.norm(delta[i] - centroid))
        assert derived["specific_residual_energy"] == pytest.approx(
            np.square(delta[i] - projection).sum()
        )
        assert derived["leave_one_out_distance"] == pytest.approx(
            np.linalg.norm(delta[i] - np.delete(delta, i, axis=0).mean(axis=0))
        )


def test_pair_distance_does_not_cancel_under_large_shared_origin():
    values = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 4.0], [1.00001, 2.00002, 4.00003]], dtype=torch.float64
    )
    first, shifted = TensorStatistics(3, []), TensorStatistics(3, [])
    first.add(values)
    shifted.add(values + 10000)
    assert moments(first.state(), "final", 1, 2).qdiff == pytest.approx(
        moments(shifted.state(), "final", 1, 2).qdiff, rel=1e-7
    )
    assert (
        shared_values([1.0, 0.0, 0.0, 1.0], 2)["projected_energy_fraction"]
        != shared_values([1.0, 0.0, 0.0, 1.0], 2)["projected_energy_fraction"]
    )


def test_spectrum_return_is_single_decomposition_and_historical_compatible(monkeypatch):
    matrix = torch.diag(torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64))
    historical = spectral_metrics(matrix, metrics=SPECTRAL_METRICS)
    original, calls = torch.linalg.svdvals, []

    def counted(value):
        calls.append(value.shape)
        return original(value)

    monkeypatch.setattr(torch.linalg, "svdvals", counted)
    stored = spectral_metrics(matrix, metrics=SPECTRAL_METRICS, return_spectrum=True)
    assert len(calls) == 1 and stored.pop("singular_values") == [4.0, 2.0, 1.0]
    assert stored.pop("spectrum_complete") is True
    assert stored == historical


def test_many_units_are_compacted_without_losing_records(tmp_path):
    with PackedStore(tmp_path, "stress", policy=POLICY) as store:
        for i in range(120):
            uid = str(i)
            store.commit(
                uid,
                [
                    record(
                        dict(unit_id=uid, tensor="tensor-" + uid),
                        kind="geometry",
                        metric="energy",
                        value=i,
                    )
                ],
            )
            store.flush()
            if len(store.files()) > 8:
                store.compact()
        before = store.files(verify=True)
        store.compact()
        files = store.files(verify=True)
        table = ds.dataset(
            [str(tmp_path / item["path"]) for item in files], schema=SCHEMA
        ).to_table()
        assert table.num_rows == 120 and len(set(table.column("record_id").to_pylist())) == 120
        assert sorted(table.column("value").to_pylist()) == list(range(120))
        assert len(files) <= 4 and usage(tmp_path)["files"] < 25
        store.compact()
        assert store.files() == files
        assert all(store.done(str(i)) for i in range(120)) and before


def test_interrupted_parquet_publish_replays_from_journal(tmp_path, monkeypatch):
    store = PackedStore(tmp_path, "interrupt", policy=POLICY)
    store.commit("one", [record(dict(unit_id="one"), kind="geometry", metric="x", value=3)])
    original = store._write

    def interrupted(table, identity):
        original(table, identity)
        raise OSError("interrupted after rename")

    monkeypatch.setattr(store, "_write", interrupted)
    with pytest.raises(OSError):
        store.flush()
    assert store.files() == [] and store.done("one")
    store.close()
    with PackedStore(tmp_path, "interrupt", policy=POLICY) as restored:
        restored.flush()
        assert len(restored.files(verify=True)) == 1
        assert len(list((tmp_path / "parquet").rglob("*.parquet"))) == 1
        path = tmp_path / restored.files()[0]["path"]
        path.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="integrity"):
            restored.files(verify=True)


def test_new_generation_keeps_reused_content_addressed_fragments(tmp_path):
    rows = [record(dict(unit_id="one"), kind="geometry", metric="x", value=1)]
    with PackedStore(tmp_path, "versions", area="aggregated", policy=POLICY) as store:
        store.generation("old")
        store.commit("one", rows)
        store.flush()
        original = store.files()
        store.generation("new")
        store.commit("one", rows)
        store.flush()
        assert store.files() == original
        store.reclaim_obsolete()
        assert store.files(verify=True) == original


def test_interrupted_compaction_reclaims_old_owned_parts_on_resume(tmp_path, monkeypatch):
    store = PackedStore(tmp_path, "compaction", policy=POLICY)
    for i in range(6):
        store.commit(str(i), [record(dict(unit_id=str(i)), kind="geometry", metric="x", value=i)])
        store.flush()
    old = {tmp_path / item["path"] for item in store.files()}
    unlink = Path.unlink

    def interrupted(path, *args, **kwargs):
        if path in old:
            raise OSError("interrupted reclaim")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted)
    with pytest.raises(OSError, match="interrupted reclaim"):
        store.compact()
    store.close()
    monkeypatch.setattr(Path, "unlink", unlink)
    with PackedStore(tmp_path, "compaction", policy=POLICY) as restored:
        restored.compact()
        assert len(restored.files(verify=True)) == 1
        assert len(list((tmp_path / "parquet").rglob("*.parquet"))) == 1
        assert sum(item["rows"] for item in restored.files()) == 6


def test_compact_experiment_end_to_end_and_f1_views(experiment, monkeypatch):
    plan = prepare(experiment)
    root = experiment[-1]
    assert len(plan["tasks"]) == 4
    original_source = experiment[0].read_bytes()
    for task in plan["tasks"]:
        status = run_task(root, task["index"], device="cpu")
        assert status["state"] == "complete", status
        assert run_task(root, task["index"], device="cpu")["state"] == "complete"

    def forbidden(*_):
        raise AssertionError("Finalization must not load checkpoint weights")

    monkeypatch.setattr("weight_analysis.core.reading.WeightReader.get", forbidden)
    report = finalize_run(root)
    assert report["state"] == "complete" and report["retry_array_indices"] == []
    assert experiment[0].read_bytes() == original_source
    assert usage(root)["files"] < 100
    joined = joined_geometry(
        root, area="aggregated", filters=dict(scope="encoder", metric="relative_l2")
    )
    assert len(joined) == 8 and joined.fold_a.eq(1).all()
    assert joined.n_folds_a.eq(4).all() and joined.seed_a.isna().all()
    assert "not_fold_F1" in joined.interpretation.iloc[0]
    assert (root / "manifest/views.sql").exists()
    after = finalize_run(root)
    assert after["files"] == report["files"]
    coverage = ds.dataset([str(root / x["path"]) for x in report["files"]], schema=SCHEMA).to_table(
        filter=ds.field("kind") == "coverage"
    )
    assert all(v == 1 for v in coverage.column("value").to_pylist())
    assert any(x["area"] == "intrinsic" for x in report["files"])
    assert report["sealed_atomic"] and all(
        "/task-studies/" in x["path"] for x in report["files"] if x["area"] == "atomic"
    )
    assert run_task(root, 0, device="cpu")["sealed"]
    summary = json.loads((root / "reports/summary.json").read_text())
    assert summary["parquet_files"] == len(report["files"])


def test_duckdb_views_equal_pyarrow_and_no_observations_are_duplicated(experiment):
    pytest.importorskip("duckdb")
    from weight_analysis.experiments.compact.query import duckdb_connection

    plan = prepare(experiment, spectral_mode="skip")
    root = experiment[-1]
    for task in plan["tasks"]:
        assert run_task(root, task["index"], device="cpu")["state"] == "complete"
    finalize_run(root)
    expected = read_geometry(root).num_rows
    with duckdb_connection(root) as database:
        assert database.sql("SELECT count(*) FROM weight_geometry").fetchall()[0][0] == expected
        assert (
            database.sql("SELECT count(DISTINCT record_id) FROM weight_geometry").fetchall()[0][0]
            == expected
        )
        assert (
            database.sql("SELECT count(*) FROM weight_geometry_f1_joined").fetchall()[0][0]
            == expected
        )
        assert database.sql("SELECT count(*) FROM f1_best_worst_pairs").fetchall()[0][0] == 4
        assert database.sql("SELECT count(*) FROM f1_model_ranking").fetchall()[0][0] == 8
        assert database.sql("SELECT min(n_folds) FROM checkpoint_scores").fetchall()[0][0] == 4


def test_changed_f1_only_blocks_f1_of_a_sealed_run(experiment):
    plan = prepare(experiment, spectral_mode="skip")
    root = experiment[-1]
    for task in plan["tasks"]:
        run_task(root, task["index"], device="cpu")
    first = finalize_run(root)
    experiment[0].write_bytes(b"source changed after finalization")
    second = finalize_run(root)
    assert first["files"] == second["files"]
    assert second["f1_status"] == "blocked_source_changed"
    assert read_geometry(root).num_rows > 0
    with pytest.raises(ValueError, match="F1 source blocked"):
        joined_geometry(root)


def test_evidence_uses_real_token_indices_not_empty_tokenizer_files(tmp_path):
    from weight_analysis.experiments.compact.evidence import embedding_evidence

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for path in (a, b):
        (path / "tokenizer.json").write_text("{}")
    assert not embedding_evidence(a, b)["compatible"]

    (a / "tokenizer.json").write_text(
        json.dumps(dict(model=dict(vocab={"one": 0, "two": 1}), normalizer="a"))
    )
    (b / "tokenizer.json").write_text(
        json.dumps(dict(model=dict(vocab={"one": 0, "two": 1}), normalizer="b"))
    )
    assert embedding_evidence(a, b)["compatible"]
    (b / "tokenizer.json").write_text(json.dumps(dict(model=dict(vocab={"one": 1, "two": 0}))))
    assert not embedding_evidence(a, b)["compatible"]


def test_malformed_tokenizer_keeps_encoder_checkpoint_and_excludes_embeddings(experiment):
    _, trained, _, _ = experiment
    tokenizer = next(trained.rglob("tokenizer.json"))
    tokenizer.write_text("{invalid tokenizer json")
    plan = prepare(experiment, spectral_mode="skip")
    assert len(plan["checkpoints"]) == 8
    invalid = [row for row in plan["checkpoints"] if row["path"] == str(tokenizer.parent)]
    assert len(invalid) == 1 and not invalid[0]["embedding_evidence"]["compatible"]
    aid = invalid[0]["artifact_id"]
    assert any(
        aid in tensor["members"]
        for task in plan["tasks"]
        for tensor in task["tensors"]
        if tensor["component"] != "embeddings"
    )
    assert all(
        aid not in tensor["members"]
        for task in plan["tasks"]
        for tensor in task["tensors"]
        if tensor["component"] == "embeddings"
    )


def test_identical_tensors_reuse_one_decomposition_and_alias_spectra(experiment, monkeypatch):
    from safetensors.torch import load_file, save_file

    _, trained, models, root = experiment
    for checkpoint in trained.rglob("model.safetensors"):
        save_file(load_file(models / checkpoint.parent.name / "model.safetensors"), checkpoint)
    plan = prepare(experiment)
    original, calls = torch.linalg.svdvals, []

    def counted(matrix):
        calls.append(tuple(matrix.shape))
        return original(matrix)

    monkeypatch.setattr(torch.linalg, "svdvals", counted)
    for task in plan["tasks"]:
        assert run_task(root, task["index"], device="cpu")["state"] == "complete"
    assert len(calls) == 4  # two base tensors in each of two backbones; no repeated final SVD.
    report = finalize_run(root)
    rows = (
        ds.dataset([str(root / item["path"]) for item in report["files"]], schema=SCHEMA)
        .to_table(
            filter=(ds.field("kind") == "fingerprint")
            & (ds.field("metric") == "spectral_fingerprint")
        )
        .to_pylist()
    )
    assert any("alias_unit_id" in row["payload"] for row in rows)


def test_retry_sends_only_incomplete_tasks_and_is_idempotent(experiment, monkeypatch):
    from weight_analysis_scheduler.compact import submit_compact

    prepare(experiment, spectral_mode="skip")
    root = experiment[-1]
    assert run_task(root, 0, device="cpu")["state"] == "complete"
    pending = finalize_run(root, intrinsic=False)
    commands = []

    def scheduler(command, **_):
        commands.append(command)
        return SimpleNamespace(stdout="{}\n".format(1000 + len(commands)), stderr="")

    monkeypatch.setattr(subprocess, "run", scheduler)
    scripts = Path(__file__).resolve().parents[1] / "scripts/picasso/compact"
    first = submit_compact(root, scripts=scripts, retry=True)
    assert first["indices"] == pending["retry_array_indices"] == [1, 2, 3]
    assert "0-2%1" in commands[0]
    assert submit_compact(root, scripts=scripts, retry=True) == first and len(commands) == 3


def test_journal_payload_corruption_is_never_published(tmp_path):
    with PackedStore(tmp_path, "corrupt", policy=POLICY) as store:
        store.commit("one", [record(dict(unit_id="one"), kind="geometry", metric="x", value=1)])
        with store.db:
            store.db.execute("UPDATE units SET body=? WHERE id='one'", (b"corrupt IPC",))
        with pytest.raises(ValueError, match="checksum"):
            store.flush()
        assert not store.files()


def test_cli_prepare_and_worker_use_documented_arguments(experiment):
    import os
    import sys

    source, trained, models, root = experiment
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-m", "weight_analysis.experiments.compact"]
    prepared = subprocess.run(
        command
        + [
            "prepare",
            "--run-dir",
            str(root),
            "--performance",
            str(source),
            "--trained",
            str(trained),
            "--originals",
            str(models),
            "--allow-unknown-quota",
            "--reserve-gib",
            "0",
            "--reserve-inodes",
            "0",
            "--spectral-mode",
            "skip",
        ],
        env=environment,
        cwd=root.parent,
        capture_output=True,
        text=True,
    )
    assert prepared.returncode == 0, prepared.stderr
    assert "[PREPARED]" in prepared.stdout
    executed = subprocess.run(
        command + ["worker", "--run-dir", str(root), "--index", "0", "--device", "cpu"],
        env=environment,
        cwd=root.parent,
        capture_output=True,
        text=True,
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr


def test_figures_use_persisted_values_and_respect_file_budget(experiment, monkeypatch):
    pytest.importorskip("matplotlib")
    from weight_analysis.experiments.compact.figures import make_figures

    plan = prepare(experiment, spectral_mode="skip")
    root = experiment[-1]
    for task in plan["tasks"]:
        run_task(root, task["index"], device="cpu")
    finalize_run(root)

    def forbidden(*_):
        raise AssertionError("Figures cannot load checkpoint weights")

    monkeypatch.setattr("weight_analysis.core.reading.WeightReader.get", forbidden)
    saved, issues = make_figures(root, metrics=["l2_distance"], max_figures=2)
    assert len(saved) == 2
    assert all((root / item["path"]).is_file() for item in saved)
    assert all("Figure budget reached" in item["reason"] for item in issues)


def test_embeddings_missing_evidence_excludes_only_embeddings(experiment):
    _, trained, models, root = experiment
    for path in [*trained.rglob("tokenizer.json"), *models.rglob("tokenizer.json")]:
        path.unlink()
    plan = prepare(experiment, spectral_mode="skip")
    for task in plan["tasks"]:
        assert run_task(root, task["index"], device="cpu")["state"] == "complete"
    report = finalize_run(root)
    assert report["state"] == "complete"
    frame = read_geometry(root, area="aggregated", filters=dict(metric="delta_energy")).to_pandas()
    assert len(frame) and "embeddings" not in set(frame.component.dropna())
    paths = [str(root / x["path"]) for x in report["files"]]
    exclusions = (
        ds.dataset(paths, schema=SCHEMA)
        .to_table(filter=ds.field("kind") == "exclusion")
        .to_pylist()
    )
    individual = [row for row in exclusions if row["metric"] == "coordinate_compatibility"]
    assert len(individual) == 8 and all("vocabulary" in x["payload"] for x in individual)
    encoder = frame[frame.scope == "encoder"]
    assert encoder.status.eq("partial_coverage").all()


def test_interrupted_worker_resumes_without_repeating_completed_units(experiment, monkeypatch):
    plan = prepare(experiment, spectral_mode="skip")
    root = experiment[-1]
    original = TensorStatistics.add
    calls = []

    def stop_after_block(self, block):
        original(self, block)
        calls.append(block.shape[1])
        raise BudgetStop()

    monkeypatch.setattr(TensorStatistics, "add", stop_after_block)
    assert run_task(root, 0, device="cpu")["state"] == "interrupted_resumable"
    monkeypatch.setattr(TensorStatistics, "add", original)
    assert run_task(root, 0, device="cpu")["state"] == "complete"
    assert calls == [4]
    pending = finalize_run(root, intrinsic=False)
    assert pending["state"] == "partial" and 0 not in pending["retry_array_indices"]
    assert set(pending["retry_array_indices"]) == set(range(1, len(plan["tasks"])))


def test_quota_parsing_and_launch_refusal(monkeypatch, tmp_path):
    text = "Filesystem kbytes quota limit grace files quota limit grace\n/fs 1024* 2048 4096 - 100 200 400 -\n"
    assert parse_quota(text) == dict(remaining_bytes=1024 * 1024, remaining_inodes=100)
    monkeypatch.setattr(
        "weight_analysis_scheduler.resources.quota_snapshot",
        lambda *_: dict(remaining_bytes=1, remaining_inodes=0),
    )
    with pytest.raises(StoragePressure):
        check_storage(tmp_path, POLICY, required_inodes=1, launch=True)
    monkeypatch.setattr("weight_analysis_scheduler.resources.quota_snapshot", lambda *_: None)
    with pytest.raises(StoragePressure, match="quota unavailable"):
        check_storage(tmp_path, dict(POLICY, require_quota=True), launch=True)


def test_compact_scheduler_is_idempotent_uses_ansi_and_three_stages(experiment, monkeypatch):
    from weight_analysis_scheduler.compact import submit_compact

    prepare(experiment, spectral_mode="skip")
    commands = []

    def scheduler(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            stdout="\x1b[31m\x1b[0m{};picasso\n".format(900000 + len(commands)), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", scheduler)
    scripts = Path(__file__).resolve().parents[1] / "scripts/picasso/compact"
    record = submit_compact(experiment[-1], scripts=scripts, concurrency=16)
    assert len(commands) == 3 and record["gpu_job_ids"] == ["900001"]
    assert (
        "0-3%16" in commands[0]
        and "afterany:900001" in commands[1]
        and "afterany:900002" in commands[2]
    )
    assert submit_compact(experiment[-1], scripts=scripts, concurrency=16) == record
    assert len(commands) == 3


def test_compact_submit_wrapper_runs_without_scientific_site_packages(experiment, tmp_path):
    from test_host_submission import host_env

    prepare(experiment, spectral_mode="skip")
    root = experiment[-1]
    env, capture = host_env(tmp_path)
    script = tmp_path / "bin/sbatch"
    script.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$WA_CAPTURE"\n'
        'case "$*" in *--array*) printf "\\033[31m\\033[0m101\\n";; '
        "*afterany:101*) echo 102;; *) echo 103;; esac\n"
    )
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "/bin/bash",
            str(project / "scripts/picasso/compact/submit.sh"),
            "--run-dir",
            str(root),
            "--concurrency",
            "16",
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    calls = capture.read_text().splitlines()
    assert len(calls) == 3 and "afterany:101" in calls[1] and "afterany:102" in calls[2]


def test_concurrent_submit_is_blocked_before_any_scheduler_call(experiment, monkeypatch):
    import fcntl

    from weight_analysis_scheduler.compact import submit_compact

    prepare(experiment, spectral_mode="skip")
    root = experiment[-1]
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with (root / "state/submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            submit_compact(root, scripts="/not-used")
    assert calls == []


def test_new_scripts_are_valid_and_old_runs_cannot_be_reused(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for file in (root / "scripts/picasso/compact").iterdir():
        subprocess.run(["bash", "-n", str(file)], check=True)
    gpu = (root / "scripts/picasso/compact/gpu_array.sbatch").read_text()
    assert "--constraint=dgx" in gpu and "--gres=gpu:1" in gpu and "--time=08:00:00" in gpu
    (tmp_path / "plan.json").write_text("legacy")
    with pytest.raises(FileExistsError, match="Legacy"):
        prepare_run(performance="missing", trained="missing", originals="missing", run_dir=tmp_path)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compact_real_cuda_worker_and_cpu_numerical_equivalence(experiment):
    plan = prepare(experiment)
    root = experiment[-1]
    result = run_task(root, 0, device="gpu")
    assert result["state"] == "complete"
    tensor = torch.randn(4, 71, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    cpu, gpu = TensorStatistics(4, []), TensorStatistics(4, [], device="cuda")
    cpu.add(tensor)
    gpu.add(tensor.cuda())
    assert moments(cpu.state(), "final", 1, 2).value("l2_distance") == pytest.approx(
        moments(gpu.state(), "final", 1, 2).value("l2_distance"), rel=1e-12
    )
    assert plan["config"]["spectral_mode"] == "exact"
