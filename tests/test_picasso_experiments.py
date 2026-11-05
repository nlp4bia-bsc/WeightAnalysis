"""Real schema contracts and synthetic end-to-end arrays without a scheduler or GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from weight_analysis import CheckpointSpec
from weight_analysis.experiments.exports import finalize_run, read_joined
from weight_analysis.experiments.plan import atomic_json, load_plan, prepare_run
from weight_analysis.experiments.source import ranking_tables, read_ner_source
from weight_analysis.experiments.worker import run_task


@pytest.fixture
def experiment(tmp_path):
    models, trained = tmp_path / "models", tmp_path / "weights"
    scores = []
    tokenizer = {"model": {"vocab": {"a": 0, "b": 1, "c": 2}}, "added_tokens": []}
    for bi, backbone in enumerate(("org--a", "org--b")):
        initial = {
            "bert.encoder.layer.0.attention.self.query.weight": torch.diag(
                torch.tensor([3.0, 2.0, 1.0])
            ),
            "bert.embeddings.word_embeddings.weight": torch.eye(3),
        }
        base = models / backbone
        base.mkdir(parents=True)
        save_file(initial, base / "model.safetensors")
        (base / "config.json").write_text(
            json.dumps({"model_type": "bert", "hidden_size": 3, "num_hidden_layers": 1})
        )
        (base / "tokenizer.json").write_text(json.dumps(tokenizer))
        for li, language in enumerate(("es", "en")):
            for ei, entity in enumerate(("disease", "symptom")):
                path = trained / language / entity / backbone
                path.mkdir(parents=True)
                save_file(
                    {key: value + (bi + li + ei + 1) * 0.1 for key, value in initial.items()},
                    path / "model.safetensors",
                )
                (path / "config.json").write_text((base / "config.json").read_text())
                (path / "tokenizer.json").write_text(json.dumps(tokenizer))
                (path / "encoding.json").write_text(
                    json.dumps({"source": "training", "opaque": "not fold evidence"})
                )
                scores.append(
                    dict(
                        language=language,
                        entity=entity,
                        model=backbone,
                        n_folds=4,
                        strict_f1_mean=0.5 + bi * 0.1 + li * 0.03 + ei * 0.02,
                        strict_f1_std=0.02,
                    )
                )
    source = tmp_path / "results.parquet"
    pd.DataFrame(scores).to_parquet(source, index=False)
    return source, trained, models, tmp_path / "run"


def prepare(experiment):
    source, trained, models, run = experiment
    return prepare_run(
        performance=source,
        trained=trained,
        originals=models,
        run_dir=run,
        layers_per_task=1,
        block_elements=4,
    )


def test_source_does_not_invent_fold_seed_or_aggregate_sd(experiment):
    source, *_ = experiment
    scores, audit = read_ner_source(source)
    assert len(scores) == 8 and scores.fold.isna().all() and scores.seed.isna().all()
    assert scores.n_folds.eq(4).all()
    assert audit["matching_status"] == "pending_weight_inventory"
    assert len(audit["schema_fields"]) == 6
    tables = ranking_tables(scores, k=1)
    assert len(tables["f1_all_valid_pairs"]) == 4
    assert len(tables["f1_best_worst_pairs"]) == 4
    assert tables["f1_ranking_by_language"].strict_f1_std.isna().all()
    assert len(tables["f1_ranking_by_backbone"]) == 2
    reordered, _ = read_ner_source(source)
    first = ranking_tables(scores.sample(frac=1, random_state=9), k=1)
    pd.testing.assert_frame_equal(
        first["f1_model_ranking"].reset_index(drop=True),
        tables["f1_model_ranking"].reset_index(drop=True),
    )
    assert set(reordered.score_id) == set(scores.score_id)


def test_invalid_sources_block_F1(experiment):
    source, *_ = experiment
    original = pd.read_parquet(source)
    for invalid in (
        pd.concat([original, original.iloc[:1]]),
        original.assign(strict_f1_mean=np.inf),
        original.assign(model=" fuzzy "),
    ):
        invalid.to_parquet(source, index=False)
        with pytest.raises(ValueError):
            read_ner_source(source)
    source.unlink()
    with pytest.raises(FileNotFoundError, match="F1 stages blocked"):
        read_ner_source(source)


def test_incomplete_macro_cells_are_not_silently_averaged(experiment):
    source, *_ = experiment
    frame = pd.read_parquet(source)
    frame.iloc[1:].to_parquet(source, index=False)
    scores, audit = read_ner_source(source)
    assert len(audit["missing_cells"]) == 1
    tables = ranking_tables(scores)
    assert len(tables["f1_ranking_by_language"]) == 3
    assert len(tables["f1_ranking_by_backbone"]) == 1


def test_manifest_selection_and_unknown_seed(experiment):
    plan = prepare(experiment)
    assert len(plan["tasks"]) == 4
    assert all(row["fold"] == 1 and row["seed"] is None for row in plan["checkpoints"])
    assert {w["comparison_type"] for t in plan["tasks"] for w in t["work"]} == {
        "checkpoint_vs_pretraining",
        "same_language_cross_entity",
        "same_entity_cross_language",
        "cross_language_cross_entity",
        "entity_mean_by_language",
        "language_mean_by_entity",
        "mean_by_backbone",
    }
    assert all(
        w["axis"] in ("entity", "language", "cell")
        for t in plan["tasks"]
        for w in t["work"]
        if w["kind"] == "shared"
    )
    run = experiment[-1]
    assert load_plan(run)["run_id"] == plan["run_id"]
    changed = dict(plan)
    changed["config"] = {**plan["config"], "block_elements": 8}
    atomic_json(run / "plan.json", changed)
    with pytest.raises(ValueError, match="manifest changed"):
        load_plan(run)
    assert CheckpointSpec("x", "es", "disease", experiment[1], fold=None).fold is None


def test_worker_finalize_and_resume_without_loading_weights_in_finalizer(experiment, monkeypatch):
    plan = prepare(experiment)
    run = experiment[-1]
    for task in plan["tasks"]:
        status = run_task(run, task["index"], device="cpu", budget_seconds=3600)
        assert status["state"] == "complete", status
    # Finalization can only consume the committed Parquet; a weight read is a regression.
    from weight_analysis.core import WeightReader

    def forbidden(*a, **kw):
        raise AssertionError("Postprocessing reloaded a checkpoint")

    monkeypatch.setattr(WeightReader, "get", forbidden)
    final = finalize_run(run)
    assert final["state"] == "complete", final
    joined = read_joined(run)
    assert not joined.empty
    assert joined.fold.dropna().eq(1).all() and joined.seed.isna().all()
    assert joined.n_folds.dropna().eq(4).all()
    pairs = joined[joined.subject_kind == "pair"]
    assert pairs.strict_f1_mean_a.notna().all() and pairs.strict_f1_std_b.notna().all()
    assert {"layer", "component", "encoder", "tensor"} <= set(joined.scope)
    assert "cross_backbone_intrinsic" in set(joined.comparison_type)
    assert (
        read_joined(run, table="weight_geometry_f1_by_layer")
        .scope.isin(["layer", "layer_component", "relative_depth"])
        .all()
    )
    from weight_analysis.experiments.figures import make_figures

    figures, figure_issues = make_figures(run)
    assert figures and not figure_issues, figure_issues
    # Completed work is reused even though tensor access is forbidden.
    resumed = run_task(run, 0, device="cpu")
    assert resumed["state"] == "complete"


def test_signal_budget_partial_finalization_and_retry_indices(experiment):
    plan = prepare(experiment)
    run = experiment[-1]
    status = run_task(run, 0, device="cpu", budget_seconds=0)
    assert status["state"] == "interrupted_resumable"
    final = finalize_run(run, intrinsic=False)
    assert final["state"] == "partial"
    assert final["retry_array_indices"] == list(range(len(plan["tasks"])))


def test_sbatch_resources_and_syntax():
    import subprocess

    root = Path(__file__).resolve().parents[1] / "scripts/picasso"
    for path in list(root.glob("*.sbatch")) + list(root.glob("*.sh")):
        subprocess.run(["bash", "-n", str(path)], check=True)
    gpu = (root / "gpu_array.sbatch").read_text()
    assert "#SBATCH --time=08:00:00" in gpu
    assert "#SBATCH --gres=gpu:1" in gpu
    assert "#SBATCH --constraint=dgx" in gpu
    assert "SLURM_ARRAY_TASK_ID" in gpu and "--signal=B:USR1@300" in gpu
    assert "sbatch" not in gpu
    checks = (root / "tests.sbatch").read_text()
    assert "#SBATCH --constraint=dgx" in checks and "#SBATCH --gres=gpu:1" in checks
    assert "torch.cuda.is_available()" in checks and "python -m pytest" in checks
    for name in ("prepare.sbatch", "finalize.sbatch"):
        cpu = (root / name).read_text()
        assert "#SBATCH --constraint=cpu" in cpu and "#SBATCH --gres" not in cpu


def test_changed_weights_fail_affected_work_but_other_checkpoints_continue(experiment):
    plan = prepare(experiment)
    run = experiment[-1]
    task = plan["tasks"][0]
    broken = next(row for row in plan["checkpoints"] if row["backbone"] == task["backbone"])
    (Path(broken["path"]) / "encoding.json").write_text('{"changed":true}')
    status = run_task(run, 0, device="cpu")
    assert status["state"] == "partial" and status["failed"] > 0 and status["completed"] > 0
    logs = (run / "tasks" / f"00000-{task['backbone']}" / "events.jsonl").read_text()
    assert "Pinned model input changed" in logs


def test_missing_source_only_blocks_F1_preparation(experiment):
    source, trained, models, run = experiment
    source.unlink()
    plan = prepare_run(performance=source, trained=trained, originals=models, run_dir=run)
    assert plan["source"]["f1_status"] == "blocked" and len(plan["tasks"]) > 0
    assert not list((run / "ner_source").glob("f1_*.parquet"))


def test_submit_uses_checked_job_ids_afterany_and_no_real_scheduler(experiment, monkeypatch):
    from types import SimpleNamespace

    from weight_analysis.experiments.__main__ import submit

    prepare(experiment)
    commands = []

    def scheduler(command, **kwargs):
        commands.append(command)
        assert kwargs["check"]
        return SimpleNamespace(stdout=str(100 + len(commands)) + ";picasso\n")

    monkeypatch.setattr("subprocess.run", scheduler)
    result = submit(experiment[-1], scripts="/reviewed/scripts", concurrency=1)
    assert result["gpu_job_ids"] == ["101"] and result["finalize_job_id"] == "102"
    assert "0-3%1" in commands[0]
    assert "afterany:101" in commands[1]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.cuda.is_available() is False")
def test_real_cuda_worker_writes_and_resumes_array_task(experiment):
    plan = prepare(experiment)
    status = run_task(experiment[-1], plan["tasks"][0]["index"], device="gpu", budget_seconds=3600)
    assert status["state"] == "complete"
    assert run_task(experiment[-1], 0, device="gpu")["state"] == "complete"


def test_corrupt_completion_marker_does_not_abort_remaining_finalization(experiment):
    plan = prepare(experiment)
    task = plan["tasks"][0]
    directory = experiment[-1] / "tasks" / f"00000-{task['backbone']}"
    directory.mkdir(parents=True)
    (directory / "status.json").write_text("invalid json")
    work = task["work"][0]
    target = directory / "work" / work["comparison_type"] / work["work_id"]
    target.mkdir(parents=True)
    (target / "completion.json").write_text("broken marker")
    final = finalize_run(experiment[-1], intrinsic=False)
    assert final["state"] == "partial"
    assert final["tasks"][0]["state"] == "unreadable_status"
    assert any(row["reason"].startswith("invalid_marker") for row in final["missing_results"])


def test_scheduler_error_keeps_actual_diagnostic(experiment, monkeypatch):
    import subprocess

    from weight_analysis.experiments.__main__ import submit

    prepare(experiment)

    def reject(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="Invalid account or partition")

    monkeypatch.setattr(subprocess, "run", reject)
    with pytest.raises(RuntimeError, match="Invalid account or partition"):
        submit(experiment[-1], scripts="/reviewed/scripts")
