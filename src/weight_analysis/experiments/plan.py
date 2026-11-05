"""Immutable, deterministic scientific tasks; no scheduler calls or implicit F1 selection."""

from __future__ import annotations

import fcntl
import json
import os
from functools import wraps
from itertools import combinations, product
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ..configuration import MetricOptions
from ..core.catalog import stable_id
from ..core.reporting import RunReporter
from ..results import AnalysisResult, write_result
from .inventory import discover_inventory, match_scores
from .source import code_digest, write_source_audit


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_preparation(function):
    @wraps(function)
    def locked(*args, **kwargs):
        root = Path(kwargs["run_dir"]).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        with (root / "prepare.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return function(*args, **kwargs)

    return locked


@_exclusive_preparation
def prepare_run(
    *,
    performance,
    trained,
    originals,
    run_dir,
    layers_per_task=4,
    evidence_path=None,
    allow_unverified_origin=False,
    fold=1,
    seed=None,
    protocol_id=None,
    top_k=3,
    spectral_mode="auto",
    max_svd_elements=3_000_000,
    block_elements=1_000_000,
):
    if layers_per_task < 1 or block_elements < 1:
        raise ValueError("Positive task/block sizes required")
    root = Path(run_dir).expanduser().resolve()
    if (root / "plan.json").exists():
        raise FileExistsError(
            "Run already frozen; use worker/submit --retry or a new run directory"
        )
    reporter = RunReporter(log_path=root / "logs" / "prepare.jsonl")
    try:
        scores, source = write_source_audit(
            performance, root / "ner_source", k=top_k, protocol_id=protocol_id
        )
        source["f1_status"] = "ready"
    except (OSError, ValueError) as exc:
        reporter.emit(
            "failed", "ner_source", "F1 stages blocked; preparing geometry only", exception=exc
        )
        scores = pd.DataFrame(
            columns=["language", "entity", "backbone", "model_id", "score_id", "fold", "seed"]
        )
        source = dict(
            source_path=str(Path(performance).resolve()),
            source_sha256=None,
            f1_status="blocked",
            error=str(exc),
        )
    records, events = discover_inventory(
        trained,
        originals,
        evidence_path=evidence_path,
        allow_unverified_origin=allow_unverified_origin,
        default_fold=fold,
        default_seed=seed,
        reporter=reporter,
    )
    matches = match_scores(scores, records)
    source.update(
        models_matched=int(matches.match_status.eq("exact_layout_aggregate_only").sum()),
        models_unmatched=int(matches.match_status.eq("score_without_checkpoint").sum()),
        checkpoints_without_score=int(matches.match_status.eq("checkpoint_without_score").sum()),
        matching_status="on_site_inventory_complete",
    )
    write_result(
        AnalysisResult("score_checkpoint_links", matches, source),
        root / "ner_source" / "score_checkpoint_links.parquet",
    )
    atomic_json(root / "ner_source" / "source_audit.json", source)
    options = dict(
        device="gpu",
        spectral_mode=spectral_mode,
        max_svd_elements=max_svd_elements,
        topk=32,
        subspace_rank=8,
        min_fingerprint_components=3,
    )
    MetricOptions(**options)
    config = dict(
        options=options,
        top_k=top_k,
        block_elements=block_elements,
        layers_per_task=layers_per_task,
        allow_unverified_origin=allow_unverified_origin,
        protocol_id=protocol_id,
        expected_languages=sorted(set(scores.language) | {row["language"] for row in records}),
        expected_entities=sorted(set(scores.entity) | {row["entity"] for row in records}),
        default_fold=fold,
        default_seed=seed,
        reference_kind="pretraining",
        pair_metrics=[
            "l2_distance",
            "rmse",
            "cosine",
            "normalized_l2",
            "pearson",
            "sign_agreement",
        ],
        drift_metrics=[
            "relative_l2",
            "delta_energy",
            "delta_rms",
            "cosine_final_base",
            "base_l2",
            "final_l2",
        ],
        spectral_metrics=[
            "matrix_fro",
            "stable_rank",
            "effective_rank",
            "spectral_energy_curve",
            "normalized_weight_quantiles",
        ],
        sources=["final", "delta"],
        complete_group_policy="equal weight all declared cells; missing cells block group, never shrink",
    )
    tasks, exclusions = [], []
    by_backbone = {}
    for row in records:
        by_backbone.setdefault(row["backbone"], []).append(row)
    for backbone, members in sorted(by_backbone.items()):
        members.sort(key=lambda r: r["artifact_id"])
        layers = sorted({layer for row in members for layer in row["layers"]})
        selectors = [
            dict(
                layers=layers[i : i + layers_per_task], components=None, embeddings=[], tensors=None
            )
            for i in range(0, len(layers), layers_per_task)
        ]
        if any(row["embedding_tensors"] for row in members):
            selectors.append(dict(layers=[], components=None, embeddings=["*"], tensors=None))
        auxiliary = sorted({key for row in members for key in row["nonlayer_tensors"]})
        if auxiliary:
            selectors.append(dict(layers=[], components=None, embeddings=[], tensors=auxiliary))
        work = [
            dict(
                kind="reference",
                members=[r["artifact_id"]],
                comparison_type="checkpoint_vs_pretraining",
            )
            for r in members
        ]
        for a, b in combinations(members, 2):
            if (a["parameter_space_id"], a["initialization_id"], a["fold"], a["seed"]) != (
                b["parameter_space_id"],
                b["initialization_id"],
                b["fold"],
                b["seed"],
            ):
                exclusions.append(
                    dict(
                        kind="pair",
                        members=[a["artifact_id"], b["artifact_id"]],
                        reason="incompatible_parameter_space_origin_or_replica",
                    )
                )
                continue
            kind = (
                "same_language_cross_entity"
                if a["language"] == b["language"]
                else (
                    "same_entity_cross_language"
                    if a["entity"] == b["entity"]
                    else "cross_language_cross_entity"
                )
            )
            work.append(
                dict(
                    kind="pair", members=[a["artifact_id"], b["artifact_id"]], comparison_type=kind
                )
            )
        groups = []
        for language in config["expected_languages"]:
            groups.append(
                (
                    "entity",
                    [r for r in members if r["language"] == language],
                    {(language, e) for e in config["expected_entities"]},
                )
            )
        for entity in config["expected_entities"]:
            groups.append(
                (
                    "language",
                    [r for r in members if r["entity"] == entity],
                    {(l, entity) for l in config["expected_languages"]},
                )
            )
        groups.append(
            (
                "cell",
                members,
                set(product(config["expected_languages"], config["expected_entities"])),
            )
        )
        for axis, group, expected in groups:
            actual = {(r["language"], r["entity"]) for r in group}
            if (
                actual != expected
                or len(group) < 2
                or len(
                    {
                        (r["parameter_space_id"], r["initialization_id"], r["fold"], r["seed"])
                        for r in group
                    }
                )
                != 1
            ):
                exclusions.append(
                    dict(
                        kind="shared_" + axis,
                        backbone=backbone,
                        reason="incomplete_or_incompatible_frozen_group",
                        missing=sorted(expected - actual),
                    )
                )
                continue
            work.append(
                dict(
                    kind="shared",
                    axis=axis,
                    members=sorted(r["artifact_id"] for r in group),
                    comparison_type={
                        "entity": "entity_mean_by_language",
                        "language": "language_mean_by_entity",
                        "cell": "mean_by_backbone",
                    }[axis],
                )
            )
        for item in work:
            item["work_id"] = stable_id(item)
        for selector in selectors:
            task = dict(backbone=backbone, selector=selector, work=work)
            task["task_id"] = stable_id(task)
            task["index"] = len(tasks)
            tasks.append(task)
    plan = dict(
        schema_version="picasso-1",
        source=source,
        config=config,
        checkpoints=records,
        tasks=tasks,
        exclusions=exclusions,
        inventory_events=events,
        code_digest=code_digest(),
    )
    plan["run_id"] = stable_id(plan)
    atomic_json(root / "plan.json", plan)
    reporter.emit(
        "summary",
        "prepare",
        "Frozen run ready",
        tasks=len(tasks),
        checkpoints=len(records),
        exclusions=len(exclusions),
        run_id=plan["run_id"],
    )
    return plan


def load_plan(run_dir):
    path = Path(run_dir) / "plan.json"
    plan = json.loads(path.read_text())
    identity = {k: v for k, v in plan.items() if k != "run_id"}
    if plan.get("schema_version") != "picasso-1" or stable_id(identity) != plan["run_id"]:
        raise ValueError("Run manifest changed or unsupported version")
    if plan["code_digest"] != code_digest():
        raise ValueError(
            "Scientific code differs from frozen plan; restore code or prepare a new run"
        )
    return plan
