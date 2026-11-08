"""New-run preparation with exact checkpoint identities and explicit exclusions."""

from __future__ import annotations

import fcntl
import json
import math
from collections import defaultdict
from functools import wraps
from itertools import product
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from weight_analysis_scheduler.resources import check_storage

from ...core import WeightReader, config_metadata, selected_keys
from ...core.blocks import parameter_location
from ...core.catalog import stable_id
from ...core.reporting import RunReporter
from ..inventory import discover_inventory, match_scores, verify_signature
from ..plan import atomic_json
from ..source import code_digest, ranking_tables, read_ner_source, runtime_versions
from .evidence import embedding_evidence

VERSION = "multiclinner-packed-1"


def _exclusive_preparation(function):
    @wraps(function)
    def locked(*args, **kwargs):
        root = Path(kwargs["run_dir"]).resolve()
        if (
            (root / "plan.json").exists()
            or (root / "tasks").exists()
            or (root / "ner_source").exists()
        ):
            raise FileExistsError("Legacy run detected; old outputs must remain intact")
        (root / "state").mkdir(parents=True, exist_ok=True)
        with (root / "state/prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return function(*args, **kwargs)

    return locked


def write_table(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = (
        frame if isinstance(frame, pa.Table) else pa.Table.from_pandas(frame, preserve_index=False)
    )
    temporary = path.with_suffix(".partial")
    pq.write_table(table, temporary, compression="zstd", compression_level=6, row_group_size=32768)
    temporary.replace(path)


def load_plan(root):
    path = Path(root) / "manifest/plan.json"
    plan = json.loads(path.read_text())
    identity = {k: v for k, v in plan.items() if k != "run_id"}
    if plan.get("schema_version") != VERSION or stable_id(identity) != plan.get("run_id"):
        raise ValueError("Compact run manifest changed or unsupported")
    if plan["code_digest"] != code_digest():
        raise ValueError("Scientific code differs from frozen compact run")
    return plan


def shared_groups(records, languages, entities):
    groups = []
    candidates = [
        (
            "entity_mean_by_language",
            [r for r in records if r["language"] == lang],
            {(lang, e) for e in entities},
        )
        for lang in languages
    ]
    candidates += [
        (
            "language_mean_by_entity",
            [r for r in records if r["entity"] == ent],
            {(lang, ent) for lang in languages},
        )
        for ent in entities
    ]
    candidates += [("mean_by_backbone", records, set(product(languages, entities)))]
    for family, members, expected in candidates:
        actual = {(r["language"], r["entity"]) for r in members}
        ids = sorted(r["artifact_id"] for r in members)
        groups.append(
            dict(
                family=family,
                members=ids,
                size=len(ids),
                group_id=stable_id([family, ids, sorted(expected)]),
                complete=actual == expected and len(members) == len(expected) and len(ids) > 1,
                missing=sorted(expected - actual),
            )
        )
    return groups


@_exclusive_preparation
def prepare_run(
    *,
    performance,
    trained,
    originals,
    run_dir,
    layers_per_task=4,
    block_elements=131072,
    spectral_mode="exact",
    max_svd_elements=3_000_000,
    topk=32,
    curve_points=64,
    quantile_points=101,
    resident_bytes=4 * 1024**3,
    target_part_bytes=64 * 1024**2,
    checkpoint_fold=1,
    checkpoint_seed=None,
    protocol_id=None,
    evidence=None,
    top_k=3,
    storage_policy=None,
):
    if (
        min(
            layers_per_task,
            block_elements,
            max_svd_elements,
            topk,
            curve_points - 1,
            quantile_points - 1,
            resident_bytes,
            target_part_bytes,
        )
        < 1
    ):
        raise ValueError("Task, block, spectral and storage sizes must be positive")
    if spectral_mode not in ("auto", "exact", "topk", "skip"):
        raise ValueError("Invalid spectral mode")
    root = Path(run_dir).expanduser().resolve()
    if (root / "manifest/plan.json").exists():
        raise FileExistsError("Run already frozen; use resume or choose a NEW run directory")
    if (root / "plan.json").exists() or (root / "tasks").exists() or (root / "ner_source").exists():
        raise FileExistsError("Legacy run detected; old outputs must remain intact")
    policy = dict(
        min_free_bytes=2 * 1024**3,
        min_free_inodes=1024,
        reserve_bytes=8 * 1024**3,
        reserve_inodes=4000,
        max_run_files=8000,
        require_quota=False,
    )
    policy.update(storage_policy or {})
    check_storage(root, policy, launch=True)
    for name in (
        "manifest",
        "logs",
        "state",
        "parquet/atomic",
        "parquet/aggregated",
        "parquet/f1",
        "parquet/intrinsic",
        "figures",
        "reports",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    reporter = RunReporter(log_path=root / "logs/prepare.jsonl")
    try:
        scores, source = read_ner_source(performance, protocol_id=protocol_id)
        source["f1_status"] = "ready"
    except (OSError, ValueError) as exc:
        scores = pd.DataFrame(
            columns=["language", "entity", "backbone", "model_id", "score_id", "fold", "seed"]
        )
        source = dict(
            source_path=str(Path(performance).resolve()),
            source_sha256=None,
            f1_status="blocked",
            error=str(exc),
        )
        reporter.emit("warning", "f1", "F1 blocked; geometry remains independent", exception=exc)
    records, events = discover_inventory(
        trained,
        originals,
        evidence_path=evidence,
        default_fold=checkpoint_fold,
        default_seed=checkpoint_seed,
        reporter=reporter,
    )
    if not records:
        raise ValueError("No valid checkpoints found; inspect logs/prepare.jsonl")
    links = match_scores(scores, records)
    reference_ids = {r["artifact_id"]: r["reference_content_id"] for r in records}
    links["reference_id"] = links.artifact_id.map(reference_ids)
    links["reference_kind"] = "pretraining"
    source.update(
        models_matched=int(links.match_status.eq("exact_layout_aggregate_only").sum()),
        models_unmatched=int(links.match_status.eq("score_without_checkpoint").sum()),
        checkpoints_without_score=int(links.match_status.eq("checkpoint_without_score").sum()),
    )
    write_table(links, root / "manifest/checkpoint_scores.parquet")
    if source["f1_status"] == "ready":
        write_table(scores, root / "parquet/f1/scores.parquet")
        rankings = ranking_tables(scores, k=top_k)
        combined = []
        for name, frame in rankings.items():
            if "pairs" in name:
                continue
            combined.append(frame.assign(ranking_table=name))
        write_table(
            pd.DataFrame([row for frame in combined for row in frame.to_dict("records")]),
            root / "parquet/f1/rankings.parquet",
        )
        write_table(rankings["f1_all_valid_pairs"], root / "parquet/f1/pairs.parquet")
    languages = sorted(set(scores.language) | {r["language"] for r in records})
    entities = sorted(set(scores.entity) | {r["entity"] for r in records})
    domains = defaultdict(list)
    vocab_cache = {}
    for row in records:
        row["embedding_evidence"] = embedding_evidence(
            row["path"], row["reference"]["path"], cache=vocab_cache
        )
        domain = (
            row["backbone"],
            row["initialization_id"],
            row["reference_content_id"],
            row["fold"],
            row["seed"],
        )
        domains[stable_id(domain)].append(row)
    tasks, exclusions = [], []
    spectral_flops, coordinates = 0, 0
    for domain_id, members in sorted(domains.items()):
        members.sort(key=lambda r: r["artifact_id"])
        groups = shared_groups(members, languages, entities)
        for group in groups:
            if not group["complete"]:
                exclusions.append(
                    dict(domain_id=domain_id, **group, reason="incomplete_experimental_grid")
                )
        baseline = members[0]["reference"]["path"]
        base_config = config_metadata(baseline, strict=True)
        configurations = {
            r["artifact_id"]: config_metadata(r["path"], strict=True) for r in members
        }
        member_headers = {}
        for row in members:
            with WeightReader(row["path"]) as reader:
                member_headers[row["artifact_id"]] = {
                    key: reader.shape_dtype(key)[0] for key in reader.keys
                }
        with WeightReader(baseline) as base:
            keys = sorted(selected_keys(base))
            headers = {
                key: (base.shape_dtype(key)[0], parameter_location(base, key)) for key in keys
            }
        tensors = []
        for key, (shape, location) in headers.items():
            eligible, reasons = [], []
            for row in members:
                reason = None
                for field in (
                    "model_type",
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                ):
                    if (
                        base_config.get(field) is not None
                        and configurations[row["artifact_id"]].get(field) is not None
                        and base_config[field] != configurations[row["artifact_id"]][field]
                    ):
                        reason = "reference_architecture_mismatch:" + field
                if member_headers[row["artifact_id"]].get(key) != shape:
                    reason = "missing_or_incompatible_tensor_coordinates"
                if (
                    location["component"] == "embeddings"
                    and not row["embedding_evidence"]["compatible"]
                ):
                    reason = row["embedding_evidence"]["kind"]
                if reason:
                    reasons.append(dict(artifact_id=row["artifact_id"], reason=reason))
                else:
                    eligible.append(row["artifact_id"])
            tensors.append(
                dict(**location, shape=list(shape), members=eligible, exclusions=reasons)
            )
            elements = math.prod(shape)
            coordinates += elements * len(eligible)
            if len(shape) == 2:
                spectral_flops += 4 * elements * min(shape) * (2 * len(eligible) + 1)
        layers = sorted({t["layer"] for t in tensors if t["layer"] is not None})
        buckets = [
            [t for t in tensors if t["layer"] in layers[start : start + layers_per_task]]
            for start in range(0, len(layers), layers_per_task)
        ]
        buckets += [[t for t in tensors if t["layer"] is None]]
        for bucket in buckets:
            if not bucket:
                continue
            task = dict(
                domain_id=domain_id,
                backbone=members[0]["backbone"],
                members=[r["artifact_id"] for r in members],
                groups=groups,
                tensors=bucket,
                total_layers=max(layers) + 1 if layers else 0,
            )
            task.update(task_id=stable_id(task), index=len(tasks))
            tasks.append(task)
    config = dict(
        block_elements=block_elements,
        spectral_mode=spectral_mode,
        max_svd_elements=max_svd_elements,
        topk=topk,
        curve_points=curve_points,
        quantile_points=quantile_points,
        resident_bytes=resident_bytes,
        target_part_bytes=target_part_bytes,
        storage_policy=policy,
        languages=languages,
        entities=entities,
        checkpoint_fold=checkpoint_fold,
        checkpoint_seed=checkpoint_seed,
        protocol_id=protocol_id,
    )
    estimate = dict(
        tasks=len(tasks),
        checkpoints=len(records),
        coordinates=coordinates,
        direct_pairs=sum(len(m) * (len(m) - 1) // 2 for m in domains.values()),
        spectral_exact_flop_upper_bound=spectral_flops,
        planned_file_envelope=64 + len(tasks) * 18,
        max_run_files=policy["max_run_files"],
        time_policy="8h allocation; 7h30 compute budget; resume unfinished units",
        spectral_policy="auto truncation explicitly labelled; exact selectable; no silent fallback",
    )
    if estimate["planned_file_envelope"] > policy["max_run_files"]:
        raise ValueError(
            "Too many task/log files for the configured file budget; increase layers per task"
        )
    plan = dict(
        schema_version=VERSION,
        config=config,
        checkpoints=records,
        tasks=tasks,
        exclusions=exclusions,
        source=source,
        code_digest=code_digest(),
        cost=estimate,
        runtime_versions=runtime_versions(),
        interpretation="descriptive geometry; source fold-aggregate F1 is not individual checkpoint F1",
    )
    plan["run_id"] = stable_id(plan)
    atomic_json(root / "reports/source_audit.json", source)
    atomic_json(root / "reports/prepare_exclusions.json", exclusions)
    atomic_json(root / "reports/cost.json", estimate)
    atomic_json(root / "manifest/plan.json", plan)
    reporter.emit("summary", "prepare", "Compact run frozen", **estimate, run_id=plan["run_id"])
    return plan


def verify_inputs(records):
    expected = {}
    for row in records:
        for signature in (row["input_signature"], row["reference_signature"]):
            for name, value in signature.items():
                if name in expected and expected[name] != value:
                    raise ValueError("Conflicting frozen signatures for " + name)
                expected[name] = value
    verify_signature(expected)
