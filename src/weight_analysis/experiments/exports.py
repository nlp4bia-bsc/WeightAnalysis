"""Postprocess persisted geometry and exact NER links without reading model weights."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pandas as pd

from ..comparisons.backbones import _profile, compare_backbones
from ..core.catalog import stable_id
from ..core.geometry_schema import observation
from ..metrics import quantile_w1
from ..provenance import file_sha256
from ..results import AnalysisResult, read_result, write_result
from ..results.comparative import ComparativeResult, merge_comparative, read_dataset, write_dataset
from ..results.models import data_digest
from ..statistics.aggregation import aggregate_geometry
from .inventory import match_scores
from .plan import atomic_json, load_plan
from .source import ranking_tables, read_ner_source, runtime_versions

BASE_FIELDS = [
    "observation_id",
    "subject_id",
    "subject_kind",
    "metric",
    "value",
    "status",
    "scope",
    "source",
    "reference_id",
    "reference_type",
    "layer",
    "relative_depth",
    "component",
    "tensor",
    "embedding_type",
    "n_parameters",
    "n_members",
    "group_id",
    "member_id",
    "axis",
    "execution_id",
    "definition_id",
    "comparison_type",
    "coverage_status",
    "ancestry_status",
    "interpretation",
    "abs_f1_gap",
    "performance_pair_id",
]
ENDPOINT_FIELDS = [
    "model_id",
    "artifact_id",
    "score_id",
    "backbone",
    "language",
    "entity",
    "fold",
    "seed",
    "strict_f1_mean",
    "strict_f1_std",
    "n_folds",
    "score_scope",
]
JOIN_FIELDS = (
    BASE_FIELDS
    + ENDPOINT_FIELDS
    + [name + "_" + side for side in ("a", "b") for name in ENDPOINT_FIELDS]
)


def joined_geometry(result, work, links, *, coverage="complete"):
    lookup = links[links.artifact_id.notna()].set_index("artifact_id").to_dict("index")
    geometry = pd.concat(
        [
            result.tables.get(name, pd.DataFrame())
            for name in ("geometry_detailed", "geometry_aggregated")
        ],
        ignore_index=True,
    )
    rows = []
    for row in geometry.to_dict("records"):
        row.update(
            comparison_type=work["comparison_type"],
            coverage_status=coverage,
            ancestry_status="user_declared_reference_ancestry_not_independently_verified",
            reference_type="pretraining"
            if pd.notna(row.get("reference_id"))
            else "origin_invariant",
            interpretation="descriptive_geometry_vs_aggregate_not_checkpoint_F1",
        )
        kind = row["subject_kind"]
        if kind == "pair":
            for side, aid in zip(("a", "b"), sorted(work["members"])):
                endpoint = {**lookup.get(aid, {}), "artifact_id": aid}
                row.update({key + "_" + side: endpoint.get(key) for key in ENDPOINT_FIELDS})
            if pd.notna(row.get("strict_f1_mean_a")) and pd.notna(row.get("strict_f1_mean_b")):
                row["abs_f1_gap"] = abs(row["strict_f1_mean_a"] - row["strict_f1_mean_b"])
                row["performance_pair_id"] = stable_id(
                    sorted([row["score_id_a"], row["score_id_b"]])
                )
            for key in ("backbone", "language", "entity"):
                row[key] = (
                    row[key + "_a"]
                    if row[key + "_a"] == row[key + "_b"]
                    else "__cross_" + key + "__"
                )
            row["interpretation"] += (
                ";cross_task_descriptive_only"
                if row["language_a"] != row["language_b"] or row["entity_a"] != row["entity_b"]
                else ""
            )
        elif kind in ("drift", "checkpoint", "member_to_group"):
            aid = row.get("member_id") if kind == "member_to_group" else row["subject_id"]
            endpoint = {**lookup.get(aid, {}), "artifact_id": aid}
            for key in ENDPOINT_FIELDS:
                row[key] = endpoint.get(key)
            if kind == "drift":
                row["reference_type"] = "pretraining"
        rows.append(row)
    frame = pd.DataFrame(rows).reindex(columns=JOIN_FIELDS)
    for column in frame:
        if column in ("value", "abs_f1_gap", "relative_depth") or column.startswith("strict_f1_"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Float64")
        elif column in (
            "layer",
            "n_parameters",
            "n_members",
            "fold",
            "seed",
            "n_folds",
        ) or column.startswith(("fold_", "seed_", "n_folds_")):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")
        else:
            frame[column] = frame[column].astype("string")
    return frame


def _write_joined(frame, root, part, provenance):
    written = []
    for name, selected in (
        ("weight_geometry_f1_joined", frame),
        (
            "weight_geometry_f1_by_layer",
            frame[frame.scope.isin(["layer", "layer_component", "relative_depth"])],
        ),
        ("weight_geometry_f1_by_component", frame[frame.scope.eq("component")]),
        ("weight_geometry_f1_by_language", frame),
        ("weight_geometry_f1_by_entity", frame),
        ("weight_geometry_f1_by_backbone", frame),
    ):
        # Dimension views retain observations verbatim; they are not averaged correlations/norms.
        directory = root / name
        dimension = {
            "weight_geometry_f1_by_language": "language",
            "weight_geometry_f1_by_entity": "entity",
            "weight_geometry_f1_by_backbone": "backbone",
        }.get(name)
        groups = (
            selected.groupby(dimension, dropna=False, sort=True)
            if dimension and not selected.empty
            else [(None, selected)]
        )
        for label, group in groups:
            destination = (
                directory / (dimension + "-" + quote(str(label), safe=""))
                if dimension
                else directory
            )
            written.append(
                write_result(
                    AnalysisResult(name, group.reset_index(drop=True), provenance),
                    destination / f"part-{part}.parquet",
                )
            )
    return written


def cached_intrinsic(a, b, spectra, drift, *, minimum_components=3):
    """Reuse historical intrinsic definitions, retaining component and relative-depth views."""
    descriptors = [
        SimpleNamespace(
            uid=e["artifact_id"],
            path=e["path"],
            backbone=e["backbone"],
            language=e["language"],
            entity=e["entity"],
        )
        for e in (a, b)
    ]
    pair_id = stable_id({"checkpoints": sorted([a["artifact_id"], b["artifact_id"]])})
    rows = []
    components = sorted(set(spectra.component) - {"embeddings", "norm", "other"})
    execution = stable_id({"kind": "persisted_intrinsic", "definition": "fine-grained-1"})
    for component in [None] + components:
        data = spectra if component is None else spectra[spectra.component == component]
        delta = drift if component is None else drift[drift.component == component]
        values = compare_backbones(
            *descriptors, data, delta, minimum_components if component is None else 1
        )
        base = dict(
            subject_id=pair_id,
            subject_kind="pair",
            component=component,
            layer=None,
            tensor=None,
            embedding_type=None,
            execution_id=execution,
        )
        for source in ("final", "delta"):
            for name in ("trace_distance_" + source, "curve_l1_" + source, "quantile_w1_" + source):
                meta = {
                    **base,
                    "source": source,
                    "reference_id": stable_id(
                        [a["reference_content_id"], b["reference_content_id"]]
                    )
                    if source == "delta"
                    else None,
                }
                rows.append(
                    observation(
                        "cached",
                        meta,
                        name,
                        values[name],
                        scope="encoder" if component is None else "component",
                    )
                )
            if component is not None:
                for column, metric in (
                    ("spectral_energy_curve", "curve_l1_" + source),
                    ("normalized_weight_quantiles", "quantile_w1_" + source),
                ):
                    x, y = [
                        _profile(
                            spectra[spectra.checkpoint == e["artifact_id"]],
                            source,
                            component,
                            column,
                        )
                        for e in (a, b)
                    ]
                    if x is None or y is None:
                        continue
                    for position, (left, right) in enumerate(zip(x, y, strict=True)):
                        value = (
                            float(abs(left - right).mean())
                            if column == "spectral_energy_curve"
                            else quantile_w1(left, right)
                        )
                        rows.append(
                            observation(
                                "cached",
                                {
                                    **base,
                                    "source": source,
                                    "reference_id": None
                                    if source == "final"
                                    else stable_id(
                                        [a["reference_content_id"], b["reference_content_id"]]
                                    ),
                                },
                                metric,
                                value,
                                scope="relative_depth",
                                relative_depth=position / 15,
                            )
                        )
        rows.append(
            observation(
                "cached",
                {
                    **base,
                    "source": "delta",
                    "reference_id": stable_id(
                        [a["reference_content_id"], b["reference_content_id"]]
                    ),
                },
                "adaptation_profile_l1",
                values["adaptation_profile_l1"],
                scope="encoder" if component is None else "component",
            )
        )
    return pd.DataFrame(rows)


def finalize_run(run_dir, *, intrinsic=True):
    root = Path(run_dir).expanduser().resolve()
    plan = load_plan(root)
    by_work = defaultdict(list)
    states, missing = [], []
    for task in plan["tasks"]:
        directory = root / "tasks" / f"{task['index']:05d}-{task['backbone']}"
        state_path = directory / "status.json"
        try:
            state = (
                json.loads(state_path.read_text())
                if state_path.is_file()
                else {"index": task["index"], "state": "not_started"}
            )
            if state["index"] != task["index"]:
                raise ValueError("Wrong array status identity")
        except Exception as exc:
            state = {"index": task["index"], "state": "unreadable_status", "reason": str(exc)}
        states.append(state)
        for work in task["work"]:
            result_path = directory / "work" / work["comparison_type"] / work["work_id"]
            marker_path = result_path / "completion.json"
            if not marker_path.is_file():
                missing.append(
                    {
                        "task_index": task["index"],
                        "work_id": work["work_id"],
                        "reason": "no_committed_result",
                    }
                )
                continue
            try:
                marker = json.loads(marker_path.read_text())
                if marker["run_id"] != plan["run_id"] or marker["task_id"] != task["task_id"]:
                    raise ValueError("Result marker belongs to another run/task")
            except Exception as exc:
                missing.append(
                    {
                        "task_index": task["index"],
                        "work_id": work["work_id"],
                        "reason": "invalid_marker: " + str(exc),
                    }
                )
                continue
            by_work[work["work_id"]].append((work, result_path, marker))
    derived = root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    f1_status, links = "ready", None
    try:
        scores, source = read_ner_source(
            plan["source"]["source_path"], protocol_id=plan["config"]["protocol_id"]
        )
        if source["source_sha256"] != plan["source"]["source_sha256"]:
            raise ValueError("Original NER source changed after freezing the run")
        links = match_scores(scores, plan["checkpoints"])
        for name, frame in ranking_tables(scores, k=plan["config"]["top_k"]).items():
            write_result(AnalysisResult(name, frame, plan["source"]), derived / f"{name}.parquet")
    except (OSError, ValueError, KeyError) as exc:
        f1_status = "blocked: " + str(exc)
    fingerprints, drift_inputs, issues = [], [], []
    if not plan["tasks"]:
        issues.append({"reason": "no_geometry_tasks; inspect inventory incidents"})
    records = {row["artifact_id"]: row for row in plan["checkpoints"]}
    work_lookup = {w["work_id"]: w for t in plan["tasks"] for w in t["work"]}
    expected_parts = defaultdict(int)
    for task in plan["tasks"]:
        for work in task["work"]:
            expected_parts[work["work_id"]] += 1
    exported = []
    for wid, parts in sorted(by_work.items()):
        try:
            results = []
            for _, path, marker in parts:
                if file_sha256(path / "dataset" / "manifest.json") != marker["manifest_sha256"]:
                    raise ValueError("Result manifest changed after completion")
                results.append(read_dataset(path / "dataset"))
            result = aggregate_geometry(merge_comparative(results))
            complete = len(parts) == expected_parts[wid] and all(
                marker["state"] == "complete" for _, _, marker in parts
            )
            observations = result.tables.get("geometry_detailed", pd.DataFrame())
            if not observations.empty and observations.execution_id.nunique() > 1:
                complete = False
                issues.append(
                    {
                        "work_id": wid,
                        "reason": "mixed_execution_devices; encoder aggregates remain separated by execution_id",
                    }
                )
            coverage = "complete" if complete else "partial_selected_coordinates_not_full_encoder"
            work = work_lookup[wid]
            destination = derived / "geometry" / work["comparison_type"] / wid
            write_dataset(result, destination)
            if links is not None:
                frame = joined_geometry(result, work, links, coverage=coverage)
                _write_joined(
                    frame,
                    derived,
                    wid,
                    {"run_id": plan["run_id"], "source": plan["source"], "work": work},
                )
                exported.append(wid)
            if work["kind"] == "reference" and complete:
                wide = result.tables.get("intrinsic_inputs", pd.DataFrame())
                if not wide.empty:
                    fingerprints.append(wide)
                frame = result.tables.get("geometry_detailed", pd.DataFrame())
                if not frame.empty:
                    data = frame[
                        (frame.subject_kind == "drift") & (frame.metric == "relative_l2")
                    ].copy()
                    data["checkpoint"] = data.subject_id
                    data["level"] = "tensor"
                    data["relative_l2"] = data.value
                    drift_inputs.append(data)
            if not complete:
                issues.append({"work_id": wid, "reason": coverage})
        except Exception as exc:
            issues.append({"work_id": wid, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[FINALIZE ERROR] {wid}: {exc}", flush=True)
    intrinsic_written, intrinsic_skipped, intrinsic_active = 0, [], []
    if fingerprints:
        spectra = pd.concat(fingerprints, ignore_index=True)
        if spectra.duplicated(["checkpoint", "tensor", "source"]).any():
            raise ValueError("Overlapping intrinsic coordinates")
        write_result(
            AnalysisResult("intrinsic_inputs", spectra, {"run_id": plan["run_id"]}),
            derived / "intrinsic_inputs.parquet",
        )
        drift = (
            pd.concat(drift_inputs, ignore_index=True)
            if drift_inputs
            else pd.DataFrame(columns=["checkpoint", "level", "component", "layer", "relative_l2"])
        )
        if intrinsic and links is not None:
            fingerprint_hashes = {
                cp: data_digest(group) for cp, group in spectra.groupby("checkpoint", sort=True)
            }
            drift_hashes = {
                cp: data_digest(group) for cp, group in drift.groupby("checkpoint", sort=True)
            }
            pairs = read_result(derived / "f1_all_valid_pairs.parquet").data
            lookup = (
                links[links.artifact_id.notna() & links.score_id.notna()]
                .set_index("model_id")
                .artifact_id.to_dict()
            )
            for pair in pairs.to_dict("records"):
                try:
                    ids = [lookup.get(pair["model_id_" + s]) for s in ("a", "b")]
                    if any(aid not in records for aid in ids):
                        intrinsic_skipped.append(
                            {
                                "performance_pair_id": pair["performance_pair_id"],
                                "reason": "unmatched_physical_checkpoint",
                            }
                        )
                        continue
                    a, b = [records[aid] for aid in ids]
                    selected = spectra[spectra.checkpoint.isin(ids)]
                    if set(selected.checkpoint) != set(ids):
                        intrinsic_skipped.append(
                            {
                                "performance_pair_id": pair["performance_pair_id"],
                                "reason": "incomplete_or_failed_fingerprint_checkpoint",
                            }
                        )
                        continue
                    dependency = stable_id(
                        [
                            [(fingerprint_hashes[aid], drift_hashes.get(aid)) for aid in ids],
                            plan["code_digest"],
                            plan["source"]["source_sha256"],
                        ]
                    )
                    marker_path = (
                        derived / "intrinsic_completion" / (pair["performance_pair_id"] + ".json")
                    )
                    if marker_path.is_file():
                        cached = json.loads(marker_path.read_text())
                        if cached["dependency"] == dependency and all(
                            file_sha256(root / name) == digest
                            for name, digest in cached["files"].items()
                        ):
                            intrinsic_written += 1
                            intrinsic_active.append(pair["performance_pair_id"])
                            continue
                    geometry = cached_intrinsic(a, b, selected, drift[drift.checkpoint.isin(ids)])
                    work = {
                        "kind": "pair",
                        "comparison_type": "cross_backbone_intrinsic",
                        "members": ids,
                    }
                    result = ComparativeResult(
                        {"geometry_detailed": geometry}, {"run_id": plan["run_id"]}
                    )
                    frame = joined_geometry(
                        result, work, links, coverage="available_fingerprint_components"
                    )
                    files = _write_joined(
                        frame,
                        derived,
                        pair["performance_pair_id"],
                        {"run_id": plan["run_id"], "source": plan["source"], "work": work},
                    )
                    atomic_json(
                        marker_path,
                        {
                            "dependency": dependency,
                            "files": {
                                str(path.relative_to(root)): file_sha256(path) for path in files
                            },
                        },
                    )
                    intrinsic_written += 1
                    intrinsic_active.append(pair["performance_pair_id"])
                except Exception as exc:
                    issues.append(
                        {"performance_pair_id": pair["performance_pair_id"], "reason": str(exc)}
                    )
    retry = sorted(
        {r["task_index"] for r in missing}
        | {s["index"] for s in states if s["state"] != "complete"}
    )
    summary = dict(
        run_id=plan["run_id"],
        f1_status=f1_status,
        tasks=states,
        missing_results=missing,
        issues=issues,
        blocked_inventory_comparisons=plan["exclusions"],
        source_matching=plan["source"].get("matching_status"),
        intrinsic_pairs_written=intrinsic_written,
        intrinsic_pairs_skipped=intrinsic_skipped,
        intrinsic_requested=intrinsic,
        runtime_versions=runtime_versions(),
        retry_array_indices=retry,
        exported_work=len(exported),
        active_export_parts=sorted(exported + intrinsic_active),
        state="complete"
        if not missing
        and not issues
        and not retry
        and not intrinsic_skipped
        and not plan["exclusions"]
        and f1_status == "ready"
        else "partial",
    )
    atomic_json(derived / "finalization.json", summary)
    print(f"[FINALIZE] {summary['state']}; F1={f1_status}; retry={retry}", flush=True)
    return summary


def read_joined(run_dir, *, table="weight_geometry_f1_joined", filters=None):
    """Read explicit derived parts and apply row filters; never load checkpoints."""
    if table not in {
        "weight_geometry_f1_joined",
        "weight_geometry_f1_by_layer",
        "weight_geometry_f1_by_component",
        "weight_geometry_f1_by_language",
        "weight_geometry_f1_by_entity",
        "weight_geometry_f1_by_backbone",
    }:
        raise ValueError("Unknown derived table")
    report_path = Path(run_dir) / "derived" / "finalization.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    if report is not None and report["f1_status"] != "ready":
        raise ValueError(
            "F1 exports are blocked; inspect finalization.json. Geometry datasets remain available."
        )
    paths = sorted((Path(run_dir) / "derived" / table).rglob("part-*.parquet"))
    if report is not None:
        active = set(report["active_export_parts"])
        paths = [path for path in paths if path.stem.removeprefix("part-") in active]
    frame = (
        pd.concat([read_result(path).data for path in paths], ignore_index=True)
        if paths
        else pd.DataFrame(columns=JOIN_FIELDS)
    )
    for key, value in (filters or {}).items():
        frame = frame[frame[key].eq(value)]
    return frame
