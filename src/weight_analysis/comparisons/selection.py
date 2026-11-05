"""Deterministic performance-ranked pair selection without reading model weights."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping, Sequence

import pandas as pd

from ..core.catalog import CheckpointCatalog, stable_id
from ..statistics.scores import bind_scores, performance_id, validate_performance

STRATA = ("language", "entity", "fold", "seed", "protocol_id", "evaluation_id")
STRATEGIES = ("best_worst", "best_all", "worst_all", "top_bottom", "all_pairs")


@dataclass(frozen=True)
class SelectionResult:
    catalog: CheckpointCatalog
    performance: pd.DataFrame
    rankings: pd.DataFrame
    pairs: pd.DataFrame
    exclusions: pd.DataFrame
    provenance: dict


def select_performance_pairs(
    *,
    checkpoints: CheckpointCatalog,
    performance: pd.DataFrame,
    strategy: str = "best_worst",
    k: int = 3,
    strata: Sequence[str] = STRATA,
    tie_policy: str = "stable_id",
    score_alignment: Mapping[str, str] | None = None,
    allow_mixed_strata: bool = False,
    experimental_policy: str | None = None,
    scale: str = "unit",
) -> SelectionResult:
    """Freeze the full ranking before fragmenting any scientific computations."""
    if strategy not in STRATEGIES or tie_policy not in ("stable_id", "include_all"):
        raise ValueError("Unknown selection strategy or tie policy")
    if not strata or len(set(strata)) != len(strata):
        raise ValueError("strata must contain distinct explicit dimensions")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not allow_mixed_strata and not set(STRATA) <= set(strata):
        raise ValueError("Mixing language/entity/fold/seed/evaluation requires an explicit policy")
    if not set(STRATA) <= set(strata) and not (experimental_policy and experimental_policy.strip()):
        raise ValueError("Declare experimental_policy when mixing experimental strata")
    scores = validate_performance(performance, scale=scale)
    rows, excluded, mapping = bind_scores(checkpoints, scores, score_alignment=score_alignment)
    identity = {
        "strategy": strategy,
        "k": k,
        "strata": list(strata),
        "tie_policy": tie_policy,
        "allow_mixed_strata": allow_mixed_strata,
        "experimental_policy": experimental_policy,
        "scores": performance_id(scores),
        "candidates": sorted(e.artifact_id for e in checkpoints.entries),
        "alignment": mapping,
    }
    selection_id = stable_id(identity)
    ranked, pairs = [], []
    if rows:
        candidates = pd.DataFrame(rows)
        for group_key, group in candidates.groupby(list(strata), dropna=False, sort=True):
            group_key = group_key if isinstance(group_key, tuple) else (group_key,)
            group = group.sort_values(
                ["strict_f1_mean", "artifact_id"], ascending=[False, True], kind="stable"
            ).copy()
            group["f1_rank"] = group.strict_f1_mean.rank(method="dense", ascending=False).astype(
                int
            )
            group["selection_order"] = range(1, len(group) + 1)
            group["selection_id"] = selection_id
            group["stratum_id"] = stable_id([None if pd.isna(x) else x for x in group_key])
            records = group.to_dict("records")
            ranked.extend(records)
            n = len(records)
            if n < 2 or strategy == "top_bottom" and n < 2 * k:
                excluded.extend(
                    {"artifact_id": r["artifact_id"], "reason": "insufficient_stratum_size"}
                    for r in records
                )
                continue
            top_count = k if strategy == "top_bottom" else 1
            top, bottom = records[:top_count], records[-top_count:]
            if tie_policy == "include_all":
                top = [r for r in records if r["strict_f1_mean"] >= top[-1]["strict_f1_mean"]]
                bottom = [r for r in records if r["strict_f1_mean"] <= bottom[0]["strict_f1_mean"]]
            if strategy in ("best_worst", "top_bottom"):
                if {r["artifact_id"] for r in top} & {r["artifact_id"] for r in bottom}:
                    raise ValueError("Ties produce overlapping top/bottom cohorts")
                chosen = product(top, bottom)
            elif strategy == "best_all":
                chosen = (
                    (a, b) for a in top for b in records if a["artifact_id"] != b["artifact_id"]
                )
            elif strategy == "worst_all":
                chosen = (
                    (a, b) for a in bottom for b in records if a["artifact_id"] != b["artifact_id"]
                )
            else:
                chosen = combinations(records, 2)
            seen = set()
            for a, b in chosen:
                left, right = sorted((a["artifact_id"], b["artifact_id"]))
                if (left, right) in seen:
                    continue
                seen.add((left, right))
                pair_id = stable_id({"checkpoints": [left, right]})
                same_space = (
                    a["parameter_space_id"] == b["parameter_space_id"]
                    and a["initialization_id"] == b["initialization_id"]
                )
                record = {
                    "comparison_id": pair_id,
                    "selection_id": selection_id,
                    "stratum_id": a["stratum_id"],
                    "artifact_a": left,
                    "artifact_b": right,
                    "performance_high": None
                    if a["strict_f1_mean"] == b["strict_f1_mean"]
                    else (
                        a["artifact_id"]
                        if a["strict_f1_mean"] > b["strict_f1_mean"]
                        else b["artifact_id"]
                    ),
                    "domain": "direct" if same_space else "intrinsic",
                    "interpretation": "descriptive_post_selection",
                }
                for side, cid in (("a", left), ("b", right)):
                    endpoint = a if a["artifact_id"] == cid else b
                    for field in (
                        "language",
                        "entity",
                        "backbone",
                        "fold",
                        "seed",
                        "score_id",
                        "score_scope",
                        "strict_f1_mean",
                        "strict_f1_std",
                        "f1_rank",
                    ):
                        record[f"{field}_{side}"] = endpoint[field]
                record["abs_f1_gap"] = abs(a["strict_f1_mean"] - b["strict_f1_mean"])
                record["performance_relation"] = "tie" if record["abs_f1_gap"] == 0 else "ordered"
                record["selection_pair_id"] = stable_id([selection_id, pair_id])
                pairs.append(record)
    return SelectionResult(
        checkpoints,
        scores,
        pd.DataFrame(ranked),
        pd.DataFrame(pairs),
        pd.DataFrame(excluded),
        {"selection_id": selection_id, **identity},
    )


def select_axis_pairs(
    *,
    checkpoints: CheckpointCatalog,
    axis: str,
    member_ids: Sequence[str] | None = None,
    performance: pd.DataFrame | None = None,
    score_alignment: Mapping[str, str] | None = None,
) -> SelectionResult:
    """Select compatible cross-entity/language pairs without ranking different tasks by F1."""
    if axis not in ("entity", "language"):
        raise ValueError("axis must be entity or language")
    scores = validate_performance(performance) if performance is not None else pd.DataFrame()
    bound = {}
    if performance is not None:
        bindings, _, _ = bind_scores(checkpoints, scores, score_alignment=score_alignment)
        bound = {r["artifact_id"]: r for r in bindings}
    fixed = "language" if axis == "entity" else "entity"
    groups = {}
    for entry in checkpoints.entries:
        if member_ids is not None and entry.artifact_id not in member_ids:
            continue
        cp = entry.checkpoint
        key = (
            cp.backbone,
            getattr(cp, fixed),
            cp.fold,
            cp.seed,
            entry.parameter_space_id,
            entry.evidence.initialization_id,
        )
        groups.setdefault(key, []).append(entry)
    if member_ids is not None and set(member_ids) - set(checkpoints.by_id):
        raise ValueError("Unknown member_ids")
    sid = stable_id(
        {
            "axis": axis,
            "groups": [
                (key, sorted(e.artifact_id for e in group))
                for key, group in sorted(groups.items(), key=lambda x: str(x[0]))
            ],
            "scores": performance_id(scores) if performance is not None else None,
        }
    )
    rows = []
    for key, group in groups.items():
        if len({getattr(e.checkpoint, axis) for e in group}) != len(group):
            raise ValueError("Select one checkpoint per axis value before comparing groups")
        for a, b in combinations(sorted(group, key=lambda e: e.artifact_id), 2):
            cid = stable_id({"checkpoints": [a.artifact_id, b.artifact_id]})
            record = {
                "comparison_id": cid,
                "selection_id": sid,
                "stratum_id": stable_id(key),
                "selection_pair_id": stable_id([sid, cid]),
                "artifact_a": a.artifact_id,
                "artifact_b": b.artifact_id,
                "domain": "direct",
                "interpretation": "descriptive_cross_" + axis,
            }
            for side, entry in (("a", a), ("b", b)):
                for field in ("language", "entity", "backbone", "fold", "seed"):
                    record[field + "_" + side] = getattr(entry.checkpoint, field)
                for field in ("score_id", "score_scope", "strict_f1_mean", "strict_f1_std"):
                    record[field + "_" + side] = bound.get(entry.artifact_id, {}).get(field)
            values = [record["strict_f1_mean_" + side] for side in ("a", "b")]
            record["abs_f1_gap"] = (
                abs(values[0] - values[1]) if all(v is not None for v in values) else None
            )
            rows.append(record)
    return SelectionResult(
        checkpoints,
        scores,
        pd.DataFrame(),
        pd.DataFrame(rows),
        pd.DataFrame(),
        {"selection_id": sid, "axis": axis, "interpretation": "descriptive_geometry"},
    )
