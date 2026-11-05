"""Audited experimental NER aggregates; never manufacture fold-level observations."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import __version__
from ..core.catalog import stable_id
from ..provenance import file_sha256
from ..results import AnalysisResult, write_result


@dataclass(frozen=True)
class SourceColumns:
    language: str = "language"
    entity: str = "entity"
    backbone: str = "model"
    n_folds: str = "n_folds"
    mean: str = "strict_f1_mean"
    std: str = "strict_f1_std"


def code_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    return stable_id({str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*.py"))})


def runtime_versions():
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("torch", "pandas", "pyarrow", "numpy", "scipy", "safetensors")
        },
    }


def read_ner_source(path, *, columns=SourceColumns(), protocol_id=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mandatory NER source unavailable: {path}; F1 stages blocked")
    initial_digest = file_sha256(path)
    parquet = pq.ParquetFile(path)
    raw = parquet.read().to_pandas()
    mapping = {
        columns.language: "language",
        columns.entity: "entity",
        columns.backbone: "backbone",
        columns.n_folds: "n_folds",
        columns.mean: "strict_f1_mean",
        columns.std: "strict_f1_std",
    }
    if len(mapping) != 6 or set(mapping) - set(raw):
        raise ValueError("Explicit NER column mapping is missing, overlapping or absent in source")
    frame = raw[list(mapping)].rename(columns=mapping).copy()
    for key in ("language", "entity", "backbone"):
        if not frame[key].map(lambda v: isinstance(v, str) and bool(v) and v == v.strip()).all():
            raise ValueError(f"Missing, non-string or whitespace-inconsistent identity: {key}")
    identity = ["language", "entity", "backbone"]
    if frame.duplicated(identity).any():
        raise ValueError(
            "Ambiguous duplicate NER configuration identities; supply a richer mapping"
        )
    numeric = frame[["n_folds", "strict_f1_mean", "strict_f1_std"]].apply(
        pd.to_numeric, errors="raise"
    )
    if (
        not np.isfinite(numeric).all().all()
        or not numeric.strict_f1_mean.between(0, 1).all()
        or (numeric.strict_f1_std < 0).any()
    ):
        raise ValueError("Invalid strict F1 mean/SD; require finite unit-scale scores")
    if not ((numeric.n_folds >= 1) & (numeric.n_folds % 1 == 0)).all():
        raise ValueError("Invalid number of folds")
    frame[numeric.columns] = numeric
    digest = file_sha256(path)
    if digest != initial_digest:
        raise ValueError("NER source changed while being read; F1 stages blocked")
    frame["model_id"] = [
        stable_id(dict(zip(identity, row)))
        for row in frame[identity].itertuples(index=False, name=None)
    ]
    frame["score_id"] = [stable_id([digest, model]) for model in frame.model_id]
    frame["score_scope"] = "configuration_aggregate_unknown_fold_ids"
    frame["fold"] = None
    frame["seed"] = None
    frame["protocol_id"] = protocol_id
    frame["protocol_status"] = "declared" if protocol_id else "not_supplied_descriptive_only"
    frame["evaluation_id"] = "source:" + digest
    frame["experimental_configuration_id"] = None
    frame["interpretation"] = "descriptive_aggregate_F1_not_checkpoint_F1"
    frame["source_sha256"] = digest
    universe = set(product(*(sorted(frame[k].unique()) for k in identity)))
    actual = set(frame[identity].itertuples(index=False, name=None))
    provenance = {
        "source_path": str(path),
        "source_sha256": digest,
        "source_bytes": path.stat().st_size,
        "schema": str(parquet.schema_arrow),
        "schema_fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in parquet.schema_arrow
        ],
        "schema_metadata": {
            k.decode(): v.decode() for k, v in (parquet.schema_arrow.metadata or {}).items()
        },
        "columns_used": mapping,
        "rows": len(raw),
        "null_counts": {k: int(v) for k, v in raw.isna().sum().items()},
        "exact_duplicates": int(raw.duplicated().sum()),
        "identity_duplicates": 0,
        "unique_backbones": int(frame.backbone.nunique()),
        "languages": sorted(frame.language.unique()),
        "entities": sorted(frame.entity.unique()),
        "missing_cells": [dict(zip(identity, row)) for row in sorted(universe - actual)],
        "zero_F1_rows": int(frame.strict_f1_mean.eq(0).sum()),
        "strict_f1_range": [float(frame.strict_f1_mean.min()), float(frame.strict_f1_mean.max())]
        if len(frame)
        else None,
        "strict_f1_std_range": [float(frame.strict_f1_std.min()), float(frame.strict_f1_std.max())]
        if len(frame)
        else None,
        "aggregation": "source mean and SD across n_folds; fold identities, seeds and configurations not supplied",
        "protocol_id": protocol_id,
        "protocol_status": frame.protocol_status.iloc[0] if len(frame) else "no_data",
        "checkpoint_matching": "exact language/entity/backbone path; hashes identify physical weights; never fuzzy",
        "models_found": len(frame),
        "models_matched": None,
        "models_unmatched": None,
        "matching_status": "pending_weight_inventory",
        "package_version": __version__,
        "code_digest": code_digest(),
        "runtime_versions": runtime_versions(),
    }
    return frame.sort_values(identity).reset_index(drop=True), provenance


def _rank(frame, groups):
    frame = frame.sort_values(
        groups + ["strict_f1_mean", "model_id"],
        ascending=[True] * len(groups) + [False, True],
        kind="stable",
    ).copy()
    frame["f1_rank"] = (
        frame.groupby(groups, dropna=False)
        .strict_f1_mean.rank(method="dense", ascending=False)
        .astype(int)
    )
    frame["selection_order"] = frame.groupby(groups, dropna=False).cumcount() + 1
    return frame


def _macro(scores, groups, varying, universe):
    rows = []
    for key, frame in scores.groupby(groups, dropna=False, sort=True):
        actual = set(frame[varying].itertuples(index=False, name=None))
        if actual != universe:
            continue  # complete balanced coverage only; no mean over surviving tasks
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        row.update(
            strict_f1_mean=float(frame.strict_f1_mean.mean()),
            strict_f1_std=None,
            source_score_ids=sorted(frame.score_id),
            source_f1_std=frame.sort_values("score_id").strict_f1_std.tolist(),
            n_tasks=len(frame),
            n_folds=int(frame.n_folds.min()),
            model_id=stable_id([row, "macro"]),
            score_scope="macro_of_fold_aggregates",
            aggregation="equal_weight_complete_cells",
            interpretation="cross_task_descriptive_only",
            std_status="not_identifiable_without_fold_covariances",
        )
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_tables(scores, *, k=3):
    if k < 1:
        raise ValueError("k must be positive")
    raw = _rank(scores, ["language", "entity"])
    raw["ranking_scope"] = "language_entity"
    raw["is_best"] = raw.selection_order.eq(1)
    sizes = raw.groupby(["language", "entity"]).model_id.transform("count")
    raw["is_worst"] = raw.selection_order.eq(sizes)
    raw["is_top_k"] = raw.selection_order.le(k)
    raw["is_bottom_k"] = raw.selection_order.gt(sizes - k)
    raw["top_bottom_eligible"] = sizes.ge(2 * k)
    raw["rank_language_backbone"] = (
        _rank(scores, ["language", "backbone"])
        .set_index("model_id")
        .selection_order.reindex(raw.model_id)
        .to_numpy()
    )
    raw["rank_entity_backbone"] = (
        _rank(scores, ["entity", "backbone"])
        .set_index("model_id")
        .selection_order.reindex(raw.model_id)
        .to_numpy()
    )
    raw["rank_language_entity_backbone"] = (
        1  # one experimental aggregate per cell, not a variant ranking
    )
    raw["cross_task_rank_interpretation"] = (
        "descriptive_cross_entity_or_language_not_comparable_tasks"
    )
    languages, entities = sorted(scores.language.unique()), sorted(scores.entity.unique())
    by_language = _macro(scores, ["language", "backbone"], ["entity"], {(x,) for x in entities})
    by_entity = _macro(scores, ["entity", "backbone"], ["language"], {(x,) for x in languages})
    by_backbone = _macro(
        scores, ["backbone"], ["language", "entity"], set(product(languages, entities))
    )
    if not by_language.empty:
        by_language = _rank(by_language, ["language"])
    if not by_entity.empty:
        by_entity = _rank(by_entity, ["entity"])
    if not by_backbone.empty:
        by_backbone["ranking_scope"] = "complete_grid_macro"
        by_backbone = _rank(by_backbone, ["ranking_scope"])
    pairs = []
    for _, group in raw.groupby(["language", "entity"], sort=True):
        for a, b in combinations(group.to_dict("records"), 2):
            a, b = sorted((a, b), key=lambda row: row["model_id"])
            record = {
                "performance_pair_id": stable_id(sorted([a["score_id"], b["score_id"]])),
                "language": a["language"],
                "entity": a["entity"],
                "comparison_type": "cross_backbone_intrinsic",
                "abs_f1_gap": abs(a["strict_f1_mean"] - b["strict_f1_mean"]),
                "is_best_worst": (a["is_best"] and b["is_worst"])
                or (b["is_best"] and a["is_worst"]),
                "is_top_bottom": (
                    (a["is_top_k"] and b["is_bottom_k"]) or (b["is_top_k"] and a["is_bottom_k"])
                )
                and a["top_bottom_eligible"]
                and b["top_bottom_eligible"],
                "interpretation": "descriptive_post_selection_aggregate_F1",
            }
            for side, endpoint in (("a", a), ("b", b)):
                for field in (
                    "model_id",
                    "score_id",
                    "backbone",
                    "strict_f1_mean",
                    "strict_f1_std",
                    "n_folds",
                    "score_scope",
                    "protocol_status",
                    "selection_order",
                ):
                    record[field + "_" + side] = endpoint[field]
            pairs.append(record)
    all_pairs = pd.DataFrame(pairs)
    empty = all_pairs.copy()
    return {
        "f1_model_ranking": raw,
        "f1_ranking_by_language": by_language,
        "f1_ranking_by_entity": by_entity,
        "f1_ranking_by_backbone": by_backbone,
        "f1_best_worst_pairs": all_pairs[all_pairs.is_best_worst].copy()
        if not all_pairs.empty
        else empty,
        "f1_top_bottom_pairs": all_pairs[all_pairs.is_top_bottom].copy()
        if not all_pairs.empty
        else empty,
        "f1_all_valid_pairs": all_pairs,
    }


def write_source_audit(path, output_dir, *, k=3, protocol_id=None):
    scores, audit = read_ner_source(path, protocol_id=protocol_id)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    audit.update(
        ranking="strict F1 descending; dense rank; tie break SHA256 model identity ascending",
        macro_policy="equal weights and complete task grid only; no aggregate SD invented",
        top_k=k,
    )
    for name, frame in ranking_tables(scores, k=k).items():
        write_result(AnalysisResult(name, frame, audit), root / f"{name}.parquet")
    (root / "source_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf8"
    )
    return scores, audit
