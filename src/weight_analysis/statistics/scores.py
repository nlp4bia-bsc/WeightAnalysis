"""Strict span-F1 tables with explicit checkpoint versus aggregate applicability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.catalog import stable_id


def validate_performance(frame: pd.DataFrame, *, scale="unit") -> pd.DataFrame:
    """Normalize explicitly declared F1 units without guessing score semantics or joins."""
    required = {
        "score_id",
        "strict_f1_mean",
        "strict_f1_std",
        "score_scope",
        "protocol_id",
        "evaluation_id",
    }
    if required - set(frame):
        raise ValueError(f"Performance table missing {sorted(required - set(frame))}")
    if scale not in ("unit", "percent"):
        raise ValueError("scale must be unit or percent")
    result = frame.copy()
    for column in ("score_id", "score_scope", "protocol_id", "evaluation_id"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing performance identity: {column}")
    if result.score_id.duplicated().any():
        raise ValueError("Duplicate score_id values")
    for column in ("strict_f1_mean", "strict_f1_std"):
        result[column] = pd.to_numeric(result[column], errors="raise") / (
            100 if scale == "percent" else 1
        )
    if not np.isfinite(result[["strict_f1_mean", "strict_f1_std"]]).all().all():
        raise ValueError("Nonfinite performance values")
    if not result.strict_f1_mean.between(0, 1).all() or (result.strict_f1_std < 0).any():
        raise ValueError("F1 must be in [0,1] and SD must be nonnegative")
    if not result.score_scope.isin(["checkpoint", "configuration_aggregate"]).all():
        raise ValueError("Unknown score_scope")
    for row in result.to_dict("records"):
        if row["score_scope"] == "checkpoint":
            if pd.isna(row.get("artifact_id")) or not row["artifact_id"]:
                raise ValueError("Checkpoint scores must identify their exact artifact")
        else:
            for column in ("configuration_id", "n_folds", "n_seeds"):
                if column not in row or pd.isna(row[column]):
                    raise ValueError(f"Aggregate scores must declare {column}")
            for column in ("folds", "seeds"):
                values = row.get(column)
                if not isinstance(values, (list, tuple, np.ndarray)) or not len(values):
                    raise ValueError(f"Aggregate scores must list {column}")
                if len(set(values)) != len(values) or any(pd.isna(value) for value in values):
                    raise ValueError(
                        f"Aggregate {column} must contain unique nonmissing identities"
                    )
            if len(set(row["folds"])) != row["n_folds"] or len(set(row["seeds"])) != row["n_seeds"]:
                raise ValueError("Aggregate replication counts disagree with listed folds/seeds")
    return result.sort_values("score_id").reset_index(drop=True)


def performance_id(frame):
    from ..results.models import data_digest

    return stable_id({"performance": data_digest(frame), "definition": "strict-span-f1"})


def bind_scores(checkpoints, scores, *, score_alignment=None):
    """Bind validated scores using explicit artifact or aggregate configuration identities."""
    mapping = dict(score_alignment or {})
    by_score = scores.set_index("score_id").to_dict("index")
    if set(mapping) - set(checkpoints.by_id) or set(mapping.values()) - set(by_score):
        raise ValueError("Score alignment contains unknown checkpoint or score identities")
    checkpoint_scores = {
        str(row["artifact_id"]): row["score_id"]
        for row in scores.to_dict("records")
        if row["score_scope"] == "checkpoint"
    }
    if len(checkpoint_scores) != int(scores.score_scope.eq("checkpoint").sum()):
        raise ValueError(
            "Several performance records target the same checkpoint; select a protocol explicitly"
        )
    rows, excluded = [], []
    for entry in checkpoints.entries:
        sid = mapping.get(entry.artifact_id, checkpoint_scores.get(entry.artifact_id))
        if sid not in by_score:
            excluded.append(
                {"artifact_id": entry.artifact_id, "reason": "missing_applicable_score"}
            )
            continue
        score = by_score[sid]
        if score["score_scope"] == "checkpoint" and score.get("artifact_id") != entry.artifact_id:
            raise ValueError("Score alignment targets a different checkpoint")
        if score["score_scope"] == "configuration_aggregate":
            if (
                entry.artifact_id not in mapping
                or score["configuration_id"] != entry.evidence.configuration_id
            ):
                raise ValueError("Aggregate F1 requires an explicit matching configuration mapping")
            if (
                entry.checkpoint.fold not in score["folds"]
                or entry.checkpoint.seed not in score["seeds"]
            ):
                raise ValueError("Checkpoint fold/seed are outside the performance aggregate")
        for field in ("backbone", "language", "entity", "fold", "seed"):
            if (
                field in score
                and not pd.isna(score[field])
                and score["score_scope"] == "checkpoint"
            ):
                if score[field] != getattr(entry.checkpoint, field):
                    raise ValueError(f"Performance identity mismatch: {field}")
        rows.append(
            {
                **entry.row(),
                **{
                    key: score[key]
                    for key in (
                        "strict_f1_mean",
                        "strict_f1_std",
                        "score_scope",
                        "protocol_id",
                        "evaluation_id",
                    )
                },
                "score_id": sid,
            }
        )
    return rows, excluded, mapping
