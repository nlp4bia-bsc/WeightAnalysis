"""Explicit F1 views and descriptive associations of normalized geometric observations."""

from __future__ import annotations

import pandas as pd

from ..core.catalog import stable_id
from .performance import correlation_row


def comparison_view(result, *, table="geometry_detailed", selection_id=None):
    """Join performance at the display/statistics boundary, never into geometry cache keys."""
    geometry = result.tables.get(table, pd.DataFrame())
    links = result.tables.get("selection_pairs", pd.DataFrame())
    if geometry.empty or links.empty:
        return pd.DataFrame()
    if selection_id is not None:
        links = links[links.selection_id == selection_id]
    elif links.selection_id.nunique() > 1:
        raise ValueError("Choose one selection_id before making a comparative view")
    geometry = geometry.drop(columns=["domain"], errors="ignore")
    frame = geometry[geometry.subject_kind == "pair"].merge(
        links, left_on="subject_id", right_on="comparison_id", how="inner", validate="m:1"
    )
    if frame.empty:
        return frame
    comparisons = result.tables.get("comparisons", pd.DataFrame())
    fields = [
        name
        for name in (
            "comparison_id",
            "comparison_type",
            "language",
            "entity",
            "backbone",
            "fold",
            "seed",
        )
        if name in comparisons and name not in frame
    ]
    if "comparison_id" in fields and len(fields) > 1:
        frame = frame.merge(comparisons[fields], on="comparison_id", validate="m:1")
    return frame


def member_performance_view(result, *, table="geometry_detailed"):
    frame = result.tables.get(table, pd.DataFrame())
    members = result.tables.get("group_members", pd.DataFrame())
    if frame.empty or members.empty or "strict_f1_mean" not in members:
        return pd.DataFrame()
    return frame[frame.subject_kind == "member_to_group"].merge(
        members,
        left_on="subject_id",
        right_on="membership_id",
        validate="m:1",
        suffixes=("", "_member"),
    )


def associate_f1(result, *, table="geometry_detailed", selection_id=None):
    """Report correlations descriptively with explicit counts of shared endpoints/scores."""
    from ..results.comparative import ComparativeResult

    frame = comparison_view(result, table=table, selection_id=selection_id)
    rows = []
    if not frame.empty:
        strata = [
            column
            for column in (
                "language_a",
                "entity_a",
                "fold_a",
                "seed_a",
                "metric",
                "scope",
                "layer",
                "tensor",
                "relative_depth",
                "execution_id",
                "stratum_id",
                "component",
                "embedding_type",
                "source",
                "reference_id",
            )
            if column in frame
        ]
        for _, group in frame.groupby(strata, dropna=False, sort=True):
            if group.comparison_id.duplicated().any():
                raise ValueError(
                    "Association input duplicates pair observations; select an explicit scope"
                )
            identity = {column: group.iloc[0][column] for column in strata}
            row = correlation_row(
                identity,
                "nonindependent_pairs",
                group.iloc[0].metric,
                "abs_f1_gap",
                group.value,
                group.abs_f1_gap,
            )
            row.update(
                n_pairs=len(group),
                n_checkpoints=len(set(group.artifact_a) | set(group.artifact_b)),
                n_scores=len(set(group.score_id_a) | set(group.score_id_b)),
                interpretation="descriptive_post_selection_no_iid_inference",
                note="Shared checkpoints and aggregated F1 are not independent replicates",
            )
            row["association_id"] = stable_id(
                [identity, sorted(group.observation_id), sorted(group.selection_id)]
            )
            rows.append(row)
    return ComparativeResult(
        {**result.tables, "associations": pd.DataFrame(rows)}, result.provenance
    )


def associate_member_f1(result, *, table="geometry_detailed"):
    """Describe group-member geometry versus exact/aggregate F1 without iid inference."""
    from ..results.comparative import ComparativeResult

    frame = member_performance_view(result, table=table)
    rows = []
    if not frame.empty:
        columns = [
            name
            for name in (
                "group_id",
                "metric",
                "scope",
                "tensor",
                "layer",
                "component",
                "embedding_type",
                "source",
                "reference_id",
                "execution_id",
            )
            if name in frame
        ]
        for _, group in frame.groupby(columns, dropna=False, sort=True):
            if group.membership_id.duplicated().any():
                raise ValueError("Choose one observation per member before associating F1")
            identity = {column: group.iloc[0][column] for column in columns}
            row = correlation_row(
                identity,
                "declared_group_members",
                group.iloc[0].metric,
                "strict_f1_mean",
                group.value,
                group.strict_f1_mean,
            )
            row.update(
                n_checkpoints=group.artifact_id.nunique(),
                n_scores=group.score_id.nunique(),
                interpretation="descriptive_geometry_no_iid_inference",
                note="Group members share a fitted geometric direction; aggregate scores retain their scope",
            )
            row["association_id"] = stable_id(
                [identity, sorted(group.observation_id), sorted(group.score_id)]
            )
            rows.append(row)
    return ComparativeResult(
        {**result.tables, "associations": pd.DataFrame(rows)}, result.provenance
    )
