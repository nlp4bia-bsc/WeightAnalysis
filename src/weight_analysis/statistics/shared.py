"""Mergeable centroid/projection statistics and explicit group-level finalization."""

from __future__ import annotations

import pandas as pd

from ..core.catalog import stable_id
from ..core.geometry_schema import observation
from ..metrics.centroids import projection_summary

SHARED_METRICS = (
    "mean_energy",
    "mean_norm",
    "dispersion",
    "update_energy",
    "distance_to_mean",
    "centered_energy",
    "projection_fraction",
    "projection_coefficient",
    "direction_residual_energy",
    "alignment_cosine",
    "loo_distance_to_mean",
    "loo_projection_fraction",
    "loo_alignment_cosine",
    "loo_direction_residual_energy",
)
INVARIANT = {"dispersion", "distance_to_mean", "centered_energy", "loo_distance_to_mean"}


def finalize_shared(records, *, metrics, scope="tensor"):
    """Reduce disjoint tensor statistics first, then finalize nonlinear group geometry."""
    rows = []
    by_member = {}
    for record in sorted(records, key=lambda r: (r["member_id"], r["tensor"])):
        by_member.setdefault(record["member_id"], []).append(record)
    if not by_member:
        return rows
    sums = {}
    for member, parts in by_member.items():
        if len({p["tensor"] for p in parts}) != len(parts):
            raise ValueError("Overlapping shared-adaptation tensor statistics")
        sums[member] = {
            key: sum(p[key] for p in parts)
            for key in (
                "update_energy",
                "mean_energy",
                "dot_mean",
                "centered_energy",
                "loo_mean_energy",
                "loo_dot",
                "loo_centered_energy",
                "n_parameters",
            )
        }
    expected = records[0]["n_members"]
    if len(sums) != expected:
        raise ValueError("Group statistics omit members of the frozen universe")
    dispersion = sum(s["centered_energy"] for s in sums.values()) / expected
    for member, values in sorted(sums.items()):
        first = by_member[member][0]
        location = {
            key: first.get(key) for key in ("layer", "component", "embedding_type", "tensor")
        }
        if scope != "tensor":
            location["tensor"] = None
        if scope not in ("tensor", "layer_component", "layer"):
            location["layer"] = None
        if scope not in ("tensor", "layer_component", "component"):
            location["component"] = location["embedding_type"] = None
        projection = projection_summary(
            values["update_energy"], values["mean_energy"], values["dot_mean"]
        )
        loo = projection_summary(
            values["update_energy"], values["loo_mean_energy"], values["loo_dot"]
        )
        measured = {
            "mean_energy": values["mean_energy"],
            "mean_norm": values["mean_energy"] ** 0.5,
            "dispersion": dispersion,
            "update_energy": values["update_energy"],
            "distance_to_mean": values["centered_energy"] ** 0.5,
            "centered_energy": values["centered_energy"],
            "loo_distance_to_mean": values["loo_centered_energy"] ** 0.5,
            **projection,
            **{"loo_" + key: value for key, value in loo.items()},
        }
        for metric in metrics:
            group_metric = metric in ("mean_energy", "mean_norm", "dispersion")
            if group_metric and member != sorted(sums)[0]:
                continue
            reference = None if metric in INVARIANT else first["reference_id"]
            meta = {
                **{
                    key: first.get(key)
                    for key in ("language", "entity", "backbone", "fold", "seed", "comparison_type")
                },
                **location,
                "subject_id": first["group_id"] if group_metric else first["membership_id"],
                "subject_kind": "group" if group_metric else "member_to_group",
                "group_id": first["group_id"],
                "member_id": None if group_metric else member,
                "source": "final" if metric in INVARIANT else "delta",
                "reference_id": reference,
                "n_parameters": int(values["n_parameters"]),
                "n_members": expected,
                "execution_id": first["execution_id"],
                "axis": first["axis"],
            }
            uid = stable_id([meta, metric, scope])
            row = observation(
                uid,
                meta,
                metric,
                measured[metric],
                scope=scope,
                interpretation="descriptive_geometry_no_causal_claim",
                self_included=not metric.startswith("loo_"),
            )
            if (
                "direction_residual_energy" in metric
                and measured[metric] <= 1e-10 * values["update_energy"]
            ):
                row["status"] = "roundoff_limited" if row["status"] == "ok" else row["status"]
            rows.append(row)
    return rows


def aggregate_shared_geometry(result, *, scopes):
    stats = result.tables.get("shared_statistics", pd.DataFrame())
    if stats.empty:
        return []
    rows = []
    for scope in scopes:
        extra = {
            "layer_component": ["layer", "component", "embedding_type"],
            "layer": ["layer"],
            "component": ["component", "embedding_type"],
            "encoder": [],
        }[scope]
        for _, frame in stats.groupby(
            ["group_id", "reference_id", "execution_id", "metric_names"] + extra,
            dropna=False,
            sort=True,
        ):
            rows.extend(
                finalize_shared(
                    frame.to_dict("records"),
                    metrics=frame.iloc[0].metric_names.split(","),
                    scope=scope,
                )
            )
    return rows
