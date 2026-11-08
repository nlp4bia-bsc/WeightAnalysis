"""Tensor/aggregate observations derived exclusively from persisted statistics."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from ...core.catalog import stable_id
from .primitives import DRIFT_METRICS, PAIR_METRICS, moments, shared_values
from .storage import record

INVARIANT = {"l2_distance", "rmse", "l1_mean_distance", "max_abs_delta"}


def family(a, b):
    if a["language"] == b["language"]:
        return "same_language_cross_entity"
    if a["entity"] == b["entity"]:
        return "same_entity_cross_language"
    return "cross_language_cross_entity"


def endpoint_metadata(a, b=None):
    result = dict(
        backbone=a["backbone"],
        artifact_a=a["artifact_id"],
        reference_id=a["reference_content_id"],
        reference_kind=a["reference"]["kind"],
    )
    for side, value in (("a", a), ("b", b)):
        for key in ("language", "entity", "fold", "seed"):
            result[key + "_" + side] = value[key] if value is not None else None
    result["artifact_b"] = b["artifact_id"] if b is not None else None
    result["language"] = (
        a["language"] if b is None or a["language"] == b["language"] else "__cross_language__"
    )
    result["entity"] = (
        a["entity"] if b is None or a["entity"] == b["entity"] else "__cross_entity__"
    )
    return result


def geometry_rows(state, members, metadata):
    rows = []
    for i, entry in enumerate(members, 1):
        meta = dict(
            metadata, **endpoint_metadata(entry), family="checkpoint_vs_pretraining", source="delta"
        )
        values = moments(state, "final", i, 0)
        for metric in DRIFT_METRICS:
            rows.append(
                record(meta, kind="geometry", metric=metric, value=values.value(metric), n=values.n)
            )
    for (i, a), (j, b) in combinations(enumerate(members, 1), 2):
        for source in ("final", "delta"):
            values = moments(state, source, i, j)
            meta = dict(metadata, **endpoint_metadata(a, b), family=family(a, b), source=source)
            for metric in PAIR_METRICS:
                if source == "delta" and metric in INVARIANT:
                    continue
                rows.append(
                    record(
                        meta,
                        kind="geometry",
                        metric=metric,
                        value=values.value(metric),
                        n=values.n,
                        exactness="exact_float64",
                    )
                )
    for group in state["groups"]:
        values = state["shared"][group["group_id"]]
        entries = [members[index - 1] for index in group["indices"]]
        meta = dict(
            metadata,
            family=group["family"],
            source="delta",
            group_id=group["group_id"],
            reference_id=entries[0]["reference_content_id"],
            reference_kind="pretraining",
        )
        languages = {entry["language"] for entry in entries}
        entities = {entry["entity"] for entry in entries}
        meta.update(
            language=next(iter(languages)) if len(languages) == 1 else "__cross_language__",
            entity=next(iter(entities)) if len(entities) == 1 else "__cross_entity__",
        )
        rows.append(
            record(
                meta,
                kind="geometry",
                metric="mean_update_energy",
                value=values[0][1],
                n=state["n"],
                payload=dict(size=len(values), members=group["members"]),
            )
        )
        rows.append(
            record(
                meta,
                kind="geometry",
                metric="dispersion_energy",
                value=math.fsum(v[3] for v in values) / len(values),
                n=state["n"],
            )
        )
        for entry, value in zip(entries, values):
            derived = shared_values(value, len(values))
            endpoint = dict(meta, **endpoint_metadata(entry))
            for metric, number in derived.items():
                if metric in (
                    "update_energy",
                    "mean_update_energy",
                    "dispersion_energy",
                    "projection_roundoff_bound",
                ):
                    continue
                rows.append(
                    record(
                        endpoint,
                        kind="geometry",
                        metric=metric,
                        value=number,
                        n=state["n"],
                        payload=dict(
                            projection_roundoff_bound=derived["projection_roundoff_bound"]
                        ),
                    )
                )
    complete = next((g for g in state["groups"] if g["family"] == "mean_by_backbone"), None)
    if complete is not None:
        gram = np.asarray(state["data"]["delta_gram"])[1:, 1:]
        cells = [(row["language"], row["entity"]) for row in members]
        languages, entities = sorted({x[0] for x in cells}), sorted({x[1] for x in cells})
        if len(cells) == len(languages) * len(entities):
            center = np.full(len(cells), 1 / len(cells))

            def energy(coef):
                return max(0, float(coef @ gram @ coef))

            total = math.fsum(energy(np.eye(len(cells))[i] - center) for i in range(len(cells)))
            lang = math.fsum(
                energy(np.array([float(c[0] == l) / len(entities) for c in cells]) - center)
                for l in languages
            ) * len(entities)
            entity = math.fsum(
                energy(np.array([float(c[1] == e) / len(languages) for c in cells]) - center)
                for e in entities
            ) * len(languages)
            meta = dict(
                metadata,
                family="factorial_language_entity",
                source="delta",
                group_id=complete["group_id"],
            )
            for metric, number in (
                ("ss_total", total),
                ("ss_language", lang),
                ("ss_entity", entity),
                ("ss_interaction", max(0, total - lang - entity)),
            ):
                rows.append(
                    record(meta, kind="geometry", metric=metric, value=number, n=state["n"])
                )
    return rows


def unit_identity(plan, task, tensor, kind, artifact=None, source=None):
    return stable_id([plan["run_id"], task["task_id"], tensor, kind, artifact, source])
