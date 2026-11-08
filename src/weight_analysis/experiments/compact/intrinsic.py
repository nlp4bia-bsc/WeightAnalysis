"""Coordinate-free comparisons of complete spectral/depth fingerprints only."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import pyarrow.dataset as ds

from ...core.catalog import stable_id
from ...metrics import paper_trace_similarity, quantile_w1
from .storage import SCHEMA, PackedStore, record


def _profiles(plan, rows):
    payloads = {row["unit_id"]: json.loads(row["payload"]) for row in rows}

    def resolved(row):
        value = payloads[row["unit_id"]]
        return payloads[value["alias_unit_id"]] if "alias_unit_id" in value else value

    expected = defaultdict(set)
    for task in plan["tasks"]:
        for tensor in task["tensors"]:
            if len(tensor["shape"]) == 2:
                for aid in tensor["members"]:
                    expected[aid].add(tensor["tensor"])
    grouped = defaultdict(list)
    for row in rows:
        if row["artifact_a"] and row["source"] in ("final", "delta"):
            grouped[row["artifact_a"], row["source"]].append(row)
    profiles, exclusions = {}, []
    for key, items in grouped.items():
        if len({item["execution_id"] for item in items}) > 1:
            exclusions.append(
                dict(artifact_id=key[0], source=key[1], reason="mixed_execution_profile")
            )
            continue
        actual = {item["tensor"] for item in items}
        if actual != expected[key[0]]:
            exclusions.append(
                dict(
                    artifact_id=key[0],
                    source=key[1],
                    reason="incomplete_fingerprint_coverage",
                    missing=sorted(expected[key[0]] - actual),
                )
            )
            continue
        components = defaultdict(list)
        for row in items:
            label = row["component"] + (
                ":" + row["embedding_type"] if row["embedding_type"] else ""
            )
            components[label].append((row, resolved(row)))
        result = {}
        for component, matrices in sorted(components.items()):
            depths = defaultdict(list)
            for row, values in matrices:
                depths[row["relative_depth"] if row["relative_depth"] is not None else -1].append(
                    values
                )
            axis = sorted(depths)
            feature = dict(depth=axis, rms=[], curves=[], quantiles=[], rank_fraction=[], exact=[])
            for depth in axis:
                values = depths[depth]
                # RMS uses sums of energy, never an average of norms.
                energy = math.fsum(float(v["matrix_fro"]) ** 2 for v in values)
                count = sum(v["n_parameters"] for v in values)
                feature["rms"].append(math.sqrt(energy / count))
                feature["exact"].append(
                    all(v["status"] in ("exact", "zero_matrix") for v in values)
                )
                for name, column in (
                    ("curves", "spectral_energy_curve"),
                    ("quantiles", "normalized_weight_quantiles"),
                ):
                    arrays = [v.get(column) for v in values]
                    feature[name].append(
                        np.mean(arrays, axis=0).tolist()
                        if all(x is not None for x in arrays)
                        else None
                    )
                ranks = [v.get("effective_rank_fraction") for v in values]
                feature["rank_fraction"].append(
                    float(np.mean(ranks)) if all(v is not None for v in ranks) else None
                )
            result[component] = feature
        profiles[key] = result
    return profiles, exclusions


def _resample(feature, name, points=32):
    values = feature[name]
    if any(value is None for value in values):
        return None
    array = np.asarray(values)
    if len(values) == 1:
        return np.repeat(array[:1], points, axis=0)
    query = np.linspace(feature["depth"][0], feature["depth"][-1], points)
    if array.ndim == 1:
        return np.interp(query, feature["depth"], array)
    return np.stack(
        [np.interp(query, feature["depth"], array[:, i]) for i in range(array.shape[1])], axis=1
    )


def derive_intrinsic(root, plan, manifests, generation):
    files = [str(root / item["path"]) for item in manifests if item["area"] == "atomic"]
    if not files or plan["config"]["spectral_mode"] == "skip":
        return []
    table = ds.dataset(files, format="parquet", schema=SCHEMA).to_table(
        filter=ds.field("kind") == "fingerprint"
    )
    profiles, exclusions = _profiles(plan, table.to_pylist())
    records = {r["artifact_id"]: r for r in plan["checkpoints"]}
    cells = defaultdict(list)
    for aid in sorted({key[0] for key in profiles}):
        row = records[aid]
        cells[row["language"], row["entity"], row["fold"], row["seed"]].append(aid)
    config = plan["config"]
    with PackedStore(
        root,
        "intrinsic",
        area="intrinsic",
        policy=config["storage_policy"],
        target_bytes=config["target_part_bytes"],
    ) as store:
        store.generation(generation)
        for aid, source in sorted(profiles):
            uid = stable_id(["intrinsic_profile", aid, source])
            if not store.done(uid):
                r = records[aid]
                store.commit(
                    uid,
                    [
                        record(
                            dict(
                                unit_id=uid,
                                artifact_a=aid,
                                source=source,
                                backbone=r["backbone"],
                                language=r["language"],
                                entity=r["entity"],
                                scope="relative_profile",
                                family="intrinsic_profile",
                            ),
                            kind="fingerprint",
                            metric="depth_profile",
                            payload=profiles[aid, source],
                        )
                    ],
                )
        for (language, entity, fold, seed), ids in sorted(
            cells.items(), key=lambda item: str(item[0])
        ):
            for aid, bid in combinations(sorted(ids), 2):
                if records[aid]["backbone"] == records[bid]["backbone"]:
                    continue
                for source in ("final", "delta"):
                    if (aid, source) not in profiles or (bid, source) not in profiles:
                        continue
                    uid = stable_id(["intrinsic_pair", aid, bid, source])
                    if store.done(uid):
                        continue
                    left, right = profiles[aid, source], profiles[bid, source]
                    meta = dict(
                        unit_id=uid,
                        artifact_a=aid,
                        artifact_b=bid,
                        language=language,
                        entity=entity,
                        language_a=language,
                        language_b=language,
                        entity_a=entity,
                        entity_b=entity,
                        fold_a=fold,
                        fold_b=fold,
                        seed_a=seed,
                        seed_b=seed,
                        source=source,
                        backbone="__cross_backbone__",
                        family="cross_backbone_intrinsic",
                        reference_kind="per_model_pretraining" if source == "delta" else None,
                        scope="encoder",
                        exactness="fingerprint_comparison",
                    )
                    trace = paper_trace_similarity(
                        {k: v["rms"] for k, v in left.items()},
                        {k: v["rms"] for k, v in right.items()},
                    )
                    trace_value = (
                        trace["trace_similarity"] if trace["trace_n_components"] >= 3 else math.nan
                    )
                    rows = [
                        record(
                            meta,
                            kind="geometry",
                            metric="trace_similarity",
                            value=trace_value,
                            payload=dict(n_components=trace["trace_n_components"], minimum=3),
                        )
                    ]
                    for component in sorted(set(left) & set(right)):
                        for feature, metric in (
                            ("curves", "spectral_curve_l1"),
                            ("quantiles", "normalized_quantile_w1"),
                            ("rank_fraction", "effective_rank_fraction_gap"),
                        ):
                            a, b = (
                                _resample(left[component], feature),
                                _resample(right[component], feature),
                            )
                            if a is None or b is None:
                                continue
                            distances = [
                                quantile_w1(x.tolist(), y.tolist())
                                if feature == "quantiles"
                                else float(np.mean(np.abs(x - y)))
                                for x, y in zip(a, b)
                            ]
                            cm = dict(meta, scope="component", component=component)
                            rows.append(
                                record(
                                    cm,
                                    kind="geometry",
                                    metric=metric,
                                    value=math.fsum(distances) / len(distances),
                                    payload=dict(
                                        depth_points=32,
                                        depth_weighting="equal_relative_grid",
                                        tensor_weighting="equal_tensor_dimensionless_features",
                                        quadrature="trapezoidal_quantile_grid"
                                        if feature == "quantiles"
                                        else None,
                                    ),
                                )
                            )
                            for index, value in enumerate(distances):
                                dm = dict(cm, scope="relative_depth", relative_depth=index / 31)
                                rows.append(record(dm, kind="geometry", metric=metric, value=value))
                    store.commit(uid, rows)
        store.compact(maximum_parts=8)
        from ..plan import atomic_json

        atomic_json(root / "reports/intrinsic_exclusions.json", exclusions)
        return [dict(item, area="intrinsic") for item in store.files(verify=True)]
