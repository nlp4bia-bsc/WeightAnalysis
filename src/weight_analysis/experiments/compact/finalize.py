"""Deterministic, checkpoint-free consolidation of sufficient statistics and coverage."""

from __future__ import annotations

import fcntl
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pyarrow.dataset as ds

from ...core.catalog import stable_id
from ...provenance import file_sha256
from ..plan import atomic_json
from ..source import code_digest
from .derive import INVARIANT, endpoint_metadata, family, unit_identity
from .plan import load_plan
from .primitives import DRIFT_METRICS, PAIR_METRICS, moment_dict, moments, shared_values
from .storage import SCHEMA, PackedStore, record


def _source_status(source):
    if source["f1_status"] != "ready":
        return source["f1_status"]
    try:
        return (
            "ready"
            if file_sha256(source["source_path"]) == source["source_sha256"]
            else "blocked_source_changed"
        )
    except OSError:
        return "blocked_source_unavailable"


def expected_units(plan, task):
    units = []
    for tensor in task["tensors"]:
        key = tensor["tensor"]
        units.append(unit_identity(plan, task, key, "exclusions"))
        if not tensor["members"]:
            continue
        units.append(unit_identity(plan, task, key, "statistics"))
        if len(tensor["shape"]) == 2 and plan["config"]["spectral_mode"] != "skip":
            units.append(unit_identity(plan, task, key, "spectrum", None, "base"))
            units.extend(
                unit_identity(plan, task, key, "spectrum", aid, source)
                for aid in tensor["members"]
                for source in ("final", "delta")
            )
    return units


def scopes(tensor):
    layer, component, embedding = tensor["layer"], tensor["component"], tensor["embedding_type"]
    result = [("encoder", None, None, None), ("component", None, component, embedding)]
    if layer is not None:
        result += [("layer", layer, None, None), ("layer_component", layer, component, None)]
    if component == "embeddings":
        result.append(("embeddings", None, component, embedding))
    return result


def _scope_metadata(scope):
    return dict(
        scope=scope[0], layer=scope[1], component=scope[2], embedding_type=scope[3], tensor=None
    )


def _merge_domain(plan, tasks, statistics):
    records = {r["artifact_id"]: r for r in plan["checkpoints"]}
    vectors, shared, expected = {}, {}, defaultdict(int)
    shared_info = {}
    domain = tasks[0]["domain_id"]
    backbone = tasks[0]["backbone"]
    unit = stable_id(["aggregate", domain])
    total_layers = tasks[0]["total_layers"]
    for task in tasks:
        for tensor in task["tensors"]:
            n = math.prod(tensor["shape"])
            for scope in scopes(tensor):
                expected[scope] += n
    for row in sorted(statistics, key=lambda r: (r["tensor"], r["unit_id"])):
        payload = json.loads(row["payload"])
        state, members = payload["state"], payload["members"]
        for scope in scopes(row):
            for i, aid in enumerate(members, 1):
                key = (scope, aid, None, "final", row["execution_id"])
                value = moments(state, "final", i, 0)
                vectors[key] = vectors[key].merge(value) if key in vectors else value
            for (i, aid), (j, bid) in combinations(enumerate(members, 1), 2):
                for source in ("final", "delta"):
                    key = (scope, aid, bid, source, row["execution_id"])
                    value = moments(state, source, i, j)
                    vectors[key] = vectors[key].merge(value) if key in vectors else value
            for group in state["groups"]:
                shared_info[group["group_id"]] = group
                for index, aid in enumerate(group["members"]):
                    key = (scope, group["group_id"], aid, row["execution_id"])
                    old = shared.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
                    for j in range(4):
                        old[j] += state["shared"][group["group_id"]][index][j]
                    old[4] += state["n"]
    rows = []
    for (scope, aid, bid, source, execution), value in sorted(
        vectors.items(), key=lambda item: str(item[0])
    ):
        a, b = records[aid], records[bid] if bid else None
        meta = dict(
            unit_id=unit,
            **_scope_metadata(scope),
            **endpoint_metadata(a, b),
            source=source if bid else "delta",
            execution_id=execution,
            family=family(a, b) if bid else "checkpoint_vs_pretraining",
            exactness="exact_float64",
            relative_depth=(scope[1] + 0.5) / total_layers if scope[1] is not None else None,
        )
        coverage = "ok" if value.n == expected[scope] else "partial_coverage"
        rows.append(
            record(
                meta,
                kind="sufficient",
                metric="vector_moments",
                payload=moment_dict(value),
                n=value.n,
                status=coverage,
            )
        )
        rows.append(
            record(
                meta,
                kind="coverage",
                metric="coordinate_fraction",
                value=value.n / expected[scope],
                n=value.n,
                payload=dict(expected=expected[scope], observed=value.n),
                status=coverage,
            )
        )
        names = PAIR_METRICS if bid else DRIFT_METRICS
        for name in names:
            if source == "delta" and bid and name in INVARIANT:
                continue
            number = value.value(name)
            status = coverage if math.isfinite(number) else "undefined"
            rows.append(
                record(meta, kind="geometry", metric=name, value=number, n=value.n, status=status)
            )
    emitted = set()
    for (scope, gid, aid, execution), values in sorted(
        shared.items(), key=lambda item: str(item[0])
    ):
        group = shared_info[gid]
        n = int(values[4])
        meta = dict(
            unit_id=unit,
            **_scope_metadata(scope),
            **endpoint_metadata(records[aid]),
            source="delta",
            family=group["family"],
            group_id=gid,
            execution_id=execution,
            exactness="exact_float64",
            relative_depth=(scope[1] + 0.5) / total_layers if scope[1] is not None else None,
        )
        coverage = "ok" if n == expected[scope] else "partial_coverage"
        rows.append(
            record(
                meta,
                kind="sufficient",
                metric="shared_moments",
                payload=dict(values=values[:4], size=group["size"], members=group["members"]),
                n=n,
                status=coverage,
            )
        )
        rows.append(
            record(
                meta,
                kind="coverage",
                metric="coordinate_fraction",
                value=n / expected[scope],
                n=n,
                payload=dict(expected=expected[scope], observed=n),
                status=coverage,
            )
        )
        derived = shared_values(values[:4], group["size"])
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
                    meta,
                    kind="geometry",
                    metric=metric,
                    value=number,
                    n=n,
                    status=coverage if math.isfinite(number) else "undefined",
                )
            )
        token = (scope, gid, execution)
        if token not in emitted:
            group_meta = {k: v for k, v in meta.items() if k not in endpoint_metadata(records[aid])}
            group_meta.update(backbone=backbone, reference_kind="pretraining", source="delta")
            group_meta["reference_id"] = records[aid]["reference_content_id"]
            languages = {records[member]["language"] for member in group["members"]}
            entities = {records[member]["entity"] for member in group["members"]}
            group_meta.update(
                language=next(iter(languages)) if len(languages) == 1 else "__cross_language__",
                entity=next(iter(entities)) if len(entities) == 1 else "__cross_entity__",
            )
            rows.append(
                record(
                    group_meta,
                    kind="geometry",
                    metric="mean_update_energy",
                    value=values[1],
                    n=n,
                    status=coverage,
                )
            )
            energies = [
                entry[3] for key, entry in shared.items() if (key[0], key[1], key[3]) == token
            ]
            rows.append(
                record(
                    group_meta,
                    kind="geometry",
                    metric="dispersion_energy",
                    value=math.fsum(energies) / len(energies),
                    n=n,
                    status=coverage,
                )
            )
            emitted.add(token)
    return rows


def finalize_run(run_dir, *, intrinsic=True):
    root = Path(run_dir).resolve()
    plan = load_plan(root)
    lock = (root / "state/finalize.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seal_lock = (root / "state/seal.lock").open("a")
    config = plan["config"]
    manifests, states, retry = [], [], []
    try:
        fcntl.flock(seal_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        published = root / "manifest/datasets.json"
        if published.exists():
            previous = json.loads(published.read_text())
            if previous.get("sealed_atomic"):
                for item in previous["files"]:
                    if file_sha256(root / item["path"]) != item["sha256"]:
                        raise ValueError("Sealed dataset integrity failure: " + item["path"])
                if intrinsic and not previous.get("intrinsic_complete"):
                    from .intrinsic import derive_intrinsic

                    previous["files"] += derive_intrinsic(
                        root, plan, previous["files"], previous["generation"]
                    )
                    previous["intrinsic_complete"] = True
                previous["f1_status"] = _source_status(plan["source"])
                atomic_json(root / "manifest/datasets.json", previous)
                atomic_json(root / "reports/finalization.json", previous)
                from .query import write_views

                write_views(root)
                from .summary import write_summary

                write_summary(root, previous)
                print("[FINALIZE] Sealed run already complete; no weights recomputed", flush=True)
                return previous
        for task in plan["tasks"]:
            tag = "{:05d}".format(task["index"])
            state = dict(
                index=task["index"],
                expected=len(expected_units(plan, task)),
                completed=0,
                state="not_started",
            )
            if (root / "state" / (tag + ".sqlite")).exists():
                task_lock = (root / "state" / (tag + ".lock")).open("a")
                try:
                    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with PackedStore(
                        root,
                        tag,
                        policy=config["storage_policy"],
                        target_bytes=config["target_part_bytes"],
                    ) as store:
                        store.flush()
                        files = store.files(verify=True)
                        state["completed"] = sum(
                            store.done(uid) for uid in expected_units(plan, task)
                        )
                        state["errors"] = [
                            json.loads(row[0])
                            for row in store.db.execute(
                                "SELECT body FROM progress WHERE id LIKE 'error:%' ORDER BY id"
                            )
                        ]
                        worker_status = root / "state" / (tag + ".json")
                        if worker_status.exists():
                            state["worker"] = json.loads(worker_status.read_text())
                        state["state"] = (
                            "complete" if state["completed"] == state["expected"] else "partial"
                        )
                        manifests += [
                            dict(item, area="atomic", task_index=task["index"]) for item in files
                        ]
                except BlockingIOError:
                    state.update(state="running", error="Task writer still holds its lock")
                except Exception as exc:
                    state.update(state="unreadable", error=str(exc))
                finally:
                    task_lock.close()
            if state["state"] != "complete":
                retry.append(task["index"])
            states.append(state)
        generation = stable_id(manifests)
        with PackedStore(
            root,
            "aggregate",
            area="aggregated",
            policy=config["storage_policy"],
            target_bytes=config["target_part_bytes"],
        ) as output:
            output.generation(generation)
            tasks_by_domain = defaultdict(list)
            for task in plan["tasks"]:
                tasks_by_domain[task["domain_id"]].append(task)
            for domain, tasks in sorted(tasks_by_domain.items()):
                uid = stable_id(["aggregate", domain])
                if output.done(uid):
                    continue
                task_indices = {t["index"] for t in tasks}
                files = [
                    str(root / item["path"])
                    for item in manifests
                    if item["task_index"] in task_indices
                ]
                if not files:
                    continue
                table = ds.dataset(files, format="parquet", schema=SCHEMA).to_table(
                    filter=ds.field("kind") == "sufficient"
                )
                output.commit(uid, _merge_domain(plan, tasks, table.to_pylist()))
            output.compact(maximum_parts=8)
            manifests += [dict(item, area="aggregated") for item in output.files(verify=True)]
            if intrinsic:
                from .intrinsic import derive_intrinsic

                manifests += derive_intrinsic(root, plan, manifests, generation)
            f1_status = _source_status(plan["source"])
            original_atomic = [item for item in manifests if item["area"] == "atomic"]
            if not retry:
                from .consolidate import consolidate_studies

                studies = consolidate_studies(root, plan, original_atomic, generation)
                manifests = studies + [item for item in manifests if item["area"] != "atomic"]
            report = dict(
                run_id=plan["run_id"],
                schema_version=plan["schema_version"],
                code_digest=code_digest(),
                state="complete" if not retry else "partial",
                tasks=states,
                retry_array_indices=retry,
                files=manifests,
                generation=generation,
                f1_status=f1_status,
                interpretation=plan["interpretation"],
                sealed_atomic=not retry,
                intrinsic_complete=intrinsic,
            )
            atomic_json(root / "manifest/datasets.json", report)
            atomic_json(root / "reports/finalization.json", report)
            output.reclaim_obsolete()
            for tag, area in (("intrinsic", "intrinsic"), ("studies", "atomic")):
                if (root / "state" / (tag + ".sqlite")).exists():
                    with PackedStore(
                        root, tag, area=area, policy=config["storage_policy"]
                    ) as owned:
                        owned.reclaim_obsolete()
            if not retry:
                from .consolidate import reclaim_task_parts

                reclaim_task_parts(root, original_atomic)
        from .query import write_views

        write_views(root)
        from .summary import write_summary

        write_summary(root, report)
        if report["f1_status"] == "ready":
            from .associations import derive_associations

            derive_associations(root)
        print(
            "[FINALIZE]",
            report["state"],
            "tasks",
            len(states),
            "retry",
            retry,
            "Parquet files",
            len(manifests),
            flush=True,
        )
        return report
    finally:
        lock.close()
        seal_lock.close()
