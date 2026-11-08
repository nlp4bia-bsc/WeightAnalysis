"""Tensor-major GPU execution, durable block checkpoints and isolated spectral failures."""

from __future__ import annotations

import fcntl
import json
import math
import signal
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from weight_analysis_scheduler.resources import StoragePressure, check_storage, is_storage_error

from ...core import WeightReader
from ...core.catalog import stable_id
from ...core.reporting import RunReporter
from ...devices import resolve_device
from ...metrics.spectral import SPECTRAL_METRICS, spectral_metrics
from ..plan import atomic_json
from ..source import runtime_versions
from .derive import endpoint_metadata, geometry_rows, unit_identity
from .plan import load_plan, verify_inputs
from .primitives import TensorStatistics
from .storage import PackedStore, record


class BudgetStop(BaseException):
    pass


def _finite(value):
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metadata(task, tensor, uid, execution):
    return dict(
        unit_id=uid,
        backbone=task["backbone"],
        scope="tensor",
        tensor=tensor["tensor"],
        layer=tensor["layer"],
        component=tensor["component"],
        embedding_type=tensor["embedding_type"],
        exactness="exact_float64",
        relative_depth=(tensor["layer"] + 0.5) / task["total_layers"]
        if tensor["layer"] is not None
        else None,
        execution_id=execution,
    )


def _statistics_state(stats, execution):
    return dict(stats.state(), execution_id=execution)


def run_task(run_dir, index, *, device="gpu", budget_seconds=27000):
    root = Path(run_dir).resolve()
    plan = load_plan(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("Task index outside compact manifest")
    task, config = plan["tasks"][index], plan["config"]
    published = root / "manifest/datasets.json"
    if published.exists():
        report = json.loads(published.read_text())
        if report.get("sealed_atomic") and report["run_id"] == plan["run_id"]:
            print("[CACHED] Sealed run; task already complete", index, flush=True)
            return dict(run_id=plan["run_id"], index=index, state="complete", sealed=True)
    tag = "{:05d}".format(index)
    reporter = RunReporter(log_path=root / "logs" / ("worker-" + tag + ".jsonl"))
    status = dict(
        run_id=plan["run_id"],
        index=index,
        task_id=task["task_id"],
        state="running",
        completed=0,
        failed=0,
        excluded=0,
        runtime_versions=runtime_versions(),
    )
    stop, deadline = False, time.monotonic() + budget_seconds

    def interrupt(*_):
        nonlocal stop
        stop = True

    def boundary():
        if stop or time.monotonic() >= deadline:
            raise BudgetStop()

    previous = signal.signal(signal.SIGUSR1, interrupt)
    lock = (root / "state" / (tag + ".lock")).open("a")
    seal_lock = (root / "state/seal.lock").open("a")
    acquired = False
    store = None
    try:
        fcntl.flock(seal_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        if published.exists() and json.loads(published.read_text()).get("sealed_atomic"):
            status.update(state="complete", sealed=True)
            return status
        target = resolve_device(device)
        hardware = torch.cuda.get_device_name(target) if target.type == "cuda" else "cpu"
        execution = stable_id([str(target), hardware, runtime_versions(), config["block_elements"]])
        status["execution_id"] = execution
        status.update(
            device=str(target),
            hardware=hardware,
            cuda_runtime=torch.version.cuda,
            block_elements=config["block_elements"],
        )
        check_storage(root, config["storage_policy"])
        selected = [r for r in plan["checkpoints"] if r["artifact_id"] in task["members"]]
        selected.sort(key=lambda r: r["artifact_id"])
        verify_inputs(selected)
        store = PackedStore(
            root, tag, policy=config["storage_policy"], target_bytes=config["target_part_bytes"]
        )
        store.files(verify=True)
        reporter.emit(
            "started",
            tag,
            "Compact tensor-major worker",
            device=str(target),
            hardware=hardware,
            tensors=len(task["tensors"]),
            checkpoints=len(selected),
        )
        with ExitStack() as stack:
            base = stack.enter_context(WeightReader(selected[0]["reference"]["path"]))
            readers = {
                r["artifact_id"]: stack.enter_context(WeightReader(r["path"])) for r in selected
            }
            for tensor in task["tensors"]:
                boundary()
                verify_inputs(selected)
                key = tensor["tensor"]
                members = [r for r in selected if r["artifact_id"] in tensor["members"]]
                uid = unit_identity(plan, task, key, "statistics")
                meta = _metadata(task, tensor, uid, execution)
                eid = unit_identity(plan, task, key, "exclusions")
                planned = [eid] + ([uid] if members else [])
                if members and len(tensor["shape"]) == 2 and config["spectral_mode"] != "skip":
                    planned += [unit_identity(plan, task, key, "spectrum", None, "base")]
                    planned += [
                        unit_identity(plan, task, key, "spectrum", entry["artifact_id"], source)
                        for entry in members
                        for source in ("final", "delta")
                    ]
                if all(store.done(item) for item in planned):
                    status["completed"] += len(planned)
                    status["excluded"] += len(tensor["exclusions"])
                    reporter.emit("cached", key, "Tensor units already committed; no weight reads")
                    continue
                if not store.done(eid):
                    rows = [
                        record(
                            dict(
                                meta,
                                unit_id=eid,
                                artifact_a=e["artifact_id"],
                                family="compatibility_exclusions",
                            ),
                            kind="exclusion",
                            metric="coordinate_compatibility",
                            status="excluded",
                            payload=e,
                        )
                        for e in tensor["exclusions"]
                    ]
                    for group in task["groups"]:
                        missing = sorted(set(group["members"]) - set(tensor["members"]))
                        if group["complete"] and missing:
                            rows.append(
                                record(
                                    dict(
                                        meta,
                                        unit_id=eid,
                                        group_id=group["group_id"],
                                        family="compatibility_exclusions",
                                    ),
                                    kind="exclusion",
                                    metric="shared_coordinate_compatibility",
                                    status="excluded",
                                    payload=dict(
                                        reason="complete_group_coordinates_required",
                                        missing=missing,
                                    ),
                                )
                            )
                    store.commit(eid, rows)
                status["excluded"] += len(tensor["exclusions"])
                if not members:
                    reporter.emit("skipped", key, "No compatible coordinates; exclusions persisted")
                    continue
                matrices, flat = [], []
                block, matrix, initial_gpu, stats = None, None, None, None
                try:
                    matrices = [base.get(key)] + [
                        readers[r["artifact_id"]].get(key) for r in members
                    ]
                    resident = sum(m.numel() * 8 for m in matrices) <= config["resident_bytes"]
                    if resident:
                        matrices = [m.to(target, dtype=torch.float64) for m in matrices]
                    if not store.done(uid):
                        reporter.emit(
                            "started",
                            uid,
                            "Shared vector reductions",
                            tensor=key,
                            models=len(members),
                            resident=resident,
                        )
                        groups = []
                        ids = [r["artifact_id"] for r in members]
                        for group in task["groups"]:
                            if group["complete"] and set(group["members"]) <= set(ids):
                                groups.append(
                                    dict(
                                        group,
                                        indices=[ids.index(aid) + 1 for aid in group["members"]],
                                    )
                                )
                        progress = store.progress(uid)
                        if progress and progress.get("execution_id") != execution:
                            raise ValueError(
                                "Partial tensor belongs to another execution environment; resume with the original device/software"
                            )
                        stats = TensorStatistics(
                            len(matrices), groups, device=target, state=progress
                        )
                        try:
                            flat = [m.reshape(-1) for m in matrices]
                            for start in range(stats.n, flat[0].numel(), config["block_elements"]):
                                boundary()
                                end = start + config["block_elements"]
                                block = torch.stack(
                                    [m[start:end].to(target, dtype=torch.float64) for m in flat]
                                )
                                stats.add(block)
                                if (start // config["block_elements"]) % 32 == 0:
                                    check_storage(root, config["storage_policy"])
                                    store.progress(uid, _statistics_state(stats, execution))
                                    reporter.emit(
                                        "progress",
                                        uid,
                                        "Block statistics checkpointed",
                                        coordinates=stats.n,
                                        total=flat[0].numel(),
                                    )
                            state = _statistics_state(stats, execution)
                            store.progress(
                                "equivalence:" + uid, dict(maximum=state["data"]["maximum"])
                            )
                            rows = [
                                record(
                                    dict(meta, family="sufficient_statistics"),
                                    kind="sufficient",
                                    metric="tensor_statistics",
                                    n=stats.n,
                                    payload=dict(state=state, members=ids),
                                )
                            ]
                            rows += geometry_rows(state, members, meta)
                            store.commit(uid, rows)
                            store.clear_progress("error:" + uid)
                            status["completed"] += 1
                        except BaseException:
                            store.progress(uid, _statistics_state(stats, execution))
                            raise
                    else:
                        status["completed"] += 1
                    if len(tensor["shape"]) == 2 and config["spectral_mode"] != "skip":
                        initial_gpu = matrices[0].to(target, dtype=torch.float64)
                        computed = {}
                        equality = (store.progress("equivalence:" + uid) or {}).get("maximum")

                        def restored_fingerprint(spectral_id):
                            payload = json.loads(store.read_unit(spectral_id)[0]["payload"])
                            canonical = payload.get("alias_unit_id")
                            return (
                                restored_fingerprint(canonical)
                                if canonical
                                else (spectral_id, payload)
                            )

                        subjects = [(None, 0, "base")]
                        subjects += [
                            (entry, i, source)
                            for i, entry in enumerate(members, 1)
                            for source in ("final", "delta")
                        ]
                        for entry, i, source in subjects:
                            sid = unit_identity(
                                plan,
                                task,
                                key,
                                "spectrum",
                                entry["artifact_id"] if entry else None,
                                source,
                            )
                            if store.done(sid):
                                computed[i, source] = restored_fingerprint(sid)
                                status["completed"] += 1
                                continue
                            boundary()
                            try:
                                equivalent = next(
                                    (
                                        result
                                        for (j, origin), result in computed.items()
                                        if equality is not None
                                        and equality[i][j] == 0
                                        and (
                                            origin == source
                                            or source == "final"
                                            and origin == "base"
                                        )
                                    ),
                                    None,
                                )
                                if equivalent:
                                    canonical, values = equivalent
                                    reporter.emit(
                                        "cached",
                                        sid,
                                        "Exact matrix fingerprint alias",
                                        tensor=key,
                                        source=source,
                                        canonical=canonical,
                                    )
                                else:
                                    reporter.emit(
                                        "started",
                                        sid,
                                        "Spectral fingerprint",
                                        tensor=key,
                                        checkpoint=entry["language"] + "/" + entry["entity"]
                                        if entry
                                        else "pretraining",
                                        source=source,
                                        mode=config["spectral_mode"],
                                    )
                                    matrix = matrices[i].to(target, dtype=torch.float64)
                                    if source == "delta":
                                        matrix = matrix - initial_gpu
                                    values = spectral_metrics(
                                        matrix,
                                        metrics=SPECTRAL_METRICS,
                                        device=str(target),
                                        mode=config["spectral_mode"],
                                        topk=config["topk"],
                                        max_exact_elements=config["max_svd_elements"],
                                        return_spectrum=True,
                                        curve_points=config["curve_points"],
                                        quantile_points=config["quantile_points"],
                                    )
                                    canonical = sid
                                    values.update(
                                        matrix_shape=tensor["shape"], n_parameters=matrix.numel()
                                    )
                                    values.update(
                                        device=str(target),
                                        svd_algorithm="torch_svdvals"
                                        if values["status"] == "exact"
                                        else "cuda_randomized_range_seed0_power4"
                                        if target.type == "cuda"
                                        else "scipy_svds_fixed_v0",
                                    )
                                spectral_meta = dict(
                                    meta,
                                    unit_id=sid,
                                    family="spectral_fingerprint",
                                    source=source,
                                    exactness=values["status"],
                                )
                                if entry:
                                    spectral_meta.update(endpoint_metadata(entry))
                                else:
                                    spectral_meta.update(
                                        reference_id=selected[0]["reference_content_id"],
                                        reference_kind="pretraining",
                                    )
                                payload = (
                                    dict(
                                        alias_unit_id=canonical,
                                        equivalence="zero_maximum_coordinate_difference",
                                    )
                                    if equivalent
                                    else _finite(values)
                                )
                                store.commit(
                                    sid,
                                    [
                                        record(
                                            spectral_meta,
                                            kind="fingerprint",
                                            metric="spectral_fingerprint",
                                            payload=payload,
                                            n=math.prod(tensor["shape"]),
                                        )
                                    ],
                                )
                                computed[i, source] = (canonical, _finite(values))
                                store.clear_progress("error:" + sid)
                                status["completed"] += 1
                            except (OSError, StoragePressure):
                                raise
                            except Exception as exc:
                                status["failed"] += 1
                                store.progress(
                                    "error:" + sid,
                                    dict(
                                        unit_id=sid,
                                        tensor=key,
                                        source=source,
                                        artifact_id=entry["artifact_id"] if entry else None,
                                        error_type=type(exc).__name__,
                                        error=str(exc),
                                    ),
                                )
                                reporter.emit(
                                    "failed",
                                    sid,
                                    "Spectral unit failed; continuing other units",
                                    exception=exc,
                                )
                except (BudgetStop, StoragePressure):
                    raise
                except Exception as exc:
                    if is_storage_error(exc):
                        raise StoragePressure(str(exc)) from exc
                    status["failed"] += 1
                    store.progress(
                        "error:" + uid,
                        dict(
                            unit_id=uid, tensor=key, error_type=type(exc).__name__, error=str(exc)
                        ),
                    )
                    reporter.emit(
                        "failed",
                        uid,
                        "Tensor failed; continuing independent tensors",
                        exception=exc,
                    )
                    continue
                finally:
                    del matrices, flat, block, matrix, initial_gpu, stats
                store.flush()
                if len(store.files()) > 8:
                    store.compact(maximum_parts=4)
                reporter.events.clear()
                reporter.emit(
                    "progress",
                    key,
                    "Tensor committed",
                    completed=status["completed"],
                    failed=status["failed"],
                    resident=resident,
                    files=len(store.files()),
                )
        store.compact(maximum_parts=4)
        status["state"] = "complete" if not status["failed"] else "partial"
    except (BudgetStop, StoragePressure, BlockingIOError) as exc:
        status.update(state="interrupted_resumable", reason=type(exc).__name__)
        reporter.emit(
            "warning", tag, "Budget/storage stop; completed units retained", reason=str(exc)
        )
    except Exception as exc:
        status.update(
            state="interrupted_resumable" if is_storage_error(exc) else "failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        reporter.emit("failed", tag, str(exc), exception=exc)
    finally:
        signal.signal(signal.SIGUSR1, previous)
        if acquired:
            if store:
                try:
                    store.flush()
                    status["files"] = store.files()
                except Exception as exc:
                    status.update(
                        state="interrupted_resumable"
                        if is_storage_error(exc) or isinstance(exc, OSError)
                        else "failed",
                        persistence_error=str(exc),
                    )
                finally:
                    store.close()
            try:
                atomic_json(root / "state" / (tag + ".json"), status)
            except OSError as exc:
                status.update(state="interrupted_resumable", persistence_error=str(exc))
                print(
                    "[STATE WRITE FAILED] Committed journal remains authoritative:", exc, flush=True
                )
        lock.close()
        seal_lock.close()
    return status
