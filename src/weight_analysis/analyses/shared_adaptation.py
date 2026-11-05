"""The same descriptive group geometry for specialization across entities or languages."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch

from ..configuration import MetricOptions
from ..core import WeightReader
from ..core.blocks import iter_blocks, parameter_location
from ..core.catalog import CheckpointCatalog, ReferenceSpec, stable_id
from ..core.geometry_schema import DEFINITION, ParameterSelection
from ..core.reporting import RunReporter
from ..execution.comparative_units import ComparativeContext
from ..metrics.centroids import update_geometry
from ..results.comparative import ComparativeResult, event_table, issue_table, merge_comparative
from ..statistics.scores import bind_scores, validate_performance
from ..statistics.shared import SHARED_METRICS, finalize_shared


def analyze_shared_adaptation(
    *,
    checkpoints: CheckpointCatalog,
    axis: str,
    references: Mapping[str, ReferenceSpec],
    parameters: ParameterSelection | None = None,
    metrics: Sequence[str] = SHARED_METRICS,
    member_ids: Sequence[str] | None = None,
    leave_one_out: bool = True,
    performance: pd.DataFrame | None = None,
    score_alignment: Mapping[str, str] | None = None,
    options: MetricOptions | None = None,
    device: str | torch.device | None = None,
    reporter: RunReporter | None = None,
    on_error: str = "continue",
    cache_dir: str | Path | None = None,
    resume: bool = True,
    block_elements: int = 100_000,
    allow_unverified_origin: bool = False,
    _context: ComparativeContext | None = None,
) -> ComparativeResult:
    """Compare one member per entity/language inside explicit fixed experimental strata."""
    if axis not in ("entity", "language", "cell") or set(metrics) - set(SHARED_METRICS):
        raise ValueError("Unknown shared adaptation axis or metric")
    if not leave_one_out and any(m.startswith("loo_") for m in metrics):
        raise ValueError("LOO metrics require leave_one_out=True")
    reporter = reporter or RunReporter()
    parameters, options = parameters or ParameterSelection(), options or MetricOptions()
    tables = {"checkpoints": checkpoints.table}
    members = [e for e in checkpoints.entries if member_ids is None or e.artifact_id in member_ids]
    if member_ids is not None and set(member_ids) - set(checkpoints.by_id):
        raise ValueError("Unknown member_ids")
    scores = {}
    if performance is not None:
        validated = validate_performance(performance)
        bindings, _, _ = bind_scores(checkpoints, validated, score_alignment=score_alignment)
        scores = {row["artifact_id"]: row for row in bindings}
        tables["performance"] = validated
    ctx = reporter.attempt(
        "preflight",
        lambda: (
            _context
            or ComparativeContext(
                options=options,
                device=device,
                parameters=parameters,
                references=references,
                reporter=reporter,
                on_error=on_error,
                cache_dir=cache_dir,
                resume=resume,
                block_elements=block_elements,
                allow_unverified=allow_unverified_origin,
            )
        ),
        on_error=on_error,
    )
    if _context is not None and (
        _context.parameters != parameters
        or _context.options != options
        or _context.references != references
        or _context.reporter is not reporter
    ):
        raise ValueError("Shared context differs from requested scientific configuration")
    groups, member_rows, statistics, detailed = [], [], [], []
    if ctx:
        strata = {}
        fixed = "cell_scope" if axis == "cell" else ("language" if axis == "entity" else "entity")
        for entry in members:
            cp = entry.checkpoint
            key = (
                cp.backbone,
                "all_cells" if axis == "cell" else getattr(cp, fixed),
                cp.fold,
                cp.seed,
                entry.parameter_space_id,
                entry.evidence.initialization_id,
            )
            strata.setdefault(key, []).append(entry)
        for identity, group in sorted(strata.items(), key=lambda item: str(item[0])):
            group = sorted(group, key=lambda e: e.artifact_id)
            gid = stable_id(
                {"axis": axis, "stratum": identity, "members": [e.artifact_id for e in group]}
            )

            def process_group():
                if len(group) < 2 or len(
                    {
                        (e.checkpoint.language, e.checkpoint.entity)
                        if axis == "cell"
                        else getattr(e.checkpoint, axis)
                        for e in group
                    }
                ) != len(group):
                    raise ValueError(
                        "Select exactly one checkpoint per axis value, with at least two members"
                    )
                references_used = [ctx.reference(e) for e in group]
                if len({refid for _, refid in references_used}) != 1:
                    raise ValueError("Group updates require one shared reference origin")
                ref, rid = references_used[0]
                keys = ctx.keys(group[0])
                if any(set(ctx.keys(e)) != set(keys) for e in group[1:]):
                    raise ValueError("Group parameter spaces differ")
                if parameters.embeddings and any(
                    not e.evidence.tokenizer_id or e.evidence.tokenizer_id != ref.tokenizer_id
                    for e in group
                ):
                    raise ValueError("Embedding group lacks common tokenizer/vocabulary evidence")
                for key in keys:

                    def compute(uid):
                        with ExitStack() as stack:
                            readers = [
                                stack.enter_context(WeightReader(e.checkpoint.path)) for e in group
                            ]
                            base = stack.enter_context(WeightReader(ref.path))
                            for reader in readers:
                                ctx.aligned(reader, base)
                            location = parameter_location(readers[0], key)
                            streams = [
                                iter_blocks(r, key, block_elements) for r in readers + [base]
                            ]
                            totals = {
                                name: torch.zeros(
                                    len(group), device=ctx.device, dtype=torch.float64
                                )
                                for name in (
                                    "update_energy",
                                    "dot_mean",
                                    "centered_energy",
                                    "loo_mean_energy",
                                    "loo_dot",
                                    "loo_centered_energy",
                                )
                            }
                            mean_energy = torch.zeros((), device=ctx.device, dtype=torch.float64)
                            count = 0
                            for blocks in zip(*streams, strict=True):
                                if len({start for start, _ in blocks}) != 1:
                                    raise ValueError("Group coordinate blocks are misaligned")
                                weights = torch.stack(
                                    [
                                        block.to(ctx.device, dtype=torch.float64)
                                        for _, block in blocks[:-1]
                                    ]
                                )
                                initial = blocks[-1][1].to(ctx.device, dtype=torch.float64)
                                raw = update_geometry(weights, leave_one_out=False)
                                updates = weights - initial
                                mean = raw["mean"] - initial
                                totals["update_energy"] += updates.square().sum(dim=1)
                                totals["dot_mean"] += updates @ mean
                                totals["centered_energy"] += raw["centered_energy"]
                                mean_energy += mean.square().sum()
                                if leave_one_out:
                                    other = (updates.sum(dim=0) - updates) / (len(group) - 1)
                                    totals["loo_mean_energy"] += other.square().sum(dim=1)
                                    totals["loo_dot"] += (updates * other).sum(dim=1)
                                    totals["loo_centered_energy"] += (
                                        raw["centered_energy"]
                                        * (len(group) / (len(group) - 1)) ** 2
                                    )
                                count += initial.numel()
                            host = {name: value.cpu().tolist() for name, value in totals.items()}
                            mean_energy_value = float(mean_energy)
                            rows = []
                            for index, entry in enumerate(group):
                                rows.append(
                                    {
                                        **location,
                                        "backbone": identity[0],
                                        fixed: identity[1],
                                        axis: "__across_" + axis + "__",
                                        "fold": identity[2],
                                        "seed": identity[3],
                                        "comparison_type": "shared_" + axis,
                                        "group_id": gid,
                                        "member_id": entry.artifact_id,
                                        "membership_id": stable_id([gid, entry.artifact_id]),
                                        "axis": axis,
                                        "metric_names": ",".join(sorted(metrics)),
                                        "reference_id": rid,
                                        "execution_id": stable_id(ctx.definition),
                                        "mean_energy": mean_energy_value,
                                        "n_members": len(group),
                                        "n_parameters": count,
                                        **{name: values[index] for name, values in host.items()},
                                        "shared_statistic_id": stable_id([uid, entry.artifact_id]),
                                    }
                                )
                        return {
                            "shared_statistics": rows,
                            "geometry_detailed": finalize_shared(rows, metrics=metrics),
                        }

                    _, unit = ctx.unit(
                        {
                            "kind": "shared_adaptation",
                            "group": gid,
                            "tensor": key,
                            "reference": rid,
                            "metrics": sorted(metrics),
                            "leave_one_out": leave_one_out,
                        },
                        compute,
                        stage="shared_adaptation",
                        axis=axis,
                        tensor=key,
                        group_id=gid,
                    )
                    statistics.extend(unit.get("shared_statistics", []))
                    detailed.extend(unit.get("geometry_detailed", []))
                groups.append(
                    {
                        "group_id": gid,
                        "axis": axis,
                        "backbone": identity[0],
                        fixed: identity[1],
                        "fold": identity[2],
                        "seed": identity[3],
                        "n_members": len(group),
                    }
                )
                for entry in group:
                    row = {
                        "membership_id": stable_id([gid, entry.artifact_id]),
                        "group_id": gid,
                        "artifact_id": entry.artifact_id,
                        **entry.row(),
                    }
                    if entry.artifact_id in scores:
                        row.update(
                            {
                                name: scores[entry.artifact_id][name]
                                for name in (
                                    "score_id",
                                    "score_scope",
                                    "strict_f1_mean",
                                    "strict_f1_std",
                                )
                            }
                        )
                    member_rows.append(row)
                return True

            reporter.attempt(gid, process_group, on_error=on_error, stage="group", axis=axis)
        tables["references"] = pd.DataFrame(ctx.reference_rows.values())
    reporter.summary()
    tables.update(
        groups=pd.DataFrame(groups),
        group_members=pd.DataFrame(member_rows),
        shared_statistics=pd.DataFrame(statistics),
        geometry_detailed=pd.DataFrame(detailed),
        issues=issue_table(reporter),
        events=event_table(reporter),
    )
    for issue in checkpoints.issues:
        tables["issues"] = pd.concat(
            [tables["issues"], pd.DataFrame([{**issue, "issue_id": stable_id(issue)}])],
            ignore_index=True,
        )
    return merge_comparative(
        [
            ComparativeResult(
                tables,
                {
                    "definition": DEFINITION,
                    "axis": axis,
                    "members": [e.artifact_id for e in members],
                    "metrics": list(metrics),
                    "parameters": asdict(parameters),
                    "computation": ctx.definition if ctx else None,
                    "allow_unverified_origin": allow_unverified_origin,
                    "leave_one_out": leave_one_out,
                    "complete": tables["issues"].empty,
                },
            )
        ]
    )
