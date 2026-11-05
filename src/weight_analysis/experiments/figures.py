"""Figures from exported Parquet only; no weight access or implicit scientific means."""

from __future__ import annotations

import pandas as pd

from ..results import read_result
from ..visualization import ColorScale, plot_f1_gap, plot_geometry_profiles, plot_pair_heatmap
from .exports import read_joined
from .plan import atomic_json, load_plan


def make_figures(run_dir):
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root = Path(run_dir)
    plan = load_plan(root)
    frame = read_joined(root)
    if frame.empty:
        raise ValueError("No committed geometry–F1 results; finalize first")
    ranks = read_result(root / "derived" / "f1_model_ranking.parquet").data
    selections = read_result(root / "derived" / "f1_best_worst_pairs.parquet").data
    output = root / "figures"
    output.mkdir(exist_ok=True)
    manifest, issues = [], []

    def save(name, operation, criteria):
        try:
            ax = operation()
            figure = ax.figure if hasattr(ax, "figure") else ax
            if hasattr(ax, "set_title"):
                context = " / ".join(
                    str(criteria[key])
                    for key in ("backbone", "language", "entity")
                    if key in criteria
                )
                ax.set_title(context + "\n" + ax.get_title() + " — F1 agregado; descriptivo")
            figure.set_size_inches(12, 10)
            figure.tight_layout()
            figure.savefig(output / (name + ".png"), dpi=160)
            plt.close(figure)
            manifest.append(dict(path=name + ".png", selection=criteria))
        except Exception as exc:
            issues.append(dict(path=name, reason=str(exc)))
            plt.close("all")

    # A common pretraining reference KIND is appropriate for intrinsic comparisons;
    # endpoint-specific reference identities remain intact in the underlying Parquet.
    cross = frame[
        (frame.comparison_type == "cross_backbone_intrinsic") & (frame.scope == "encoder")
    ]
    for metric in ("trace_distance_delta", "curve_l1_delta", "quantile_w1_final"):
        values = cross.loc[cross.metric == metric, "value"].dropna().astype(float)
        if values.empty:
            continue
        maximum = max(float(values.max()), 1e-12)
        scale = ColorScale(metric, 0, maximum)
        for (language, entity), group in cross[cross.metric == metric].groupby(
            ["language", "entity"], dropna=False
        ):
            ranking = ranks[(ranks.language == language) & (ranks.entity == entity)].sort_values(
                "selection_order"
            )
            labels = {
                row.model_id: f"{row.backbone}\nF1={row.strict_f1_mean:.3f} ± {row.strict_f1_std:.3f}"
                for row in ranking.itertuples()
            }
            plot = group.rename(
                columns={"artifact_id_a": "artifact_a", "artifact_id_b": "artifact_b"}
            ).copy()
            plot["artifact_a"] = plot.model_id_a.map(labels)
            plot["artifact_b"] = plot.model_id_b.map(labels)
            plot = plot.dropna(subset=["artifact_a", "artifact_b", "value"])
            plot["reference_id"] = "per_model_pretraining" if metric.endswith("delta") else None
            plot["comparison_id"] = plot.subject_id
            present = set(plot.artifact_a) | set(plot.artifact_b)
            order = [labels[mid] for mid in ranking.model_id if labels[mid] in present]
            criteria = dict(
                language=str(language),
                entity=str(entity),
                metric=metric,
                scope="encoder",
                ordering="F1 descending, stable model ID tie break",
                scale=[0, maximum],
            )
            if plot.empty:
                continue
            save(
                f"all_models-{language}-{entity}-{metric}",
                lambda p=plot, o=order: plot_pair_heatmap(p, metric=metric, order=o, scale=scale),
                criteria,
            )
            best = plot[plot.performance_pair_id.isin(selections.performance_pair_id)]
            if not best.empty:
                identities = set(best.artifact_a) | set(best.artifact_b)
                best_order = [label for label in order if label in identities]
                save(
                    f"best_worst-{language}-{entity}-{metric}",
                    lambda p=best, o=best_order: plot_pair_heatmap(
                        p, metric=metric, order=o, scale=scale
                    ),
                    {**criteria, "selection": "best_worst_from_full_F1_ranking"},
                )
            save(
                f"F1_gap-{language}-{entity}-{metric}",
                lambda p=plot: plot_f1_gap(p, metric=metric),
                criteria,
            )
    direct = frame[
        (
            frame.comparison_type.isin(
                [
                    "same_entity_cross_language",
                    "same_language_cross_entity",
                    "cross_language_cross_entity",
                ]
            )
        )
        & (frame.scope == "encoder")
        & (frame.metric == "l2_distance")
    ]
    for backbone, group in direct.groupby("backbone", dropna=False):
        plot = group.rename(
            columns={"artifact_id_a": "artifact_a", "artifact_id_b": "artifact_b"}
        ).copy()
        for side in ("a", "b"):
            plot["artifact_" + side] = (
                plot["language_" + side].astype(str)
                + "/"
                + plot["entity_" + side].astype(str)
                + " F1="
                + plot["strict_f1_mean_" + side].map(
                    lambda x: "unknown" if pd.isna(x) else f"{x:.3f}"
                )
            )
        save(
            f"same_backbone-{backbone}-all_cells",
            lambda p=plot: plot_pair_heatmap(
                p.drop(columns=["language", "entity"], errors="ignore"), metric="l2_distance"
            ),
            dict(backbone=str(backbone), metric="l2_distance", cross_task="descriptive_only"),
        )
    references = frame[
        frame.comparison_type.eq("checkpoint_vs_pretraining") & frame.metric.eq("relative_l2")
    ]
    for (backbone, language), group in references.groupby(["backbone", "language"], dropna=False):
        for scope, x in (("layer", "layer"), ("component", "component")):
            data = group[group.scope.eq(scope)].copy()
            if data.empty:
                continue
            data["subject_id"] = (
                data.entity.astype(str)
                + " / F1="
                + data.strict_f1_mean.map(
                    lambda value: "unknown" if pd.isna(value) else f"{value:.3f}"
                )
            )
            data["reference_id"] = "pretraining"
            save(
                f"pretraining-{backbone}-{language}-{scope}",
                lambda p=data, axis=x: plot_geometry_profiles(
                    p,
                    metric="relative_l2",
                    x=axis,
                    series=("subject_id", "source", "reference_id", "embedding_type"),
                ),
                dict(
                    backbone=str(backbone),
                    language=str(language),
                    metric="relative_l2",
                    scope=scope,
                    reference="pretraining",
                ),
            )
    atomic_json(
        output / "figures.json",
        dict(
            run_id=plan["run_id"],
            figures=manifest,
            issues=issues,
            policy="only committed precomputed values; no implicit aggregation; shared scales per intrinsic metric",
        ),
    )
    print(f"[FIGURES] saved={len(manifest)} issues={len(issues)}", flush=True)
    return manifest, issues
