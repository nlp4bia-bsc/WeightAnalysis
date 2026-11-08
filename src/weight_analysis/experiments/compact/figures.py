"""Figures read selected persisted values; no weights or implicit scientific pooling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from weight_analysis_scheduler.resources import check_storage

from ..plan import atomic_json
from .query import joined_geometry, read_geometry


def make_figures(
    run_dir, *, metrics=("l2_distance", "cosine", "trace_similarity"), max_figures=2400
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root = Path(run_dir)
    manifest = json.loads((root / "manifest/datasets.json").read_text())
    plan = json.loads((root / "manifest/plan.json").read_text())
    best_pairs = set()
    if manifest["f1_status"] == "ready":
        pairs = pd.read_parquet(root / "parquet/f1/pairs.parquet")
        if len(pairs):
            best_pairs = {
                tuple(sorted((r.model_id_a, r.model_id_b)))
                for r in pairs[pairs.is_best_worst].itertuples()
            }
    f1 = manifest["f1_status"] == "ready"
    reader = (
        joined_geometry if f1 else lambda path, **kwargs: read_geometry(path, **kwargs).to_pandas()
    )
    output = root / "figures"
    output.mkdir(exist_ok=True)
    saved, issues = [], []

    def save(name, draw, context):
        try:
            if len(saved) >= max_figures:
                raise ValueError(
                    "Figure budget reached; request a selected subset for additional panels"
                )
            check_storage(
                root,
                plan["config"]["storage_policy"],
                required_bytes=2 * 1024**2,
                required_inodes=2,
            )
            fig = draw()
            fig.suptitle("Descriptivo — cobertura " + manifest["state"])
            fig.tight_layout()
            path = output / (name + ".png")
            temporary = path.with_suffix(".partial.png")
            fig.savefig(temporary, dpi=140)
            temporary.replace(path)
            saved.append(dict(path=str(path.relative_to(root)), selection=context))
        except Exception as exc:
            issues.append(dict(name=name, reason=str(exc)))
        finally:
            plt.close("all")

    pairs = reader(
        root,
        filters=dict(
            scope="encoder",
            metric=list(metrics),
            family=[
                "same_language_cross_entity",
                "same_entity_cross_language",
                "cross_language_cross_entity",
                "cross_backbone_intrinsic",
            ],
        ),
    )
    # One scale per metric/source across comparable panels; no aggregation in plotting.
    for (metric, source), comparable in pairs.groupby(["metric", "source"], dropna=False):
        values = comparable.value.dropna()
        if values.empty:
            continue
        low, high = float(values.min()), float(values.max())
        high = max(high, low + 1e-12)
        for (backbone, language, entity), frame in comparable.groupby(
            ["backbone", "language", "entity"], dropna=False
        ):
            if frame.empty:
                continue
            rows = frame.to_dict("records")
            ids = set(frame.artifact_a.dropna()) | set(frame.artifact_b.dropna())
            scores, labels, model_ids = {}, {}, {}
            for row in rows:
                for side in ("a", "b"):
                    aid = row["artifact_" + side]
                    score = row.get("strict_f1_mean_" + side)
                    sd = row.get("strict_f1_std_" + side)
                    scores[aid] = float(score) if score is not None and pd.notna(score) else -np.inf
                    model_ids[aid] = row.get("model_id_" + side, aid)
                    label = (
                        str(row.get("backbone_" + side, aid))
                        if backbone == "__cross_backbone__"
                        else str(row["language_" + side]) + "/" + str(row["entity_" + side])
                    )
                    labels[aid] = label + (
                        f"\nF1={score:.3f} ± {sd:.3f}" if pd.notna(score) and pd.notna(sd) else ""
                    )
            order = sorted(ids, key=lambda aid: (-scores[aid], str(model_ids[aid])))

            def heatmap(selected, ordered):
                matrix = np.full((len(ordered), len(ordered)), np.nan)
                locations = {aid: index for index, aid in enumerate(ordered)}
                seen = set()
                for row in selected:
                    a, b = row["artifact_a"], row["artifact_b"]
                    pair = tuple(sorted((a, b)))
                    if pair in seen:
                        raise ValueError("Duplicate pair; select scope/replica explicitly")
                    seen.add(pair)
                    i, j = locations[a], locations[b]
                    matrix[i, j] = matrix[j, i] = (
                        row["value"] if row["value"] is not None else np.nan
                    )
                fig, ax = plt.subplots(figsize=(12, 10))
                image = ax.imshow(np.ma.masked_invalid(matrix), vmin=low, vmax=high)
                ax.set_xticks(range(len(ordered)), [labels[aid] for aid in ordered], rotation=90)
                ax.set_yticks(range(len(ordered)), [labels[aid] for aid in ordered])
                ax.set_title(f"{backbone} / {language} / {entity} / {metric} / {source}")
                fig.colorbar(image, ax=ax)
                return fig

            name = (
                "pairs-"
                + str(backbone)
                + "-"
                + str(language)
                + "-"
                + str(entity)
                + "-"
                + metric
                + "-"
                + str(source)
            )
            context = dict(
                metric=metric,
                source=source,
                backbone=backbone,
                language=language,
                entity=entity,
                scale=[low, high],
            )
            save(name, lambda r=rows, o=order: heatmap(r, o), context)
            if f1 and backbone == "__cross_backbone__" and len(order) >= 2:
                selected = [
                    r
                    for r in rows
                    if tuple(sorted((r["model_id_a"], r["model_id_b"]))) in best_pairs
                ]
                if selected:
                    best, worst = selected[0]["artifact_a"], selected[0]["artifact_b"]
                    save(
                        "best-worst-" + name,
                        lambda s=selected, o=[best, worst]: heatmap(s, o),
                        context,
                    )
            if f1:

                def scatter(data=frame):
                    fig, ax = plt.subplots()
                    valid = data.dropna(subset=["abs_f1_gap", "value"])
                    ax.scatter(valid.abs_f1_gap, valid.value, alpha=0.5)
                    ax.set(
                        xlabel="|Δ F1 estricto medio|",
                        ylabel=metric,
                        title="Pares dependientes; asociación descriptiva",
                    )
                    return fig

                save("f1-gap-" + name, scatter, context)
    for scope, axis in (
        ("layer", "layer"),
        ("component", "component"),
        ("embeddings", "embedding_type"),
    ):
        frame = reader(
            root,
            filters=dict(scope=scope, family="checkpoint_vs_pretraining", metric="relative_l2"),
        )
        if scope == "component" and len(frame):
            frame = frame.copy()
            frame["component"] = frame.component + frame.embedding_type.map(
                lambda value: ":" + value if isinstance(value, str) else ""
            )
        for (backbone, language), group in frame.groupby(["backbone", "language"], dropna=False):

            def profile(data=group):
                fig, ax = plt.subplots(figsize=(10, 6))
                for aid, series in data.groupby("artifact_a", sort=True):
                    if series[axis].duplicated().any():
                        raise ValueError("Profile mixes components or replicas; select explicitly")
                    series = series.sort_values(axis)
                    ax.plot(
                        series[axis], series.value, marker="o", label=str(series.entity.iloc[0])
                    )
                ax.set(
                    xlabel=axis,
                    ylabel="||ΔW|| / ||Wpre||",
                    title=str(backbone) + " / " + str(language),
                )
                ax.legend()
                return fig

            save(
                f"pretraining-{backbone}-{language}-{scope}",
                profile,
                dict(scope=scope, backbone=backbone, language=language),
            )
    atomic_json(
        root / "reports/figures.json",
        dict(
            run_id=manifest["run_id"],
            files=saved,
            issues=issues,
            policy="persisted values only; no averaging in plot functions",
        ),
    )
    print("[FIGURES] saved", len(saved), "issues", len(issues), flush=True)
    return saved, issues
