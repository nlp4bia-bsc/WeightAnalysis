"""Historical layer correlations with explicit paths and unchanged formulas.

This opt-in report preserves historical exploratory medians/permutations. It is
not part of the normalized comparative workflow or a confirmatory analysis.
"""

from __future__ import annotations

from pathlib import Path


def run(*, data_dir=None, performance=None, output_dir, seed=2026, permutations=0, input_path=None):
    """Reproduce the historical report without executing I/O on import."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(
        "[LEGACY REPORT] Exploratory definitions; aggregate F1 is not fold-specific performance.",
        flush=True,
    )

    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    base = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    f1 = pd.read_parquet(
        Path(performance), columns=["language", "entity", "model", "strict_f1_mean"]
    ).rename(columns={"model": "backbone"})

    data = pd.read_parquet(
        base / "01_pretraining_drift.parquet",
        columns=[
            "level",
            "fold",
            "backbone",
            "language",
            "entity",
            "layer",
            "component",
            "relative_l2",
            "delta_rms",
            "n_parameters",
        ],
    )

    data = data[(data["fold"] == 1) & data["level"].isin(["layer", "layer_component"])].copy()

    data = data.merge(
        f1, on=["backbone", "language", "entity"], how="inner", validate="many_to_one"
    )

    # Comparación dentro de un mismo índice de capa.
    # Las profundidades de las arquitecturas pueden variar,
    # por lo que también empleamos profundidad relativa.
    max_layer = data.groupby("backbone")["layer"].transform("max")

    data["relative_depth"] = np.where(max_layer > 0, data["layer"] / max_layer, 0.0)

    data["depth_zone"] = pd.cut(
        data["relative_depth"],
        bins=[-0.01, 1 / 3, 2 / 3, 1.01],
        labels=["inferior", "media", "superior"],
    )

    # Nos quedamos con una observación por
    # modelo × idioma × entidad × zona × nivel.
    groups = ["language", "entity", "backbone", "level", "depth_zone"]

    data["relative_l2"] = data["relative_l2"].replace([np.inf, -np.inf], np.nan)

    agg = data.groupby(groups, observed=True, as_index=False).agg(
        relative_l2=("relative_l2", "median"),
        delta_rms=("delta_rms", "median"),
        strict_f1_mean=("strict_f1_mean", "first"),
    )

    results = []

    for (lang, ent, level, zone), g in agg.groupby(
        ["language", "entity", "level", "depth_zone"], observed=True
    ):
        for metric in ["relative_l2", "delta_rms"]:
            g2 = g.dropna(subset=[metric, "strict_f1_mean"])

            if len(g2) < 8:
                continue

            rho = spearmanr(g2[metric], g2["strict_f1_mean"]).statistic

            results.append(
                {
                    "language": lang,
                    "entity": ent,
                    "level": level,
                    "depth_zone": str(zone),
                    "metric": metric,
                    "n_backbones": len(g2),
                    "rho": rho,
                }
            )

    result = pd.DataFrame(results)

    result.to_csv(out / "correlacion_capas_f1.csv", index=False)

    agg.to_csv(out / "perfil_capas_f1.csv", index=False)

    print("\n=== CORRELACIONES POR PROFUNDIDAD ===")

    print(
        result.groupby(["level", "depth_zone", "metric"])["rho"]
        .agg(["median", "count", lambda x: (x < 0).sum()])
        .round(4)
        .to_string()
    )

    print("\nResultados guardados.")
