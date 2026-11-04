from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(
    "/mnt/fscratch/users/gaias_001_uma/fgd/"
    "ner_lab/TFG_Aziz/weight_analysis_trial460"
)
PERFORMANCE = Path(
    "/mnt/home/users/gaias_001_uma/fgd/"
    "ner_lab/TFG_Aziz/results_aggregated.parquet"
)

KEY = ["language", "entity", "backbone"]
RNG = np.random.default_rng(2026)

f1 = pd.read_parquet(PERFORMANCE)
f1 = f1.rename(columns={"model": "backbone"})
f1 = f1.dropna(subset=["strict_f1_mean"]).copy()

if f1.duplicated(KEY).any():
    raise ValueError("Hay claves F1 duplicadas")

print("=== COBERTURA ===")
print("Combinaciones con F1:", len(f1))
print("Número de folds:")
print(f1["n_folds"].value_counts().to_string())

# 1. Magnitud de actualización frente a F1
w = pd.read_parquet(
    ROOT / "01_pretraining_drift.parquet",
    columns=[
        "level", "backbone", "language", "entity",
        "relative_l2", "delta_rms", "delta_energy"
    ]
)
w = w[w["level"] == "encoder"].copy()

w = w.groupby(KEY, as_index=False)[
    ["relative_l2", "delta_rms", "delta_energy"]
].median()

missing = w.merge(f1[KEY], on=KEY, how="left", indicator=True)
missing = missing[missing["_merge"] == "left_only"]

print("\nCheckpoints sin F1:")
print(missing[KEY].to_string(index=False))

m = w.merge(
    f1[KEY + ["strict_f1_mean", "char_f1_mean"]],
    on=KEY,
    validate="one_to_one"
)
print("Checkpoints cruzados con F1:", len(m))

print("\n=== 01: F1 Y MAGNITUD DEL CAMBIO ===")

for metric in ["relative_l2", "delta_rms", "delta_energy"]:
    rows = []
    for (lang, ent), g in m.groupby(["language", "entity"]):
        g = g.dropna(subset=[metric])
        if len(g) >= 8:
            rho = spearmanr(
                g[metric], g["strict_f1_mean"]
            ).statistic
            rows.append((lang, ent, len(g), rho))

    result = pd.DataFrame(
        rows, columns=["language", "entity", "n", "rho"]
    )
    print(f"\nMétrica: {metric}")
    print(result.round(4).to_string(index=False))
    print("Mediana rho:", round(result["rho"].median(), 4))
    print("Correlaciones positivas:", (result["rho"] > 0).sum(),
          "de", len(result))

# 2. Distancias entre backbones frente a F1
p = pd.read_parquet(ROOT / "07_cross_backbone.parquet")

a = f1[KEY + ["strict_f1_mean"]].rename(columns={
    "backbone": "backbone_a",
    "strict_f1_mean": "f1_a"
})
b = f1[KEY + ["strict_f1_mean"]].rename(columns={
    "backbone": "backbone_b",
    "strict_f1_mean": "f1_b"
})

p = p.merge(a, on=["language", "entity", "backbone_a"])
p = p.merge(b, on=["language", "entity", "backbone_b"])
p["f1_gap"] = (p["f1_a"] - p["f1_b"]).abs()

print("\n=== 07: DISTANCIA ESTRUCTURAL Y DIFERENCIA F1 ===")
print("Pares cruzados:", len(p))

METRICS = [
    "trace_distance_delta",
    "curve_l1_delta",
    "quantile_w1_delta",
    "adaptation_profile_l1",
]

comparisons = []

for (lang, ent), g in p.groupby(["language", "entity"]):
    eligible = set(g["backbone_a"]) | set(g["backbone_b"])

    ranking = f1[
        (f1["language"] == lang) &
        (f1["entity"] == ent) &
        f1["backbone"].isin(eligible)
    ].sort_values("strict_f1_mean", ascending=False)

    if len(ranking) < 6:
        continue

    best = set(ranking.head(3)["backbone"])
    worst = set(ranking.tail(3)["backbone"])

    aa = g["backbone_a"]
    bb = g["backbone_b"]

    top_top = aa.isin(best) & bb.isin(best)
    bottom_bottom = aa.isin(worst) & bb.isin(worst)
    top_bottom = (
        (aa.isin(best) & bb.isin(worst)) |
        (aa.isin(worst) & bb.isin(best))
    )

    for metric in METRICS:
        x = g[metric]

        rho = spearmanr(
            x, g["f1_gap"], nan_policy="omit"
        ).statistic

        d_tt = x[top_top].mean()
        d_bb = x[bottom_bottom].mean()
        d_tb = x[top_bottom].mean()

        comparisons.append({
            "language": lang,
            "entity": ent,
            "metric": metric,
            "n_pairs": len(g),
            "rho_gap": rho,
            "top_top": d_tt,
            "bottom_bottom": d_bb,
            "top_bottom": d_tb,
            "contrast": d_tb - d_tt
        })

comparisons = pd.DataFrame(comparisons)

print("\n=== RESUMEN POR MÉTRICA ===")
for metric, g in comparisons.groupby("metric"):
    print("\n", metric)
    print("Rho mediana distancia vs diferencia F1:",
          round(g["rho_gap"].median(), 5))
    print("Grupos con rho positiva:",
          int((g["rho_gap"] > 0).sum()), "/", len(g))
    print("Contraste TopBottom - TopTop mediano:",
          round(g["contrast"].median(), 5))
    print("Grupos donde TopTop es más próximo:",
          int((g["contrast"] > 0).sum()), "/", len(g))

print("\n=== CONTRASTES POR IDIOMA ===")
for metric in METRICS:
    print("\n", metric)
    g = comparisons[comparisons["metric"] == metric]
    print(
        g.groupby("language")[
            ["rho_gap", "contrast"]
        ].median().round(5).to_string()
    )

# 3. Control por similitud estructural inicial
print("\n=== CONTROL POR DISTANCIA ORIGINAL ===")
perm_results = []

for (lang, ent), g in p.groupby(["language", "entity"]):
    if len(g) < 50:
        continue

    names = sorted(
        set(g["backbone_a"]) | set(g["backbone_b"])
    )

    ranks = f1[
        (f1["language"] == lang) &
        (f1["entity"] == ent) &
        f1["backbone"].isin(names)
    ].sort_values("strict_f1_mean", ascending=False)

    names = ranks["backbone"].tolist()
    if len(names) < 6:
        continue

    x = g["trace_distance_base"].to_numpy()
    y = g["trace_distance_delta"].to_numpy()

    # Elimina la tendencia lineal asociada a la
    # distancia estructural existente antes de entrenar.
    slope = np.polyfit(x, y, 1)[0]
    residual = y - (
        y.mean() + slope * (x - x.mean())
    )

    left = g["backbone_a"].to_numpy()
    right = g["backbone_b"].to_numpy()

    def contrast(best, worst):
        tt = np.isin(left, best) & np.isin(right, best)
        tb = (
            (np.isin(left, best) & np.isin(right, worst)) |
            (np.isin(left, worst) & np.isin(right, best))
        )
        if not tt.any() or not tb.any():
            return np.nan
        return residual[tb].mean() - residual[tt].mean()

    observed = contrast(names[:3], names[-3:])

    null = []
    for _ in range(1000):
        shuffled = RNG.permutation(names)
        null.append(
            contrast(shuffled[:3], shuffled[-3:])
        )

    null = np.asarray(null)
    null = null[np.isfinite(null)]

    p_value = (
        (1 + (null >= observed).sum()) / (len(null) + 1)
        if len(null) else np.nan
    )

    perm_results.append({
        "language": lang,
        "entity": ent,
        "effect": observed,
        "permutation_p_exploratory": p_value
    })

perm = pd.DataFrame(perm_results)
print(perm.round(5).to_string(index=False))

# 4. Rankings concretos para interpretar resultados
print("\n=== MEJORES Y PEORES POR IDIOMA Y ENTIDAD ===")
for (lang, ent), g in f1.groupby(["language", "entity"]):
    g = g.sort_values("strict_f1_mean", ascending=False)
    print(f"\n{lang} / {ent}")
    print("TOP 3:")
    print(
        g[["backbone", "strict_f1_mean"]]
        .head(3).to_string(index=False)
    )
    print("BOTTOM 3:")
    print(
        g[["backbone", "strict_f1_mean"]]
        .tail(3).to_string(index=False)
    )

# Guardar tablas procesables
comparisons.to_csv(
    "f1_weightspace_comparaciones.csv", index=False
)
perm.to_csv(
    "f1_weightspace_permutaciones.csv", index=False
)

print("\n=== FIN ===")
