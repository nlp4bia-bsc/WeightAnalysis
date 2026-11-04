from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

path = Path("resultados/rankings_f1.csv")
df = pd.read_csv(path)

rows = []

for (lang, ent), group in df.groupby(
    ["language", "entity"]
):
    for excluded in group["backbone"]:
        g = group[group["backbone"] != excluded]
        g = g.dropna(
            subset=["relative_l2", "strict_f1_mean"]
        )

        if len(g) < 8:
            continue

        rho = spearmanr(
            g["relative_l2"],
            g["strict_f1_mean"]
        ).statistic

        rows.append({
            "language": lang,
            "entity": ent,
            "excluded": excluded,
            "rho": rho
        })

out = pd.DataFrame(rows)
out.to_csv(
    "resultados/leave_one_backbone_out.csv",
    index=False
)

summary = out.groupby(
    ["language", "entity"]
)["rho"].agg(["min", "median", "max"])

print(summary.round(4).to_string())

print("\nGrupos que mantienen rho negativa")
print("en todas las exclusiones:")

stable = (
    out.groupby(["language", "entity"])["rho"]
    .max() < 0
)

print(stable.to_string())
print(f"\nTotal: {stable.sum()}/{len(stable)}")
