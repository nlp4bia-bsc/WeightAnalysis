"""Historical backbone sensitivity with explicit paths and unchanged formulas.

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

    import pandas as pd
    from scipy.stats import spearmanr

    path = Path(input_path) if input_path else Path(output_dir) / "rankings_f1.csv"
    df = pd.read_csv(path)

    rows = []

    for (lang, ent), group in df.groupby(["language", "entity"]):
        for excluded in group["backbone"]:
            g = group[group["backbone"] != excluded]
            g = g.dropna(subset=["relative_l2", "strict_f1_mean"])

            if len(g) < 8:
                continue

            rho = spearmanr(g["relative_l2"], g["strict_f1_mean"]).statistic

            rows.append({"language": lang, "entity": ent, "excluded": excluded, "rho": rho})

    out = pd.DataFrame(rows)
    out.to_csv((Path(output_dir) / "leave_one_backbone_out.csv"), index=False)

    summary = out.groupby(["language", "entity"])["rho"].agg(["min", "median", "max"])

    print(summary.round(4).to_string())

    print("\nGrupos que mantienen rho negativa")
    print("en todas las exclusiones:")

    stable = out.groupby(["language", "entity"])["rho"].max() < 0

    print(stable.to_string())
    print(f"\nTotal: {stable.sum()}/{len(stable)}")
