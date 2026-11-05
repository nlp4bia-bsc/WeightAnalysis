"""Historical validation with explicit paths and unchanged formulas.

This opt-in report preserves historical exploratory medians/permutations. It is
not part of the normalized comparative workflow or a confirmatory analysis.
"""

from __future__ import annotations

from pathlib import Path


def run(
    *,
    data_dir=None,
    performance=None,
    output_dir,
    seed=20261030,
    permutations=1999,
    input_path=None,
):
    """Reproduce the historical report without executing I/O on import."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(
        "[LEGACY REPORT] Exploratory definitions; aggregate F1 is not fold-specific performance.",
        flush=True,
    )

    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    BASE = Path(data_dir)
    F1_PATH = Path(performance)
    OUT = Path(output_dir)
    OUT.mkdir(exist_ok=True)

    RNG = np.random.default_rng(seed)
    N_PERM = permutations

    KEY = ["language", "entity", "backbone"]

    METRICS = {
        "trace": ("trace_distance_base", "trace_distance_delta"),
        "spectral_curve": ("curve_l1_base", "curve_l1_delta"),
        "quantiles": ("quantile_w1_base", "quantile_w1_delta"),
    }

    f1 = pd.read_parquet(F1_PATH)
    f1 = f1.rename(columns={"model": "backbone"})
    f1 = f1.dropna(subset=["strict_f1_mean"])

    assert not f1.duplicated(KEY).any(), "Hay modelos duplicados en F1"

    drift = pd.read_parquet(
        BASE / "01_pretraining_drift.parquet",
        columns=KEY + ["level", "fold", "relative_l2", "delta_rms", "delta_energy"],
    )
    drift = drift[(drift["level"] == "encoder") & (drift["fold"] == 1)].copy()

    assert not drift.duplicated(KEY).any(), "Hay varios registros encoder por checkpoint"

    joined = f1.merge(drift, on=KEY, how="inner", validate="one_to_one")

    pairs = pd.read_parquet(
        BASE / "07_cross_backbone.parquet",
        columns=[
            "language",
            "entity",
            "fold",
            "backbone_a",
            "backbone_b",
            "trace_distance_base",
            "trace_distance_delta",
            "curve_l1_base",
            "curve_l1_delta",
            "quantile_w1_base",
            "quantile_w1_delta",
            "adaptation_profile_l1",
        ],
    )

    pairs = pairs[pairs["fold"] == 1].copy()

    assert not pairs.duplicated(["language", "entity", "backbone_a", "backbone_b"]).any()

    fa = f1[KEY + ["strict_f1_mean"]].rename(
        columns={"backbone": "backbone_a", "strict_f1_mean": "f1_a"}
    )

    fb = f1[KEY + ["strict_f1_mean"]].rename(
        columns={"backbone": "backbone_b", "strict_f1_mean": "f1_b"}
    )

    pairs = pairs.merge(fa, on=["language", "entity", "backbone_a"], validate="many_to_one")
    pairs = pairs.merge(fb, on=["language", "entity", "backbone_b"], validate="many_to_one")

    pairs["f1_gap"] = (pairs["f1_a"] - pairs["f1_b"]).abs()

    rankings = []
    correlations = []
    contrasts = []
    permutations = []
    sensitivity = []

    def spearman(x, y):
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 6:
            return np.nan
        if len(np.unique(x[valid])) < 2:
            return np.nan
        if len(np.unique(y[valid])) < 2:
            return np.nan
        return spearmanr(x[valid], y[valid]).statistic

    def distances_for_group(frame, scores, metric, top_k=3):
        ordered = sorted(scores, key=lambda name: (-scores[name], name))

        if len(ordered) < top_k * 2:
            return np.nan, np.nan, np.nan

        top = set(ordered[:top_k])
        bottom = set(ordered[-top_k:])

        a = frame["backbone_a"].to_numpy()
        b = frame["backbone_b"].to_numpy()
        d = frame[metric].to_numpy(dtype=float)

        tt = np.isin(a, list(top)) & np.isin(b, list(top))
        bb = np.isin(a, list(bottom)) & np.isin(b, list(bottom))
        tb = (np.isin(a, list(top)) & np.isin(b, list(bottom))) | (
            np.isin(a, list(bottom)) & np.isin(b, list(top))
        )

        def avg(mask):
            v = d[mask]
            v = v[np.isfinite(v)]
            return float(np.mean(v)) if len(v) else np.nan

        return avg(tt), avg(bb), avg(tb)

    def permutation_test(frame, scores, metric, n_perm):
        tt, bb, tb = distances_for_group(frame, scores, metric)
        observed = tb - tt

        if not np.isfinite(observed):
            return observed, np.nan

        names = sorted(scores)
        values = np.array([scores[n] for n in names])

        null = np.empty(n_perm)

        for i in range(n_perm):
            shuffled = RNG.permutation(values)
            random_scores = dict(zip(names, shuffled))

            rtt, _, rtb = distances_for_group(frame, random_scores, metric)
            null[i] = rtb - rtt

        null = null[np.isfinite(null)]

        if len(null) == 0:
            return observed, np.nan

        p = (1 + np.sum(null >= observed)) / (1 + len(null))
        return observed, p

    for (lang, ent), g in joined.groupby(["language", "entity"]):
        g = g.sort_values("strict_f1_mean", ascending=False)

        for rank, (_, r) in enumerate(g.iterrows(), 1):
            rankings.append(
                {
                    "language": lang,
                    "entity": ent,
                    "rank": rank,
                    "backbone": r["backbone"],
                    "strict_f1_mean": r["strict_f1_mean"],
                    "relative_l2": r["relative_l2"],
                    "delta_rms": r["delta_rms"],
                    "delta_energy": r["delta_energy"],
                }
            )

        for metric in ["relative_l2", "delta_rms", "delta_energy"]:
            rho = spearman(
                g[metric].to_numpy(dtype=float), g["strict_f1_mean"].to_numpy(dtype=float)
            )
            correlations.append(
                {
                    "language": lang,
                    "entity": ent,
                    "metric": metric,
                    "n_models": len(g),
                    "spearman_rho": rho,
                    "analysis": "drift_vs_f1",
                }
            )

        pg = pairs[(pairs["language"] == lang) & (pairs["entity"] == ent)].copy()

        scores = dict(zip(g["backbone"], g["strict_f1_mean"]))

        # Comparación de magnitud estructural y F1
        for name, (base_metric, delta_metric) in METRICS.items():
            for kind, metric in [("base", base_metric), ("delta", delta_metric)]:
                rho = spearman(pg[metric].to_numpy(dtype=float), pg["f1_gap"].to_numpy(dtype=float))

                correlations.append(
                    {
                        "language": lang,
                        "entity": ent,
                        "metric": f"{name}_{kind}",
                        "n_models": len(g),
                        "spearman_rho": rho,
                        "analysis": "distance_vs_f1_gap",
                    }
                )

                tt, bb, tb = distances_for_group(pg, scores, metric)

                contrasts.append(
                    {
                        "language": lang,
                        "entity": ent,
                        "metric": metric,
                        "top_top": tt,
                        "bottom_bottom": bb,
                        "top_bottom": tb,
                        "topbottom_minus_toptop": tb - tt,
                        "topbottom_minus_bottombottom": tb - bb,
                        "topbottom_minus_within_avg": (tb - np.nanmean([tt, bb])),
                    }
                )

            # Control simple por distancia original:
            # residuo de delta tras ajustar por base
            x = pg[base_metric].to_numpy(dtype=float)
            y = pg[delta_metric].to_numpy(dtype=float)

            valid = np.isfinite(x) & np.isfinite(y)

            if valid.sum() >= 8 and np.std(x[valid]) > 0:
                matrix = np.column_stack([np.ones(valid.sum()), x[valid]])
                beta = np.linalg.lstsq(matrix, y[valid], rcond=None)[0]

                pg[f"residual_{name}"] = np.nan
                pg.loc[valid, f"residual_{name}"] = y[valid] - matrix @ beta

                residual_metric = f"residual_{name}"

                observed, p_value = permutation_test(pg, scores, residual_metric, N_PERM)

                permutations.append(
                    {
                        "language": lang,
                        "entity": ent,
                        "metric": name,
                        "contrast_residual": observed,
                        "p_one_sided": p_value,
                    }
                )

        # Sensibilidad: excluir RuBioRoBERTa
        excluded = "alexyalunin--RuBioRoBERTa"

        reduced = pg[(pg["backbone_a"] != excluded) & (pg["backbone_b"] != excluded)].copy()

        reduced_scores = {k: v for k, v in scores.items() if k != excluded}

        for name, (_, metric) in METRICS.items():
            tt, bb, tb = distances_for_group(reduced, reduced_scores, metric)

            sensitivity.append(
                {
                    "language": lang,
                    "entity": ent,
                    "metric": name,
                    "without_backbone": excluded,
                    "topbottom_minus_toptop": tb - tt,
                    "n_models": len(reduced_scores),
                }
            )

    def save(data, name):
        df = pd.DataFrame(data)
        df.to_csv(OUT / name, index=False)
        return df

    rank = save(rankings, "rankings_f1.csv")
    corr = save(correlations, "correlaciones.csv")
    cont = save(contrasts, "top_bottom_contrastes.csv")
    perm = save(permutations, "permutaciones_control_base.csv")
    save(sensitivity, "sensibilidad_sin_rubio.csv")

    # Corrección FDR por las 21 pruebas de cada métrica
    perm["q_bh"] = np.nan

    for metric, idx in perm.groupby("metric").groups.items():
        pvals = perm.loc[idx, "p_one_sided"]
        mask = pvals.notna()

        if mask.any():
            q = multipletests(pvals[mask].to_numpy(), method="fdr_bh")[1]
            perm.loc[pvals[mask].index, "q_bh"] = q

    perm.to_csv(OUT / "permutaciones_control_base.csv", index=False)

    # Resumen descriptivo automático
    with open(OUT / "resumen_ejecutivo.md", "w", encoding="utf-8") as h:
        h.write("# Validación F1 y WeightSpace\n\n")
        h.write(
            "Resultados exploratorios. Los pares comparten "
            "backbones y no son réplicas independientes.\n\n"
        )

        h.write("## Cobertura\n\n")
        h.write(f"- Modelos con F1 y drift: {len(joined)}\n")
        h.write(f"- Pares con F1: {len(pairs)}\n")
        h.write(f"- Permutaciones por grupo: {N_PERM}\n\n")

        h.write("## Correlaciones F1–drift\n\n")
        for metric, group in corr[corr["analysis"] == "drift_vs_f1"].groupby("metric"):
            h.write(
                f"- {metric}: rho mediana "
                f"{group['spearman_rho'].median():.4f}; "
                f"{(group['spearman_rho'] < 0).sum()}"
                f"/{len(group)} negativas.\n"
            )

        h.write("\n## Distancia estructural y diferencia F1\n\n")
        for metric, group in corr[corr["analysis"] == "distance_vs_f1_gap"].groupby("metric"):
            h.write(
                f"- {metric}: rho mediana "
                f"{group['spearman_rho'].median():.4f}; "
                f"{(group['spearman_rho'] > 0).sum()}"
                f"/{len(group)} positivas.\n"
            )

        h.write("\n## Top–Top vs Top–Bottom\n\n")
        for metric, group in cont.groupby("metric"):
            vals = group["topbottom_minus_toptop"]
            h.write(
                f"- {metric}: contraste mediano "
                f"{vals.median():.5f}; "
                f"{(vals > 0).sum()}/{len(vals)} positivos.\n"
            )

        h.write("\n## Permutaciones con control base\n\n")
        for metric, group in perm.groupby("metric"):
            h.write(
                f"- {metric}: "
                f"{(group['p_one_sided'] < 0.05).sum()} "
                f"p nominales < 0.05; "
                f"{(group['q_bh'] < 0.05).sum()} "
                f"q BH < 0.05.\n"
            )

        h.write(
            "\n## Advertencia\n\n"
            "Estos p y q son exploratorios y locales. "
            "Se requiere una prueba global que preserve "
            "la dependencia entre idiomas, entidades y "
            "backbones antes de declarar confirmación.\n"
        )

    print("\nVALIDACIÓN FINALIZADA")
    print("Checkpoints con F1:", len(joined))
    print("Pares con F1:", len(pairs))
    print("Archivos disponibles en:", OUT.resolve())
    print("\nRESUMEN:")
    print((OUT / "resumen_ejecutivo.md").read_text())
