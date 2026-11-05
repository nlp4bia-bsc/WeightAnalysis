"""Deterministic, nonclinical wide tables for historical report regression oracles."""

from itertools import combinations
from pathlib import Path

import pandas as pd


def report_inputs(root):
    root = Path(root)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    scores = []
    drift = []
    pairs = []
    for i in range(12):
        backbone = f"model-{i:02d}"
        scores.append(
            dict(
                language="es",
                entity="disease",
                model=backbone,
                n_folds=4,
                strict_f1_mean=0.3 + i * 0.04,
                strict_f1_std=0.01,
                char_f1_mean=0.4 + i * 0.04,
            )
        )
        for level, layer, component in [("encoder", None, None)] + [
            (kind, j, "q" if kind == "layer_component" else None)
            for kind in ("layer", "layer_component")
            for j in range(3)
        ]:
            x = (i + 1) * 0.02 * (1 if layer is None else layer + 1)
            drift.append(
                dict(
                    language="es",
                    entity="disease",
                    backbone=backbone,
                    level=level,
                    fold=1,
                    layer=layer,
                    component=component,
                    n_parameters=100,
                    relative_l2=x,
                    delta_rms=x / 2,
                    delta_energy=x * x * 100,
                )
            )
    for i, j in combinations(range(12), 2):
        base = (j - i) / 13 + (i + j) / 1000
        delta = base * 0.4 + (i + j) / 50 + (i * j % 7) / 100
        pairs.append(
            dict(
                language="es",
                entity="disease",
                fold=1,
                backbone_a=f"model-{i:02d}",
                backbone_b=f"model-{j:02d}",
                trace_distance_base=base,
                trace_distance_delta=delta,
                curve_l1_base=base * 0.8,
                curve_l1_delta=delta * 0.7,
                quantile_w1_base=base * 0.6,
                quantile_w1_delta=delta * 0.9,
                adaptation_profile_l1=delta * 0.5,
            )
        )
    performance = root / "f1.parquet"
    pd.DataFrame(scores).to_parquet(performance, index=False)
    pd.DataFrame(drift).to_parquet(data / "01_pretraining_drift.parquet", index=False)
    pd.DataFrame(pairs).to_parquet(data / "07_cross_backbone.parquet", index=False)
    return data, performance
