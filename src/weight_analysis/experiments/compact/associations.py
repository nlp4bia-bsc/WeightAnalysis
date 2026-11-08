"""Descriptive correlations only: dependent pairs never receive iid p-values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .plan import write_table
from .query import joined_geometry


def derive_associations(root):
    frame = joined_geometry(root, filters=dict(scope="encoder"))
    frame["abs_sd_gap"] = (frame.strict_f1_std_a - frame.strict_f1_std_b).abs()
    frame["mean_pair_f1"] = (frame.strict_f1_mean_a + frame.strict_f1_mean_b) / 2
    rows = []
    strata = ["family", "backbone", "language", "entity", "metric", "source", "fold_a", "seed_a"]
    for identity, group in frame.groupby(strata, dropna=False, sort=True):
        for target in (
            "abs_f1_gap",
            "strict_f1_mean_a",
            "strict_f1_std_a",
            "abs_sd_gap",
            "mean_pair_f1",
        ):
            valid = group[["value", target]].replace([np.inf, -np.inf], np.nan).dropna()
            correlation = (
                float(valid.value.corr(valid[target]))
                if len(valid) >= 3 and valid.nunique().min() > 1
                else None
            )
            rank_correlation = (
                float(valid.value.rank().corr(valid[target].rank()))
                if correlation is not None
                else None
            )
            endpoints = set(group.artifact_a.dropna()) | set(group.artifact_b.dropna())
            rows.append(
                dict(
                    zip(strata, identity),
                    target=target,
                    pearson=correlation,
                    spearman=rank_correlation,
                    n_observations=len(valid),
                    n_checkpoints=len(endpoints),
                    inference="descriptive_only_no_iid_p_values",
                    pair_dependence="shared_endpoints",
                    f1_scope="configuration_fold_aggregate_not_individual_checkpoint",
                    status="ok" if correlation is not None else "undefined",
                )
            )
    write_table(pd.DataFrame(rows), root / "reports/associations.parquet")
