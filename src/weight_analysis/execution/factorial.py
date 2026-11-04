"""Selective factorial for independent scientific computations."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .catalog import FACTORIAL_METRICS
from .descriptions import _base


def _factorial(checkpoints, originals, *, layers, components, tensors, metrics, options):
    from ..analyses import factorial, validate_factorial_design

    strata = defaultdict(list)
    for checkpoint in checkpoints:
        strata[checkpoint.backbone, checkpoint.fold, checkpoint.seed].append(checkpoint)
    rows = []
    for (_, fold, seed), group in sorted(strata.items(), key=lambda item: str(item[0])):
        validate_factorial_design(group)
        for checkpoint in group:
            _base(checkpoint, originals)
        base_path = originals[group[0].backbone]
        values = factorial(
            group, base_path, options, layers=layers, components=components, tensors=tensors
        )
        for row in values:
            row = {
                key: value
                for key, value in row.items()
                if key not in FACTORIAL_METRICS or key in metrics
            }
            rows.append({**row, "fold": fold, "seed": seed})
    return pd.DataFrame(rows)
