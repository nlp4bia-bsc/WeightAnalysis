"""Plot explicitly grouped profiles from already computed analysis results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..results import AnalysisResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes


IDENTITY_COLUMNS = (
    "checkpoint",
    "source",
    "backbone",
    "language",
    "entity",
    "fold",
    "seed",
    "component",
    "checkpoint_a",
    "checkpoint_b",
    "language_a",
    "language_b",
    "entity_a",
    "entity_b",
    "fold_a",
    "fold_b",
    "seed_a",
    "seed_b",
    "comparison_kind",
)


def plot_metric(
    result: pd.DataFrame | AnalysisResult,
    *,
    metric: str,
    x: str = "layer",
    group_by: Sequence[str] = ("checkpoint", "source"),
    ax: Axes | None = None,
) -> Axes:
    """Draw metric profiles without reading weights, aggregating or writing files.

    Tensor rows are selected when a level column is present. Repeated x values
    or mixed scientific identities within a group are rejected: callers must
    select rows, add grouping columns or aggregate explicitly before plotting.
    Missing x/metric values are omitted. The returned axes can be saved or
    displayed explicitly by the caller. Matplotlib is an optional dependency.
    """
    frame = result.data if isinstance(result, AnalysisResult) else result
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("result must be a pandas DataFrame or AnalysisResult")
    missing = {x, metric} - set(frame.columns)
    if missing:
        raise ValueError(f"result is missing plot columns: {sorted(missing)}")
    if isinstance(group_by, str):
        raise TypeError("group_by must be a sequence of column names, not a string")
    if len(set(group_by)) != len(group_by):
        raise ValueError("group_by columns must be unique")
    default_groups = ("checkpoint", "source")
    groups = [column for column in group_by if column in frame]
    if tuple(group_by) != default_groups:
        absent = set(group_by) - set(frame.columns)
        if absent:
            raise ValueError(f"result is missing group_by columns: {sorted(absent)}")

    frame = frame.copy()
    if "level" in frame:
        frame = frame[frame.level == "tensor"].copy()
    try:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"metric column {metric!r} must be numeric") from error
    if pd.api.types.is_bool_dtype(frame[metric]):
        raise ValueError(f"metric column {metric!r} must contain numeric measurements")
    frame = frame.dropna(subset=[x, metric])
    if frame.empty:
        raise ValueError("result contains no nonmissing tensor observations to plot")
    if not np.isfinite(frame[metric].to_numpy(dtype=float)).all():
        raise ValueError(f"metric column {metric!r} must contain finite measurements")
    if (
        pd.api.types.is_numeric_dtype(frame[x])
        and not np.isfinite(frame[x].to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"x column {x!r} must contain finite positions")
    grouped = list(frame.groupby(groups, dropna=False, sort=True)) if groups else [((), frame)]
    profiles = []
    for identity, observations in grouped:
        mixed = [
            column
            for column in IDENTITY_COLUMNS
            if column in observations and observations[column].nunique(dropna=False) > 1
        ]
        if mixed:
            raise ValueError(f"group_by must separate scientific identities: {mixed}")
        if observations[x].duplicated().any():
            raise ValueError(
                f"Multiple observations at the same {x!r}; add group_by columns or aggregate explicitly"
            )
        observations = observations.sort_values(x, kind="stable")
        identity = identity if isinstance(identity, tuple) else (identity,)
        label = " / ".join(str(value) for value in identity) if identity else metric
        profiles.append((label, observations[x], observations[metric]))

    if ax is None:
        try:
            from matplotlib import pyplot as plt
        except ImportError as error:
            raise ImportError(
                "plot_metric requires pip install 'weight-analysis[visualization]'"
            ) from error
        _, ax = plt.subplots()
    for label, positions, values in profiles:
        ax.plot(positions, values, marker="o", label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(metric)
    if len(profiles) > 1 or groups:
        ax.legend()

    return ax
