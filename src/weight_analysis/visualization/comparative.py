"""Comparative figures from explicit, already aggregated geometric tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColorScale:
    metric: str
    minimum: float
    maximum: float

    def __post_init__(self):
        if not np.isfinite([self.minimum, self.maximum]).all() or self.minimum >= self.maximum:
            raise ValueError("Color scale limits must be finite and increasing")


def _selected(frame, metric):
    required = {"metric", "value"}
    if required - set(frame):
        raise ValueError(f"Plot table missing {sorted(required - set(frame))}")
    selected = frame[frame.metric == metric].copy()
    if selected.empty:
        raise ValueError("No observations for selected metric")
    for column in (
        "source",
        "reference_id",
        "scope",
        "component",
        "embedding_type",
        "layer",
        "relative_depth",
        "selection_id",
    ):
        if column in selected and selected[column].nunique(dropna=False) > 1:
            raise ValueError(f"Select a single {column} or explicitly aggregate before plotting")
    return selected


def _axes(ax):
    if ax is not None:
        return ax
    from matplotlib import pyplot as plt

    return plt.subplots()[1]


def plot_pair_heatmap(
    frame, *, metric, axis="artifact", order=None, scale=None, ax=None, directed=False
):
    """Plot unique pair values; language/entity variants use the same implementation."""
    selected = _selected(frame, metric)
    left, right = axis + "_a", axis + "_b"
    if {left, right} - set(selected):
        raise ValueError("Heatmap endpoint columns are missing")
    for column in ("fold_a", "fold_b", "seed_a", "seed_b", "language", "entity", "backbone"):
        if column in selected and selected[column].nunique(dropna=False) > 1:
            raise ValueError(f"Heatmap mixes {column}; select a stratum first")
    if not directed and metric in ("norm_a", "norm_b"):
        raise ValueError("Endpoint norms are not symmetric pair distances")
    pairs = [
        tuple(sorted((str(a), str(b)))) if not directed else (str(a), str(b))
        for a, b in zip(selected[left], selected[right])
    ]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate heatmap pairs require explicit scientific aggregation")
    identities = set(selected[left].astype(str)) | set(selected[right].astype(str))
    labels = list(map(str, order)) if order is not None else sorted(identities)
    if len(labels) != len(set(labels)) or set(labels) != identities:
        raise ValueError("order must list every displayed identity exactly once")
    matrix = np.full((len(labels), len(labels)), np.nan)
    positions = {label: index for index, label in enumerate(labels)}
    for a, b, value in zip(selected[left].astype(str), selected[right].astype(str), selected.value):
        matrix[positions[a], positions[b]] = value
        if not directed:
            matrix[positions[b], positions[a]] = value
    if scale is not None and scale.metric != metric:
        raise ValueError("ColorScale belongs to a different metric")
    ax = _axes(ax)
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        vmin=scale.minimum if scale else None,
        vmax=scale.maximum if scale else None,
        cmap="viridis",
    )
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(metric)
    ax.figure.colorbar(image, ax=ax)
    return ax


def plot_geometry_profiles(
    frame,
    *,
    metric,
    x="layer",
    series=("subject_id", "source", "reference_id", "component"),
    ax=None,
):
    """Plot layer or component profiles; refuse duplicate x values rather than average."""
    if {"metric", "value", x} - set(frame):
        raise ValueError("Profile fields are missing")
    selected = frame[(frame.metric == metric) & frame[x].notna()].copy()
    groups = [column for column in series if column in selected and column != x]
    if selected.empty:
        raise ValueError("Profile has no available positions")
    ax = _axes(ax)
    for identity, group in (
        selected.groupby(groups, dropna=False, sort=True) if groups else [(metric, selected)]
    ):
        if group[x].duplicated().any():
            raise ValueError("Profile positions repeat; aggregate explicitly or refine series")
        for column in ("scope", "layer", "component", "embedding_type", "reference_id", "source"):
            if column != x and column in group and group[column].nunique(dropna=False) > 1:
                raise ValueError(f"Profile mixes scientific identities: {column}")
        group = group.sort_values(x, kind="stable")
        ax.plot(group[x], group.value, marker="o", label=str(identity))
    ax.set_xlabel(x)
    ax.set_ylabel(metric)
    ax.legend()
    return ax


def plot_f1_gap(frame, *, metric, ax=None):
    """Display geometry versus absolute F1 gap without fitting an inferential model."""
    selected = _selected(frame, metric)
    if {"abs_f1_gap", "comparison_id"} - set(selected):
        raise ValueError("Use comparison_view to provide performance endpoints")
    if selected.comparison_id.duplicated().any():
        raise ValueError("Select one observation per pair; layers are not independent pairs")
    ax = _axes(ax)
    ax.scatter(selected.value, selected.abs_f1_gap)
    ax.set_xlabel(metric)
    ax.set_ylabel("Absolute strict F1 gap")
    return ax


def plot_reference_panels(frames, *, metric, scale, x="layer"):
    """Display pretraining/baseline panels on common limits without computing contrasts."""
    from matplotlib import pyplot as plt

    if scale.metric != metric or not frames:
        raise ValueError("Provide panels and a common matching scale")
    figure, axes = plt.subplots(1, len(frames), squeeze=False, sharey=True)
    for ax, (name, frame) in zip(axes[0], frames.items(), strict=True):
        plot_geometry_profiles(frame, metric=metric, x=x, ax=ax)
        ax.set_ylim(scale.minimum, scale.maximum)
        ax.set_title(name)
    return figure


def plot_member_f1(frame, *, metric, ax=None):
    """Display member-to-group geometry with declared F1 SD, without aggregation or fit."""
    selected = _selected(frame, metric)
    if {"membership_id", "strict_f1_mean", "strict_f1_std", "group_id"} - set(selected):
        raise ValueError("Use member_performance_view to supply score and group identities")
    if selected.group_id.nunique() != 1 or selected.membership_id.duplicated().any():
        raise ValueError("Choose one group and one observation per member before plotting")
    ax = _axes(ax)
    ax.errorbar(selected.value, selected.strict_f1_mean, yerr=selected.strict_f1_std, fmt="o")
    ax.set_xlabel(metric)
    ax.set_ylabel("Strict F1 mean (declared SD)")
    return ax
