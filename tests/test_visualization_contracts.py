"""Plots consume result tables and preserve explicit scientific grouping."""

from __future__ import annotations

import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from weight_analysis.results import AnalysisResult
from weight_analysis.visualization import plot_metric


@pytest.fixture
def profile():
    return pd.DataFrame(
        [
            {"checkpoint": "a", "source": "delta", "layer": 2, "delta_l2": 3.0, "level": "tensor"},
            {"checkpoint": "a", "source": "delta", "layer": 0, "delta_l2": 1.0, "level": "tensor"},
            {
                "checkpoint": "a",
                "source": "delta",
                "layer": None,
                "delta_l2": 99.0,
                "level": "encoder",
            },
        ]
    )


def test_plot_sorts_tensor_rows_and_accepts_analysis_result(profile):
    before = profile.copy(deep=True)
    result = AnalysisResult("pretraining_drift", profile, {})
    ax = plot_metric(result, metric="delta_l2")
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == [0.0, 2.0]
    assert list(ax.lines[0].get_ydata()) == [1.0, 3.0]
    assert ax.get_xlabel() == "layer"
    assert ax.get_ylabel() == "delta_l2"
    pd.testing.assert_frame_equal(profile, before)
    plt.close(ax.figure)


def test_plot_does_not_display_or_save_and_reuses_supplied_axes(profile, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("plot_metric must not display or save")

    monkeypatch.setattr(plt, "show", forbidden)
    monkeypatch.setattr(plt, "savefig", forbidden)
    figure, ax = plt.subplots()
    assert plot_metric(profile, metric="delta_l2", ax=ax) is ax
    plt.close(figure)


def test_plot_keeps_checkpoints_in_separate_lines(profile):
    first = profile[profile.level == "tensor"]
    second = first.assign(checkpoint="b", delta_l2=first.delta_l2 * 2)
    ax = plot_metric(pd.concat([second, first]), metric="delta_l2")
    assert len(ax.lines) == 2
    assert list(ax.lines[0].get_ydata()) == [1.0, 3.0]
    assert list(ax.lines[1].get_ydata()) == [2.0, 6.0]
    plt.close(ax.figure)


def test_plot_rejects_duplicate_depth_without_implicit_aggregation(profile):
    doubled = pd.concat([profile, profile.iloc[[0]]])
    with pytest.raises(ValueError, match="Multiple observations"):
        plot_metric(doubled, metric="delta_l2")


def test_plot_rejects_mixed_checkpoints_even_if_depths_differ(profile):
    frame = profile[profile.level == "tensor"].copy()
    frame.loc[0, "checkpoint"] = "b"
    with pytest.raises(ValueError, match="scientific identities"):
        plot_metric(frame, metric="delta_l2", group_by=())


@pytest.mark.parametrize("bad", ["not a number", float("inf")])
def test_plot_rejects_non_numeric_or_infinite_measurements(profile, bad):
    profile["delta_l2"] = profile.delta_l2.astype(object)
    profile.loc[0, "delta_l2"] = bad
    with pytest.raises(ValueError, match="numeric|finite"):
        plot_metric(profile, metric="delta_l2")


def test_plot_missing_columns_and_empty_rows_are_errors(profile):
    with pytest.raises(ValueError, match="missing plot columns"):
        plot_metric(profile, metric="not_a_metric")
    with pytest.raises(ValueError, match="no nonmissing"):
        plot_metric(profile[profile.level == "encoder"], metric="delta_l2")


def test_plot_rejects_infinite_positions(profile):
    profile.loc[0, "layer"] = float("inf")
    with pytest.raises(ValueError, match="finite positions"):
        plot_metric(profile, metric="delta_l2")


def test_plot_rejects_mixed_pair_identities(profile):
    frame = profile[profile.level == "tensor"].drop(columns="checkpoint").copy()
    frame["checkpoint_a"] = ["a", "b"]
    frame["checkpoint_b"] = "c"
    with pytest.raises(ValueError, match="scientific identities"):
        plot_metric(frame, metric="delta_l2")
