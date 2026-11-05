"""Optional plotting functions consuming precomputed result tables."""

from .comparative import (
    ColorScale,
    plot_f1_gap,
    plot_geometry_profiles,
    plot_member_f1,
    plot_pair_heatmap,
    plot_reference_panels,
)
from .profiles import plot_metric

__all__ = [
    "plot_metric",
    "ColorScale",
    "plot_pair_heatmap",
    "plot_geometry_profiles",
    "plot_f1_gap",
    "plot_member_f1",
    "plot_reference_panels",
]
