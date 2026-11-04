"""Names of independently executable analyses and their metric families."""

ANALYSES = (
    "pretraining_drift",
    "spectral_fingerprints",
    "cross_language",
    "cross_entity",
    "language_entity",
    "cross_backbone",
    "language_entity_decomposition",
    "model_centrality",
)

OVERLAP_METRICS = ("topk_jaccard", "weighted_jaccard_abs", "topk_sign_agreement")

SUBSPACE_METRICS = ("subspace_mean_cos2", "subspace_chordal_normalized", "subspace_angle_mean_rad")

CROSS_METRICS = tuple(
    f"{metric}_{source}"
    for metric in ("trace_distance", "trace_similarity", "curve_l1", "quantile_w1")
    for source in ("base", "final", "delta")
) + ("adaptation_profile_l1",)

FACTORIAL_METRICS = (
    "ss_language",
    "ss_entity",
    "ss_interaction",
    "ss_total",
    "fraction_language",
    "fraction_entity",
    "fraction_interaction",
)
