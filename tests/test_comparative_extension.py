"""Independent mathematics and end-to-end contracts for fine-grained weight geometry."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from weight_analysis import (
    CheckpointEvidence,
    CheckpointSpec,
    ParameterSelection,
    ReferenceSpec,
    RunReporter,
    aggregate_geometry,
    analyze_shared_adaptation,
    build_catalog,
    compare_selected,
    comparison_view,
    merge_comparative,
    read_dataset,
    select_axis_pairs,
    select_performance_pairs,
    write_dataset,
)
from weight_analysis.configuration import MetricOptions
from weight_analysis.core import WeightReader
from weight_analysis.core.blocks import iter_blocks
from weight_analysis.core.catalog import input_digest
from weight_analysis.metrics import DRIFT_METRICS, PAIR_METRICS, drift_metrics, pair_metrics
from weight_analysis.metrics.centroids import projection_summary, update_geometry
from weight_analysis.results.models import data_digest
from weight_analysis.statistics.aggregation import VectorMoments


@pytest.fixture
def comparative_study(tmp_path):
    checkpoints, evidence, references = [], {}, {}
    for index, backbone in enumerate(("a", "b")):
        initial = {
            f"bert.encoder.layer.{layer}.attention.self.query.weight": torch.eye(3)
            * (layer + 1 + index * 0.2)
            for layer in range(2)
        }
        initial["bert.embeddings.word_embeddings.weight"] = torch.ones(5, 3)
        initial["bert.encoder.layer.0.output.LayerNorm.weight"] = torch.ones(3)
        base = tmp_path / backbone / "base"
        base.mkdir(parents=True)
        save_file(initial, base / "model.safetensors")
        config = json.dumps({"model_type": "bert", "hidden_size": 3})
        (base / "config.json").write_text(config)
        initialization = input_digest(base)
        references[backbone] = ReferenceSpec(base, tokenizer_id="vocab-1")
        for language, bump in (("es", 0.1), ("en", 0.3)):
            for entity, scale in (("disease", 1), ("symptom", 0.5)):
                for run in range(2):
                    path = tmp_path / backbone / language / entity / str(run)
                    path.mkdir(parents=True)
                    save_file(
                        {k: v + bump * scale * (run + 1) for k, v in initial.items()},
                        path / "model.safetensors",
                    )
                    (path / "config.json").write_text(config)
                    cp = CheckpointSpec(
                        backbone, language, entity, path, seed=7, checkpoint_id=f"run-{run}"
                    )
                    checkpoints.append(cp)
                    evidence[cp.uid] = CheckpointEvidence(
                        initialization,
                        "external-training-manifest",
                        tokenizer_id="vocab-1",
                        training_run_id=f"{backbone}-{language}-{entity}-{run}",
                        configuration_id=f"{backbone}-{run}",
                    )
    catalog = build_catalog(
        checkpoints, evidence, reporter=RunReporter(verbose=False), on_error="raise"
    )
    performance = pd.DataFrame(
        [
            {
                "score_id": e.artifact_id + ":score",
                "artifact_id": e.artifact_id,
                "strict_f1_mean": 0.4
                + (0.1 if e.checkpoint.backbone == "b" else 0)
                + (0.2 if e.checkpoint.checkpoint_id == "run-1" else 0),
                "strict_f1_std": 0.02,
                "score_scope": "checkpoint",
                "protocol_id": "strict-span-micro-v1",
                "evaluation_id": "evaluation-v1",
            }
            for e in catalog.entries
        ]
    )
    return catalog, performance, references, evidence


@pytest.mark.parametrize(
    "strategy,count",
    [("best_worst", 4), ("best_all", 12), ("worst_all", 12), ("top_bottom", 16), ("all_pairs", 24)],
)
def test_selection_schemes_are_deterministic(comparative_study, strategy, count):
    catalog, scores, _, _ = comparative_study
    first = select_performance_pairs(
        checkpoints=catalog, performance=scores, strategy=strategy, k=2
    )
    second = select_performance_pairs(
        checkpoints=replace(catalog, entries=tuple(reversed(catalog.entries))),
        performance=scores.iloc[::-1],
        strategy=strategy,
        k=2,
    )
    assert len(first.pairs) == count
    assert set(first.pairs.comparison_id) == set(second.pairs.comparison_id)
    assert not first.pairs.comparison_id.duplicated().any()
    assert set(first.pairs.interpretation) == {"descriptive_post_selection"}


def test_ties_and_mixed_strata_are_explicit(comparative_study):
    catalog, scores, _, _ = comparative_study
    scores.strict_f1_mean = 0.5
    selected = select_performance_pairs(checkpoints=catalog, performance=scores)
    assert selected.pairs.abs_f1_gap.eq(0).all()
    assert selected.rankings.f1_rank.eq(1).all()
    with pytest.raises(ValueError, match="overlapping"):
        select_performance_pairs(checkpoints=catalog, performance=scores, tie_policy="include_all")
    with pytest.raises(ValueError, match="Mixing"):
        select_performance_pairs(checkpoints=catalog, performance=scores, strata=("language",))


@pytest.mark.parametrize("max_elements", [1, 2, 7, 20])
def test_block_reader_has_exact_coordinates(comparative_study, max_elements):
    catalog, _, _, _ = comparative_study
    with WeightReader(catalog.entries[0].checkpoint.path) as reader:
        for key in sorted(reader.keys):
            blocks = list(iter_blocks(reader, key, max_elements))
            assert all(block.numel() <= max_elements for _, block in blocks)
            assert [start for start, _ in blocks] == list(
                np.cumsum([0] + [v.numel() for _, v in blocks[:-1]])
            )
            torch.testing.assert_close(
                torch.cat([v for _, v in blocks]), reader.get(key).reshape(-1)
            )


def test_stable_moments_match_atomic_metrics_and_merging():
    rng = np.random.default_rng(7)
    a = torch.tensor(rng.normal(size=53), dtype=torch.float64)
    b = torch.tensor(rng.normal(size=53), dtype=torch.float64)
    parts = [VectorMoments.from_tensors(a[i : i + 7], b[i : i + 7]) for i in range(0, len(a), 7)]
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.merge(part)
    for name, value in pair_metrics(a, b, metrics=PAIR_METRICS).items():
        assert combined.value(name) == pytest.approx(value, abs=1e-12, nan_ok=True)
    for name, value in drift_metrics(a, b, metrics=DRIFT_METRICS).items():
        assert combined.value(name) == pytest.approx(value, abs=1e-12, nan_ok=True)
    x = torch.arange(20, dtype=torch.float64) + 1e12
    moments = VectorMoments.from_tensors(x, 2 * x)
    assert moments.value("pearson") == pytest.approx(1)


def test_centroid_projection_residual_and_loo_have_independent_oracles():
    a = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = update_geometry(torch.tensor(a))
    mean = np.mean(a, axis=0)
    np.testing.assert_allclose(result["mean"], mean)
    np.testing.assert_allclose(result["centered_energy"], np.sum((a - mean) ** 2, axis=1))
    np.testing.assert_allclose(
        result["loo_centered_energy"],
        np.array(
            [np.sum((a[i] - np.mean(np.delete(a, i, axis=0), axis=0)) ** 2) for i in range(3)]
        ),
    )
    dispersion = float(result["centered_energy"].mean())
    assert dispersion == pytest.approx(8 / 9)
    assert np.sum(a * a) / 3 == pytest.approx(float(result["mean_energy"]) + dispersion)
    for index in range(3):
        summary = projection_summary(
            float(result["update_energy"][index]),
            float(result["mean_energy"]),
            float(result["dot_mean"][index]),
        )
        projected = np.dot(a[index], mean) / np.dot(mean, mean) * mean
        assert summary["direction_residual_energy"] == pytest.approx(
            np.sum((a[index] - projected) ** 2)
        )
    shifted = update_geometry(torch.tensor(a + [10.0, -7.0]))
    np.testing.assert_allclose(result["centered_energy"], shifted["centered_energy"])
    assert not torch.allclose(result["mean_energy"], shifted["mean_energy"])


def direct_selection(catalog, scores):
    selected = select_performance_pairs(
        checkpoints=catalog, performance=scores, strategy="all_pairs"
    )
    return replace(selected, pairs=selected.pairs[selected.pairs.domain == "direct"].copy())


def test_direct_geometry_aggregation_partition_and_resume(comparative_study, tmp_path, monkeypatch):
    catalog, scores, refs, _ = comparative_study
    selection = direct_selection(catalog, scores)
    args = dict(
        selection=selection,
        references=refs,
        device="cpu",
        parameters=ParameterSelection(components=("q",), embeddings=("token",)),
        metrics=("l2_distance", "cosine"),
        drift_metrics=("relative_l2", "delta_energy"),
        cache_dir=tmp_path / "cache",
        reporter=RunReporter(verbose=False),
        on_error="raise",
        block_elements=4,
    )
    result = compare_selected(**args)
    assert result.complete
    assert not result.tables["geometry_detailed"].empty
    aggregated = aggregate_geometry(result)
    assert set(aggregated.tables["geometry_aggregated"].scope) == {
        "layer_component",
        "layer",
        "component",
        "encoder",
    }
    ids = list(selection.pairs.comparison_id)
    pieces = [
        compare_selected(**{**args, "pair_ids": ids[::2]}),
        compare_selected(**{**args, "pair_ids": ids[1::2]}),
    ]
    merged = merge_comparative(pieces)
    assert data_digest(merged.tables["geometry_detailed"]) == data_digest(
        result.tables["geometry_detailed"]
    )
    reverse = merge_comparative(list(reversed(pieces)))
    assert data_digest(merged.tables["geometry_detailed"]) == data_digest(
        reverse.tables["geometry_detailed"]
    )
    write_dataset(aggregated, tmp_path / "dataset")
    restored = read_dataset(tmp_path / "dataset")
    assert data_digest(restored.tables["geometry_detailed"]) == data_digest(
        result.tables["geometry_detailed"]
    )

    def cannot_read(*args, **kwargs):
        raise AssertionError("Valid cached vector unit was recomputed")

    monkeypatch.setattr(WeightReader, "get", cannot_read)
    resumed = compare_selected(**{**args, "reporter": RunReporter(verbose=False)})
    assert resumed.complete


def test_intrinsic_route_reuses_fingerprints_and_never_directly_subtracts_models(comparative_study):
    catalog, scores, refs, _ = comparative_study
    selection = select_performance_pairs(
        checkpoints=catalog, performance=scores, strategy="best_worst"
    )
    result = compare_selected(
        selection=selection,
        references=refs,
        device="cpu",
        parameters=ParameterSelection(components=("q",), embeddings=("token",)),
        metrics=(),
        intrinsic_metrics=("trace_distance_final", "curve_l1_delta", "quantile_w1_final"),
        options=MetricOptions(min_fingerprint_components=1, spectral_mode="exact"),
        reporter=RunReporter(verbose=False),
        on_error="raise",
    )
    assert result.complete
    view = comparison_view(result)
    assert not view.empty and view.domain.eq("intrinsic").all()
    assert {"strict_f1_mean_a", "strict_f1_std_b", "abs_f1_gap"} <= set(view)


@pytest.mark.parametrize("axis", ["entity", "language"])
def test_shared_adaptation_and_axis_pairs(comparative_study, axis):
    catalog, scores, refs, _ = comparative_study
    members = [e.artifact_id for e in catalog.entries if e.checkpoint.checkpoint_id == "run-0"]
    pairs = select_axis_pairs(
        checkpoints=catalog, axis=axis, member_ids=members, performance=scores
    )
    assert len(pairs.pairs) == 4
    result = analyze_shared_adaptation(
        checkpoints=catalog,
        axis=axis,
        references=refs,
        member_ids=members,
        parameters=ParameterSelection(components=("q",)),
        performance=scores,
        device="cpu",
        reporter=RunReporter(verbose=False),
        on_error="raise",
        block_elements=4,
    )
    assert result.complete and not result.tables["geometry_detailed"].empty
    aggregated = aggregate_geometry(result)
    assert "encoder" in set(aggregated.tables["geometry_aggregated"].scope)


def test_failure_is_recorded_and_remaining_pairs_continue(comparative_study, tmp_path):
    catalog, scores, refs, _ = comparative_study
    selection = direct_selection(catalog, scores)
    broken = next(e for e in catalog.entries if e.checkpoint.backbone == "a")
    (broken.checkpoint.path / "model.safetensors").unlink()
    log = tmp_path / "events.jsonl"
    result = compare_selected(
        selection=selection,
        references=refs,
        device="cpu",
        parameters=ParameterSelection(components=("q",)),
        reporter=RunReporter(verbose=False, log_path=log),
    )
    assert not result.complete
    assert not result.tables["geometry_detailed"].empty
    assert result.tables["issues"].traceback.str.contains("Traceback").any()
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert {"started", "completed", "failed", "summary"} <= set(e["status"] for e in events)
    assert events[-1]["status"] == "summary"


def test_configuration_f1_needs_explicit_mapping_and_keeps_scope(comparative_study):
    catalog, scores, _, _ = comparative_study
    entry = catalog.entries[0]
    record = dict(scores.iloc[0])
    record.update(
        artifact_id=None,
        score_id="aggregate-1",
        score_scope="configuration_aggregate",
        configuration_id=entry.evidence.configuration_id,
        n_folds=2,
        n_seeds=1,
        folds=[entry.checkpoint.fold, 2],
        seeds=[7],
    )
    frame = pd.DataFrame([record])
    unbound = select_performance_pairs(checkpoints=catalog, performance=frame)
    assert unbound.rankings.empty and len(unbound.exclusions) == len(catalog.entries)
    bound = select_performance_pairs(
        checkpoints=catalog, performance=frame, score_alignment={entry.artifact_id: "aggregate-1"}
    )
    assert bound.rankings.score_scope.tolist() == ["configuration_aggregate"]
    other = next(
        e for e in catalog.entries if e.evidence.configuration_id != entry.evidence.configuration_id
    )
    with pytest.raises(ValueError, match="configuration"):
        select_performance_pairs(
            checkpoints=catalog,
            performance=frame,
            score_alignment={other.artifact_id: "aggregate-1"},
        )


def test_reference_shift_preserves_distances_but_changes_directions(comparative_study):
    catalog, scores, refs, _ = comparative_study
    selection = direct_selection(catalog, scores)
    baselines = {}
    for backbone in refs:
        entry = next(e for e in catalog.entries if e.checkpoint.backbone == backbone)
        baselines[backbone] = ReferenceSpec(
            entry.checkpoint.path,
            kind="finetuned_baseline",
            initialization_id=entry.evidence.initialization_id,
            provenance_id="baseline-training-manifest",
            tokenizer_id="vocab-1",
            scope="shared_declared",
        )
    args = dict(
        selection=selection,
        parameters=ParameterSelection(components=("q",)),
        device="cpu",
        metrics=("l2_distance", "cosine"),
        drift_metrics=("delta_energy",),
        reporter=RunReporter(verbose=False),
        on_error="raise",
    )
    original = compare_selected(**args, references=refs)
    shifted = compare_selected(**args, references=baselines)
    left = original.tables["geometry_detailed"].query("metric == 'l2_distance'")
    right = shifted.tables["geometry_detailed"].query("metric == 'l2_distance'")
    assert data_digest(left) == data_digest(right)
    assert left.reference_id.isna().all()
    assert set(original.tables["references"].reference_id).isdisjoint(
        set(shifted.tables["references"].reference_id)
    )
    merged = merge_comparative([original, shifted])
    assert len(merged.tables["geometry_detailed"].query("metric == 'l2_distance'")) == len(left)
    bad = {key: replace(ref, tokenizer_id="another-vocabulary") for key, ref in refs.items()}
    failed = compare_selected(
        **{
            **args,
            "on_error": "continue",
            "parameters": ParameterSelection(embeddings=("token",), components=()),
        },
        references=bad,
    )
    assert not failed.complete


def test_shared_aggregation_matches_concatenated_numpy_and_reference_invariance(comparative_study):
    catalog, _, refs, _ = comparative_study
    members = [
        e
        for e in catalog.entries
        if e.checkpoint.backbone == "a"
        and e.checkpoint.language == "es"
        and e.checkpoint.checkpoint_id == "run-1"
    ]
    args = dict(
        checkpoints=catalog,
        axis="entity",
        member_ids=[e.artifact_id for e in members],
        parameters=ParameterSelection(components=("q",)),
        device="cpu",
        reporter=RunReporter(verbose=False),
        on_error="raise",
        metrics=("mean_energy", "dispersion", "projection_fraction", "distance_to_mean"),
    )
    result = aggregate_geometry(analyze_shared_adaptation(**args, references=refs))
    frame = result.tables["geometry_aggregated"].query("scope == 'encoder'")
    updates = []
    for entry in sorted(members, key=lambda e: e.artifact_id):
        with WeightReader(entry.checkpoint.path) as reader, WeightReader(refs["a"].path) as base:
            keys = sorted(k for k in reader.keys if ".query." in k)
            updates.append(
                np.concatenate(
                    [(reader.get(k) - base.get(k)).double().reshape(-1).numpy() for k in keys]
                )
            )
    updates = np.array(updates)
    mean = updates.mean(0)
    assert frame.query("metric == 'mean_energy'").value.item() == pytest.approx(mean @ mean)
    assert frame.query("metric == 'dispersion'").value.item() == pytest.approx(
        np.sum((updates - mean) ** 2) / len(updates)
    )
    for entry, update in zip(sorted(members, key=lambda e: e.artifact_id), updates):
        value = frame[
            (frame.metric == "projection_fraction") & (frame.member_id == entry.artifact_id)
        ].value.item()
        assert value == pytest.approx((update @ mean) ** 2 / ((update @ update) * (mean @ mean)))
    baseline = ReferenceSpec(
        members[0].checkpoint.path,
        kind="finetuned_baseline",
        initialization_id=members[0].evidence.initialization_id,
        provenance_id="training",
        scope="shared_declared",
    )
    shifted = aggregate_geometry(analyze_shared_adaptation(**args, references={"a": baseline}))
    invariant = ["dispersion", "distance_to_mean"]
    left = frame[frame.metric.isin(invariant)]
    right = shifted.tables["geometry_aggregated"].query("scope == 'encoder'")
    assert data_digest(left) == data_digest(right[right.metric.isin(invariant)])
    assert set(frame.metric) == set(args["metrics"])


def test_relocated_checkpoints_have_the_same_identities(comparative_study, tmp_path):
    import shutil

    catalog, _, _, _ = comparative_study
    entry = catalog.entries[0]
    path = tmp_path / "relocated"
    shutil.copytree(entry.checkpoint.path, path)
    cp = replace(entry.checkpoint, path=path)
    relocated = build_catalog(
        [cp], {cp.uid: entry.evidence}, reporter=RunReporter(verbose=False), on_error="raise"
    )
    assert relocated.entries[0].artifact_id == entry.artifact_id
    assert relocated.entries[0].content_id == entry.content_id


def test_corrupt_cache_recomputes_and_log_write_error_does_not_abort(comparative_study, tmp_path):
    catalog, scores, refs, _ = comparative_study
    selection = direct_selection(catalog, scores)
    args = dict(
        selection=selection,
        pair_ids=selection.pairs.comparison_id.tolist()[:1],
        references=refs,
        device="cpu",
        metrics=("l2_distance",),
        parameters=ParameterSelection(layers=(0,), components=("q",)),
        cache_dir=tmp_path / "cache",
        on_error="raise",
    )
    first = compare_selected(**args, reporter=RunReporter(verbose=False))
    cache = next((tmp_path / "cache").glob("*.parquet"))
    cache.write_bytes(b"broken")
    unavailable_log = tmp_path / "not-a-directory"
    unavailable_log.write_text("file")
    reporter = RunReporter(verbose=False, log_path=unavailable_log / "events.jsonl")
    resumed = compare_selected(**args, reporter=reporter)
    assert resumed.complete
    assert data_digest(first.tables["geometry_detailed"]) == data_digest(
        resumed.tables["geometry_detailed"]
    )
    assert "warning" in {event["status"] for event in reporter.events}


def test_prints_summary_and_does_not_swallow_interrupt(capsys):
    reporter = RunReporter()
    assert reporter.attempt("unit", lambda: 7) == 7
    reporter.summary()
    assert "[COMPLETED]" in capsys.readouterr().out

    def interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        reporter.attempt("interrupt", interrupt)


def test_comparative_plots_refuse_implicit_aggregation():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from weight_analysis.visualization import (
        ColorScale,
        plot_f1_gap,
        plot_geometry_profiles,
        plot_pair_heatmap,
        plot_reference_panels,
    )

    frame = pd.DataFrame(
        [
            dict(
                metric="rmse",
                value=0.3,
                artifact_a="a",
                artifact_b="b",
                entity_a="disease",
                entity_b="symptom",
                language_a="es",
                language_b="en",
                abs_f1_gap=0.2,
                comparison_id="p",
                subject_id="p",
                scope="encoder",
            )
        ]
    )
    for axis in ("artifact", "entity", "language"):
        assert plot_pair_heatmap(frame, metric="rmse", axis=axis, scale=ColorScale("rmse", 0, 1))
    assert plot_f1_gap(frame, metric="rmse")
    with pytest.raises(ValueError, match="Duplicate"):
        plot_pair_heatmap(pd.concat([frame, frame]), metric="rmse")
    profile = pd.DataFrame(
        [
            dict(metric="rmse", value=0.2, layer=0, subject_id="p"),
            dict(metric="rmse", value=0.3, layer=1, subject_id="p"),
        ]
    )
    assert plot_geometry_profiles(profile, metric="rmse")
    assert plot_reference_panels(
        {"pretraining": profile, "baseline": profile}, metric="rmse", scale=ColorScale("rmse", 0, 1)
    )
    with pytest.raises(ValueError, match="repeat"):
        plot_geometry_profiles(pd.concat([profile, profile]), metric="rmse")
    plt.close("all")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.cuda.is_available() is False")
@pytest.mark.parametrize("workflow", ["direct", "intrinsic", "entity", "language"])
def test_new_workflows_cuda_parity(comparative_study, workflow):
    catalog, scores, refs, _ = comparative_study
    args = dict(
        parameters=ParameterSelection(components=("q",), embeddings=("token",)),
        reporter=RunReporter(verbose=False),
        on_error="raise",
        references=refs,
        block_elements=4,
    )
    if workflow in ("direct", "intrinsic"):
        selection = (
            direct_selection(catalog, scores)
            if workflow == "direct"
            else select_performance_pairs(checkpoints=catalog, performance=scores)
        )
        args.update(
            selection=selection,
            metrics=("l2_distance", "cosine"),
            spectral_metrics=("matrix_fro", "stable_rank"),
            intrinsic_metrics=("curve_l1_delta", "quantile_w1_final", "adaptation_profile_l1"),
            options=MetricOptions(min_fingerprint_components=1, spectral_mode="exact"),
        )
        function = compare_selected
    else:
        args.update(
            checkpoints=catalog,
            axis=workflow,
            member_ids=[
                e.artifact_id for e in catalog.entries if e.checkpoint.checkpoint_id == "run-0"
            ],
        )
        function = analyze_shared_adaptation
    cpu = aggregate_geometry(function(**args, device="cpu"))
    cuda = aggregate_geometry(function(**args, device="gpu"))
    for table in ("geometry_detailed", "geometry_aggregated"):
        fields = [
            name
            for name in (
                "subject_id",
                "metric",
                "source",
                "reference_id",
                "scope",
                "tensor",
                "layer",
                "component",
                "embedding_type",
                "relative_depth",
            )
            if name in cpu.tables[table]
        ]
        left = cpu.tables[table].sort_values(fields).reset_index(drop=True)
        right = cuda.tables[table].sort_values(fields).reset_index(drop=True)
        pd.testing.assert_frame_equal(left[fields], right[fields])
        np.testing.assert_allclose(
            left.value.astype(float),
            right.value.astype(float),
            atol=1e-7,
            rtol=1e-6,
            equal_nan=True,
        )
    updates = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], device="cuda")
    assert all(value.is_cuda for value in update_geometry(updates).values())


def test_parameter_fragments_and_separate_metric_requests_merge_exactly(comparative_study):
    catalog, scores, refs, _ = comparative_study
    selection = direct_selection(catalog, scores)
    args = dict(
        selection=selection,
        pair_ids=selection.pairs.comparison_id.tolist()[:1],
        references=refs,
        device="cpu",
        metrics=("l2_distance", "cosine"),
        sources=("final",),
        reporter=RunReporter(verbose=False),
        on_error="raise",
    )
    joint = aggregate_geometry(
        compare_selected(**args, parameters=ParameterSelection(components=("q",)))
    )
    pieces = [
        compare_selected(**args, parameters=ParameterSelection(components=("q",), layers=(layer,)))
        for layer in (1, 0)
    ]
    combined = aggregate_geometry(merge_comparative(pieces))
    for table in ("geometry_detailed", "geometry_aggregated", "sufficient_statistics"):
        assert data_digest(joint.tables[table]) == data_digest(combined.tables[table])
    pieces = [
        compare_selected(
            **{**args, "metrics": (metric,)}, parameters=ParameterSelection(components=("q",))
        )
        for metric in ("l2_distance", "cosine")
    ]
    combined = aggregate_geometry(merge_comparative(pieces))
    assert data_digest(joint.tables["geometry_aggregated"]) == data_digest(
        combined.tables["geometry_aggregated"]
    )


def test_unknown_selection_and_same_shape_different_architecture_are_rejected(comparative_study):
    from weight_analysis.core.catalog import parameter_space

    catalog, scores, refs, _ = comparative_study
    entry = catalog.entries[0]
    old = parameter_space(entry.checkpoint.path)
    config = entry.checkpoint.path / "config.json"
    value = json.loads(config.read_text())
    value["num_attention_heads"] = 3
    config.write_text(json.dumps(value))
    assert parameter_space(entry.checkpoint.path) != old
    # The catalog hash also prevents use of a modified config after freezing selection.
    result = compare_selected(
        selection=direct_selection(catalog, scores),
        references=refs,
        device="cpu",
        reporter=RunReporter(verbose=False),
        parameters=ParameterSelection(layers=(999,)),
    )
    assert not result.complete
    assert result.tables.get("geometry_detailed", pd.DataFrame()).empty


def test_f1_associations_count_shared_checkpoints_without_iid_inference(comparative_study):
    from weight_analysis import associate_f1, member_performance_view

    catalog, scores, refs, _ = comparative_study
    selection = select_performance_pairs(
        checkpoints=catalog, performance=scores, strategy="all_pairs"
    )
    args = dict(
        selection=selection,
        references=refs,
        device="cpu",
        parameters=ParameterSelection(components=("q",)),
        metrics=("l2_distance",),
        intrinsic_metrics=("quantile_w1_final",),
        reporter=RunReporter(verbose=False),
        on_error="raise",
        options=MetricOptions(min_fingerprint_components=1),
    )
    result = aggregate_geometry(compare_selected(**args))
    associated = associate_f1(result, table="geometry_aggregated")
    table = associated.tables["associations"]
    assert not table.empty
    assert table.interpretation.eq("descriptive_post_selection_no_iid_inference").all()
    assert not any("pvalue" in column or "p_value" in column for column in table)
    assert (table.n_pairs > 0).all() and (table.n_checkpoints >= 2).all()
    assert set(table.note) == {
        "Shared checkpoints and aggregated F1 are not independent replicates"
    }
    members = [e.artifact_id for e in catalog.entries if e.checkpoint.checkpoint_id == "run-0"]
    shared = analyze_shared_adaptation(
        checkpoints=catalog,
        axis="language",
        references=refs,
        member_ids=members,
        performance=scores,
        device="cpu",
        parameters=args["parameters"],
        reporter=RunReporter(verbose=False),
        on_error="raise",
    )
    view = member_performance_view(aggregate_geometry(shared), table="geometry_aggregated")
    assert not view.empty and view.strict_f1_mean.notna().all()


def test_spectral_dependencies_are_reused_between_intrinsic_and_descriptions(
    comparative_study, monkeypatch
):
    import weight_analysis.execution.comparative_units as units

    catalog, scores, refs, _ = comparative_study
    selected = select_performance_pairs(checkpoints=catalog, performance=scores)
    real = units.spectral_metrics
    calls = []

    def traced(matrix, **kwargs):
        calls.append(tuple(kwargs["metrics"]))
        return real(matrix, **kwargs)

    monkeypatch.setattr(units, "spectral_metrics", traced)
    result = compare_selected(
        selection=selected,
        references=refs,
        device="cpu",
        parameters=ParameterSelection(components=("q",)),
        metrics=(),
        spectral_metrics=("stable_rank",),
        sources=("delta",),
        intrinsic_metrics=("trace_distance_delta", "curve_l1_delta"),
        options=MetricOptions(min_fingerprint_components=1),
        reporter=RunReporter(verbose=False),
        on_error="raise",
    )
    assert result.complete
    endpoints = set(selected.pairs.artifact_a) | set(selected.pairs.artifact_b)
    assert len(calls) == 2 * len(endpoints)  # two matrices per endpoint, once per source
    assert all(
        set(names) == {"matrix_fro", "stable_rank", "spectral_energy_curve"} for names in calls
    )


@pytest.mark.parametrize("n,width", [(2, 1), (2, 7), (3, 11), (7, 19)])
def test_shared_block_products_against_random_noncollinear_oracle(n, width):
    from weight_analysis.analyses.shared_adaptation import SHARED_METRICS, finalize_shared

    updates = np.random.default_rng(42 + n + width).normal(size=(n, width))
    mean = updates.mean(0)
    records = []
    for block, start in enumerate(range(0, width, 3)):
        x = updates[:, start : start + 3]
        mu = x.mean(0)
        for i in range(n):
            loo = np.delete(x, i, axis=0).mean(0)
            records.append(
                dict(
                    member_id=f"m{i}",
                    tensor=f"t{block}",
                    layer=block,
                    component="q",
                    embedding_type=None,
                    group_id="g",
                    membership_id=f"g-m{i}",
                    n_members=n,
                    n_parameters=x.shape[1],
                    reference_id="r",
                    execution_id="e",
                    axis="entity",
                    update_energy=float(x[i] @ x[i]),
                    mean_energy=float(mu @ mu),
                    dot_mean=float(x[i] @ mu),
                    centered_energy=float(np.sum((x[i] - mu) ** 2)),
                    loo_mean_energy=float(loo @ loo),
                    loo_dot=float(x[i] @ loo),
                    loo_centered_energy=float(np.sum((x[i] - loo) ** 2)),
                )
            )
    rows = pd.DataFrame(finalize_shared(records[::-1], metrics=SHARED_METRICS, scope="encoder"))
    assert rows.query("metric == 'mean_norm'").value.item() == pytest.approx(np.linalg.norm(mean))
    assert rows.query("metric == 'dispersion'").value.item() == pytest.approx(
        np.mean(np.sum((updates - mean) ** 2, axis=1))
    )
    for i, update in enumerate(updates):
        member = rows[rows.member_id == f"m{i}"].set_index("metric").value
        for prefix, direction in (("", mean), ("loo_", np.delete(updates, i, axis=0).mean(0))):
            projected = np.dot(update, direction) * direction / np.dot(direction, direction)
            assert member[prefix + "projection_fraction"] == pytest.approx(
                np.dot(projected, projected) / np.dot(update, update)
            )
            assert member[prefix + "direction_residual_energy"] == pytest.approx(
                np.sum((update - projected) ** 2), abs=1e-12
            )
            assert member[prefix + "alignment_cosine"] == pytest.approx(
                np.dot(update, direction) / (np.linalg.norm(update) * np.linalg.norm(direction))
            )
        assert member["distance_to_mean"] == pytest.approx(np.linalg.norm(update - mean))
        assert member["loo_distance_to_mean"] == pytest.approx(
            np.linalg.norm(update - np.delete(updates, i, axis=0).mean(0))
        )


def test_degenerate_geometries_remain_explicitly_undefined():
    geometry = update_geometry(torch.tensor([[1.0, 2.0], [-1.0, -2.0]]))
    assert geometry["mean_energy"] == 0
    values = projection_summary(5, 0, 0)
    assert all(np.isnan(value) for value in values.values())
    assert all(np.isnan(value) for value in projection_summary(0, 2, 0).values())
    with pytest.raises(ValueError, match="Projection exceeds"):
        projection_summary(1, 1, 2)
    with pytest.raises(ValueError, match="Negative"):
        projection_summary(-1, 1, 0)
    for invalid in (torch.ones(1, 2), torch.ones(2, 0), torch.tensor([[np.nan], [1.0]])):
        with pytest.raises(ValueError):
            update_geometry(invalid)


def test_adaptation_profile_device_path_preserves_historical_formula():
    from types import SimpleNamespace

    from weight_analysis.comparisons.backbones import compare_backbones

    checkpoints = [
        SimpleNamespace(uid=uid, path=".", backbone=uid, language="es", entity="disease")
        for uid in ("a", "b")
    ]
    spectra = pd.DataFrame(columns=["checkpoint", "component", "layer", "source"])
    drift = pd.DataFrame(
        [
            dict(checkpoint=uid, component="q", layer=layer, level="tensor", relative_l2=value)
            for uid, values in (("a", [0.1, 0.2]), ("b", [0.4, 0.7]))
            for layer, value in enumerate(values)
        ]
    )
    historical = compare_backbones(*checkpoints, spectra, drift, min_components=1)
    explicit = compare_backbones(*checkpoints, spectra, drift, min_components=1, device="cpu")
    axis = np.linspace(0, 1, 16)
    expected = np.abs(
        np.interp(axis, [0, 1], np.log1p([0.1, 0.2]))
        - np.interp(axis, [0, 1], np.log1p([0.4, 0.7]))
    ).mean()
    assert historical["adaptation_profile_l1"] == pytest.approx(expected)
    assert explicit["adaptation_profile_l1"] == pytest.approx(expected)


def test_adaptation_only_intrinsic_has_component_results(comparative_study):
    catalog, scores, refs, _ = comparative_study
    result = compare_selected(
        selection=select_performance_pairs(checkpoints=catalog, performance=scores),
        references=refs,
        device="cpu",
        metrics=(),
        parameters=ParameterSelection(components=("q",)),
        intrinsic_metrics=("adaptation_profile_l1",),
        options=MetricOptions(min_fingerprint_components=1),
        reporter=RunReporter(verbose=False),
        on_error="raise",
    )
    assert result.complete
    view = comparison_view(result)
    assert set(view.scope) == {"encoder", "component"}
    assert view.value.notna().all()


def test_member_f1_association_and_plot_keep_group_identity(comparative_study):
    from matplotlib import pyplot as plt

    from weight_analysis import associate_member_f1, member_performance_view
    from weight_analysis.visualization import plot_member_f1

    catalog, scores, refs, _ = comparative_study
    members = [e.artifact_id for e in catalog.entries if e.checkpoint.checkpoint_id == "run-0"]
    result = aggregate_geometry(
        analyze_shared_adaptation(
            checkpoints=catalog,
            axis="language",
            member_ids=members,
            references=refs,
            performance=scores,
            device="cpu",
            parameters=ParameterSelection(components=("q",)),
            reporter=RunReporter(verbose=False),
            on_error="raise",
        )
    )
    associated = associate_member_f1(result, table="geometry_aggregated")
    assert set(associated.tables["associations"].interpretation) == {
        "descriptive_geometry_no_iid_inference"
    }
    view = member_performance_view(result, table="geometry_aggregated")
    view = view[(view.scope == "encoder") & (view.metric == "distance_to_mean")]
    group = view[view.group_id == view.group_id.iloc[0]]
    assert plot_member_f1(group, metric="distance_to_mean")
    with pytest.raises(ValueError, match="one group"):
        plot_member_f1(view, metric="distance_to_mean")
    plt.close("all")


def test_pooling_requires_a_recorded_experimental_policy(comparative_study):
    catalog, scores, _, _ = comparative_study
    with pytest.raises(ValueError, match="experimental_policy"):
        select_performance_pairs(
            checkpoints=catalog, performance=scores, strata=("language",), allow_mixed_strata=True
        )
    result = select_performance_pairs(
        checkpoints=catalog,
        performance=scores,
        strata=("language",),
        allow_mixed_strata=True,
        experimental_policy="Exploratory pooled entity comparison; no confirmatory inference",
    )
    assert len(result.pairs) == 2
    assert result.provenance["experimental_policy"].startswith("Exploratory")
