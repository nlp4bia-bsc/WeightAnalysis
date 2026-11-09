import json

import pandas as pd
import pyarrow.parquet as pq
import pytest
from test_picasso_experiments import experiment as experiment

from weight_analysis.experiments.compact.finalize import finalize_run
from weight_analysis.experiments.compact.plan import prepare_run
from weight_analysis.experiments.compact.worker import run_task
from weight_analysis.provenance import file_sha256
from weight_analysis_exports.__main__ import STUDIES, export_studies


def test_exports_preserve_all_rows_endpoints_aliases_and_source(experiment, tmp_path):
    source, trained, models, root = experiment
    plan = prepare_run(
        performance=source,
        trained=trained,
        originals=models,
        run_dir=root,
        block_elements=4,
        storage_policy=dict(
            min_free_bytes=0,
            min_free_inodes=0,
            reserve_bytes=0,
            reserve_inodes=0,
            require_quota=False,
        ),
    )
    for task in plan["tasks"]:
        assert run_task(root, task["index"], device="cpu")["state"] == "complete"
    manifest = finalize_run(root)
    original = {r["path"]: file_sha256(root / r["path"]) for r in manifest["files"]}
    output = tmp_path / "exports"
    report = export_studies(root, output, batch_size=23)
    assert set(report["studies"]) == set(STUDIES)
    drift = pd.read_parquet(output / "01_pretraining_drift.parquet")
    assert len(drift) and drift.f1_n_folds_a.eq(4).all()
    assert drift.f1_strict_f1_mean_b.isna().all()
    fingerprints = pd.read_parquet(output / "02_spectral_fingerprints.parquet")
    assert all("alias_unit_id" not in json.loads(value) for value in fingerprints.payload)
    assert (output / "11_model_centrality.parquet").exists()
    for name, metadata in report["studies"].items():
        assert pq.ParquetFile(output / (name + ".parquet")).metadata.num_rows == metadata["rows"]
    assert all(file_sha256(root / name) == digest for name, digest in original.items())
    with pytest.raises(FileExistsError):
        export_studies(root, output)
    with pytest.raises(ValueError):
        export_studies(root, root / "exports")


def test_centrality_missing_edges_do_not_create_complete_ranks(tmp_path):
    from weight_analysis_exports.__main__ import centrality_export

    rows = []
    for a, b, value in (("a", "b", 1.0), ("b", "c", 3.0)):
        rows.append(
            dict(
                artifact_a=a,
                artifact_b=b,
                value=value,
                language="es",
                entity="disease",
                metric="normalized_quantile_w1",
                scope="component",
                source="delta",
                component="q",
                fold_a=1,
                seed_a=None,
            )
        )
    pd.DataFrame(rows).to_parquet(tmp_path / "07_cross_backbone.parquet", index=False)
    links = pd.DataFrame(
        [
            dict(
                artifact_id=x,
                backbone=x,
                language="es",
                entity="disease",
                fold=1,
                seed=None,
                strict_f1_mean=0.8,
                strict_f1_std=0.02,
                n_folds=4,
                score_scope="configuration_aggregate",
            )
            for x in "abc"
        ]
    )
    centrality_export(tmp_path, links)
    result = pd.read_parquet(tmp_path / "11_model_centrality.parquet").set_index("artifact_id")
    assert result.loc["b", "mean_distance"] == 2
    assert result.loc["b", "complete_comparison"]
    assert not result.loc["a", "complete_comparison"]
    assert pd.isna(result.loc["a", "rank_most_central"])
    assert result.n_possible_neighbors.eq(2).all()
