"""Preserved maintenance/report contracts and real wheel installation outside the checkout."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _legacy_report_fixture import report_inputs

from weight_analysis.maintenance.review import export_review
from weight_analysis.maintenance.snapshot import create_snapshot
from weight_analysis.maintenance.testing import build_bundle, installed_environment

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_code_and_compatibility_packages_live_under_src():
    assert not list(ROOT.glob("*.py"))
    assert not (ROOT / "weightspace").exists()
    assert not (ROOT / "validacion_f1_weightspace").exists()
    import weightspace

    assert Path(weightspace.__file__).is_relative_to(ROOT / "src")
    for name in (
        "analyses",
        "checkpoints",
        "metrics",
        "parallel",
        "storage",
        "performance",
        "slurm_guard",
        "recovery",
    ):
        assert importlib.import_module("weightspace." + name)


def test_report_imports_have_no_io_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for module in (
        "correlation_summary",
        "validation",
        "data_audit",
        "layer_correlations",
        "backbone_sensitivity",
    ):
        importlib.reload(importlib.import_module("weightspace.reports." + module))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "report",
    [
        "correlation_summary",
        "validation",
        "data_audit",
        "layer_correlations",
        "backbone_sensitivity",
    ],
)
def test_historical_reports_match_pre_cleanup_numeric_oracles(tmp_path, report):
    if report in ("validation", "backbone_sensitivity"):
        pytest.importorskip("statsmodels")
    data, performance = report_inputs(tmp_path)
    expected = json.loads((ROOT / "tests/fixtures/legacy_reports.json").read_text())["reports"][
        report
    ]
    output = tmp_path / "output"
    if report == "backbone_sensitivity":
        from weightspace.reports.validation import run

        run(data_dir=data, performance=performance, output_dir=output, permutations=7)
    module = importlib.import_module("weightspace.reports." + report)
    module.run(data_dir=data, performance=performance, output_dir=output, permutations=7)
    for name, table in expected.items():
        found = pd.read_csv(output / name)
        assert list(found.columns) == table["columns"]
        baseline = pd.DataFrame(table["data"], columns=table["columns"])
        assert len(found) == len(baseline)
        for column in found:
            if pd.api.types.is_numeric_dtype(found[column]):
                np.testing.assert_allclose(
                    found[column],
                    baseline[column].astype(float),
                    rtol=1e-10,
                    atol=1e-12,
                    equal_nan=True,
                )
            else:
                assert (
                    found[column].fillna("<null>").tolist()
                    == baseline[column].fillna("<null>").tolist()
                )


def test_review_export_streams_nested_parquets_and_explicit_legacy_policy(tmp_path):
    import gzip
    import zipfile

    source = tmp_path / "inputs"
    (source / "nested").mkdir(parents=True)
    frame = pd.DataFrame(
        dict(level=["encoder", "tensor"], value=[1.0, 2.0], curve=[[0.0, 1.0], [0.0, 1.0]])
    )
    frame.to_parquet(source / "03_pairs.parquet", index=False)
    frame.to_parquet(source / "nested/03_pairs.parquet", index=False)
    destination = tmp_path / "review"
    archive = export_review(source, destination, legacy_encoder_pairs=True, batch_size=1)
    with gzip.open(destination / "03_pairs.csv.gz", "rt") as stream:
        found = pd.read_csv(stream)
    assert found.level.tolist() == ["encoder"] and "curve" not in found
    assert (destination / "nested/03_pairs.csv.gz").is_file()
    assert "inventory.json" in zipfile.ZipFile(archive).namelist()
    with pytest.raises(FileExistsError):
        export_review(source, destination)
    with pytest.raises(ValueError):
        export_review(source, source / "nested/output")


def test_snapshot_selection_survives_new_live_model_folders(tmp_path):
    import torch
    from safetensors.torch import save_file

    live = tmp_path / "weights"
    checkpoint = live / "es/disease/model"
    checkpoint.mkdir(parents=True)
    save_file(
        {"bert.embeddings.word_embeddings.weight": torch.ones(2, 2)},
        checkpoint / "model.safetensors",
    )
    (checkpoint / "config.json").write_text('{"model_type":"bert"}')
    snapshot = tmp_path / "snapshot"
    assert create_snapshot(live, snapshot, set(), 1) == 1
    assert (snapshot / "es/disease/model").is_symlink()
    (live / "es/disease/new-model").mkdir()
    assert sorted(p.name for p in (snapshot / "es/disease").iterdir()) == ["model"]


def test_wheel_runs_cli_without_pythonpath_and_preserves_module_dependencies(tmp_path, monkeypatch):
    import os
    import site
    import sys

    bundle = (
        Path(os.environ["WEIGHT_ANALYSIS_TEST_BUNDLE"])
        if "WEIGHT_ANALYSIS_TEST_BUNDLE" in os.environ
        else tmp_path / "bundle"
    )
    if "WEIGHT_ANALYSIS_TEST_BUNDLE" not in os.environ:
        pytest.importorskip(
            "hatchling",
            reason="Local wheel build requires the build backend; Picasso uses its prebuilt bundle",
        )
        build_bundle(ROOT, bundle, build_isolation=False)
    # Reproduce a module that overrides a global installation: active sys.path
    # precedence must win over getsitepackages(), including in nested venvs.
    active = tmp_path / "module" / "site-packages"
    fallback = tmp_path / "global" / "site-packages"
    for path, value in ((active, "module"), (fallback, "global")):
        path.mkdir(parents=True)
        (path / "wa_dependency_priority_probe.py").write_text(f"VALUE = {value!r}\n")
    original_sites = site.getsitepackages()
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(fallback), *original_sites])
    monkeypatch.setattr(sys, "path", [str(active), str(fallback), *sys.path])
    python, environment = installed_environment(bundle, ROOT, tmp_path / "installed")
    assert "PYTHONPATH" not in environment
    working = tmp_path / "outside"
    working.mkdir()
    completed = subprocess.run(
        [str(python), "-m", "weight_analysis", "analyses"],
        cwd=working,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "pretraining_drift" in completed.stdout
    subprocess.run(
        [
            str(python),
            "-c",
            "import torch; import weightspace.metrics; import importlib.metadata; import wa_dependency_priority_probe as probe; assert probe.VALUE == 'module'; print(torch.__version__); assert importlib.metadata.version('weight-analysis')",
        ],
        cwd=working,
        env=environment,
        check=True,
    )
    import shutil

    private_bundle = tmp_path / "tampered-bundle"
    shutil.copytree(bundle, private_bundle)
    manifest = private_bundle / "bundle.json"
    altered = json.loads(manifest.read_text())
    altered["wheel_sha256"] = "bad"
    manifest.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="wheel changed"):
        installed_environment(private_bundle, ROOT, tmp_path / "rejected")
