"""Real installed CLI calls remain thin, selective, and independent of the checkout cwd."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from weight_analysis import CheckpointSpec, analyze, read_result

QUERY = "bert.encoder.layer.0.attention.self.query.weight"
ANALYSIS_NAMES = {
    "pretraining_drift",
    "spectral_fingerprints",
    "cross_language",
    "cross_entity",
    "language_entity",
    "cross_backbone",
    "language_entity_decomposition",
    "model_centrality",
}


def installed_environment():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    environment["OPENBLAS_NUM_THREADS"] = "2"

    return environment


def run_installed_cli(arguments, directory):
    return subprocess.run(
        [sys.executable, "-m", "weight_analysis", *arguments],
        cwd=directory,
        env=installed_environment(),
        capture_output=True,
        text=True,
        check=True,
    )


def test_installed_analysis_listing_works_outside_the_checkout(tmp_path):
    completed = run_installed_cli(["analyses"], tmp_path)

    assert set(completed.stdout.splitlines()) == ANALYSIS_NAMES
    assert len(completed.stdout.splitlines()) == len(ANALYSIS_NAMES)
    assert not list(tmp_path.iterdir())


def test_cli_listing_never_imports_tensor_or_plotting_libraries(tmp_path):
    code = """
import importlib.abc
import runpy
import sys

forbidden = {"torch", "numpy", "pandas", "scipy", "pyarrow", "matplotlib", "lab", "ner_lab"}
class RejectComputeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in forbidden:
            raise AssertionError("Listing analyses imported " + fullname)
        return None
sys.meta_path.insert(0, RejectComputeImports())
sys.argv = ["weight_analysis", "analyses"]
runpy.run_module("weight_analysis", run_name="__main__", alter_sys=True)
assert not forbidden.intersection(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=installed_environment(),
        capture_output=True,
        text=True,
        check=True,
    )

    assert set(completed.stdout.splitlines()) == ANALYSIS_NAMES


@pytest.fixture
def cli_study(tmp_path):
    original_path = tmp_path / "original"
    trained_path = tmp_path / "trained"
    original_path.mkdir()
    trained_path.mkdir()
    initial = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    final = initial + 0.25
    save_file({QUERY: initial}, original_path / "model.safetensors")
    save_file(
        {QUERY: final, "classifier.weight": torch.ones(3, 2)}, trained_path / "model.safetensors"
    )
    for directory in (original_path, trained_path):
        (directory / "config.json").write_text(json.dumps({"model_type": "bert"}))
    checkpoint = CheckpointSpec("tiny", "xx", "custom", trained_path, fold=2, seed=17)

    return checkpoint, original_path


def test_cli_json_run_matches_python_api_and_writes_only_requested_parquet(cli_study, tmp_path):
    checkpoint, original_path = cli_study
    working_dir = tmp_path / "independent-cwd"
    working_dir.mkdir()
    output_path = tmp_path / "artifacts" / "requested.parquet"
    configuration = {
        "analysis": "pretraining_drift",
        "checkpoints": [
            {
                "backbone": checkpoint.backbone,
                "language": checkpoint.language,
                "entity": checkpoint.entity,
                "path": str(checkpoint.path),
                "fold": checkpoint.fold,
                "seed": checkpoint.seed,
            }
        ],
        "originals": {"tiny": str(original_path)},
        "metrics": ["relative_frobenius"],
        "tensors": [QUERY],
        "output": str(output_path),
    }
    configuration_path = tmp_path / "compute.json"
    configuration_path.write_text(json.dumps(configuration))
    expected = analyze(
        analysis="pretraining_drift",
        checkpoints=[checkpoint],
        originals={"tiny": original_path},
        metrics=["relative_frobenius"],
        tensors=[QUERY],
    )
    before = set(tmp_path.rglob("*"))

    completed = run_installed_cli(["run", str(configuration_path)], working_dir)
    report = json.loads(completed.stdout)
    loaded = read_result(output_path)

    assert report == {
        "analysis": "pretraining_drift",
        "rows": 1,
        "unit_id": expected.unit_id,
        "output": str(output_path),
    }
    pd.testing.assert_frame_equal(loaded.data, expected.data, check_dtype=False)
    assert loaded.provenance == expected.provenance
    assert loaded.data.iloc[0].relative_frobenius == pytest.approx(0.5 / np.sqrt(5))
    assert set(tmp_path.rglob("*")) - before == {output_path.parent, output_path}
    assert not list(working_dir.iterdir())
    assert loaded.data.iloc[0].trained_path == str(checkpoint.path)
    assert loaded.provenance["inputs"][0]["path"] == str(checkpoint.path)
    assert loaded.provenance["originals"]["tiny"]["path"] == str(original_path)


def test_cli_without_output_returns_summary_without_implicit_files(cli_study, tmp_path):
    checkpoint, _ = cli_study
    working_dir = tmp_path / "no-output-cwd"
    working_dir.mkdir()
    configuration_path = tmp_path / "in-memory.json"
    configuration_path.write_text(
        json.dumps(
            {
                "analysis": "spectral_fingerprints",
                "checkpoints": [
                    {
                        "backbone": checkpoint.backbone,
                        "language": checkpoint.language,
                        "entity": checkpoint.entity,
                        "path": str(checkpoint.path),
                    }
                ],
                "metrics": ["matrix_fro"],
                "sources": ["final"],
                "tensors": [QUERY],
            }
        )
    )
    before = set(tmp_path.rglob("*"))

    completed = run_installed_cli(["run", str(configuration_path)], working_dir)
    report = json.loads(completed.stdout)

    assert report["analysis"] == "spectral_fingerprints"
    assert report["rows"] == 1
    assert report["unit_id"]
    assert report["output"] is None
    assert set(tmp_path.rglob("*")) == before
