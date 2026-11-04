"""Device routing and mathematical parity; real CUDA checks skip without hardware."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from weight_analysis import CheckpointSpec, analyze
from weight_analysis.configuration import MetricOptions
from weight_analysis.devices import resolve_device
from weight_analysis.metrics import (
    DRIFT_METRICS,
    PAIR_METRICS,
    SPECTRAL_METRICS,
    drift_metrics,
    pair_metrics,
    paper_trace_similarity,
    principal_angles,
    principal_basis,
    quantile_w1,
    spectral_metrics,
    topk_overlap,
)
from weight_analysis.metrics.decomposition import (
    entry_quantiles,
    interpolate_positions,
    truncated_singular_values,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA GPU unavailable")


@pytest.fixture
def study(tmp_path):
    originals, checkpoints = {}, []
    for index, backbone in enumerate(("a", "b")):
        base = tmp_path / backbone / "base"
        base.mkdir(parents=True)
        initial = {
            f"bert.encoder.layer.{layer}.attention.self.query.weight": torch.diag(
                torch.tensor([4.0, 3.0, 2.0, 1.0])
            )
            * (layer + 1 + index * 0.2)
            for layer in range(3)
        }
        initial["bert.embeddings.word_embeddings.weight"] = torch.ones(8, 4)
        save_file(initial, base / "model.safetensors")
        (base / "config.json").write_text(json.dumps({"model_type": "bert"}))
        originals[backbone] = base
        for language, bump in (("es", 0.1), ("en", 0.3)):
            for entity, scale in (("disease", 1), ("symptom", -1)):
                path = tmp_path / backbone / language / entity
                path.mkdir(parents=True)
                save_file(
                    {key: value + bump * scale for key, value in initial.items()},
                    path / "model.safetensors",
                )
                (path / "config.json").write_text(json.dumps({"model_type": "bert"}))
                checkpoints.append(CheckpointSpec(backbone, language, entity, path))
    return checkpoints, originals


def test_gpu_default_and_no_silent_fallback(study, monkeypatch):
    assert MetricOptions().device == "gpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoints, originals = study
    with pytest.raises(RuntimeError, match="device='cpu'"):
        analyze(analysis="pretraining_drift", checkpoints=checkpoints, originals=originals)
    result = analyze(
        device="cpu", analysis="pretraining_drift", checkpoints=checkpoints, originals=originals
    )
    assert result.provenance["configuration"]["options"]["device"] == "cpu"


@pytest.mark.parametrize("name", ["gpu", "GPU", "cuda", "gpu:1", "cuda:1"])
def test_gpu_aliases_are_canonical(name, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    expected = "cuda:1" if name.endswith(":1") else "cuda:0"
    assert str(resolve_device(name)) == expected


@pytest.mark.parametrize("name", ["tpu", "mps", "cpu:1", "cuda:-1", "gpu:99"])
def test_invalid_devices_rejected(name, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError):
        resolve_device(name)


def test_torch_approximation_matches_independent_exact_svd_and_preserves_rng():
    x = torch.diag(torch.logspace(0, -6, 80, dtype=torch.float64))
    before = torch.random.get_rng_state().clone()
    got = truncated_singular_values(x, 4)
    expected = torch.linalg.svdvals(x)[:4]
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-9)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(got, truncated_singular_values(x, 4))


def test_device_profile_and_overlap_backends_match_numpy_oracles():
    a, b = torch.tensor([0.0, -4.0, 4.0, 2.0]), torch.tensor([0.0, 3.0, -4.0, 1.0])
    assert topk_overlap(a, b, 0.5, device="cpu") == topk_overlap(a, b, 0.5)
    x, y = {"q": [1.0, 4.0, 2.0]}, {"q": [3.0, 1.0, 2.0, 5.0]}
    assert paper_trace_similarity(x, y, device="cpu")["trace_similarity"] == pytest.approx(
        paper_trace_similarity(x, y)["trace_similarity"], abs=1e-14
    )
    assert quantile_w1([0.0, 1.0, 2.0], [0.0, 2.0, 4.0], device="cpu") == 1.0
    values = torch.tensor([-4.0, 2.0, 1.0, 8.0], dtype=torch.float64)
    np.testing.assert_allclose(
        entry_quantiles(values, 21), np.quantile(values, np.linspace(0, 1, 21))
    )
    positions = torch.tensor([0.1, 0.7, 1.0], dtype=torch.float64)
    query = torch.linspace(0, 1, 19, dtype=torch.float64)
    actual = interpolate_positions(positions, values[:3], query)
    np.testing.assert_allclose(actual, np.interp(query, positions, values[:3]), atol=1e-14)


@pytest.mark.parametrize("fraction", [0.01, 0.5, 1.0])
def test_partial_topk_preserves_sparse_support_and_index_ties(fraction):
    a = torch.tensor([0.0, 3.0, -3.0, 3.0, 0.0, 2.0])
    b = torch.tensor([0.0, -3.0, 2.0, 3.0, 0.0, 3.0])
    assert topk_overlap(a, b, fraction, device="cpu") == pytest.approx(
        topk_overlap(a, b, fraction), nan_ok=True
    )


@pytest.mark.parametrize("zeros", [False, True])
@pytest.mark.parametrize("family", ["drift", "pair"])
def test_batched_device_vector_backend_matches_reference(family, zeros):
    from weight_analysis.metrics.device_vectors import drift_on_device, pairs_on_device

    x = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float64)
    y = torch.zeros_like(x) if zeros else x + 0.25
    if family == "drift":
        expected = drift_metrics(x, y, metrics=DRIFT_METRICS)
        actual = drift_on_device(x, y, DRIFT_METRICS)
    else:
        expected = pair_metrics(x, y, metrics=PAIR_METRICS)
        actual = pairs_on_device(x, y, PAIR_METRICS, 1e-8)
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12, nan_ok=True)


@pytest.mark.parametrize(
    "analysis",
    [
        "pretraining_drift",
        "spectral_fingerprints",
        "cross_language",
        "cross_entity",
        "language_entity",
        "cross_backbone",
        "language_entity_decomposition",
        "model_centrality",
    ],
)
@cuda
def test_all_analyses_cpu_cuda_parity(study, analysis):
    checkpoints, originals = study
    arguments = dict(
        analysis=analysis,
        checkpoints=checkpoints,
        originals=originals,
        options=MetricOptions(spectral_mode="exact", min_fingerprint_components=1),
    )
    if analysis == "model_centrality":
        pairs = analyze(
            device="cpu", analysis="cross_backbone", checkpoints=checkpoints, originals=originals
        )
        arguments = dict(analysis=analysis, pair_results=pairs)
    cpu = analyze(device="cpu", **arguments)
    gpu = analyze(device="GPU", **arguments)
    pd.testing.assert_frame_equal(cpu.data, gpu.data, rtol=1e-6, atol=1e-8)
    assert cpu.unit_id != gpu.unit_id
    assert gpu.provenance["configuration"]["options"]["device"].startswith("cuda:")


@cuda
def test_cuda_primitives_stay_on_device_and_never_use_scipy(monkeypatch):
    import importlib

    module = importlib.import_module("weight_analysis.metrics.spectral")

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA decomposition used SciPy")

    monkeypatch.setattr(module, "svds", forbidden)
    x = torch.diag(torch.linspace(20.0, 1.0, 20, dtype=torch.float64))
    y = x + 0.1
    assert topk_overlap(x, y, 0.2, device="gpu") == pytest.approx(
        topk_overlap(x, y, 0.2), nan_ok=True
    )
    for function, names in ((drift_metrics, DRIFT_METRICS), (pair_metrics, PAIR_METRICS)):
        expected = function(y, x, metrics=names, device="cpu")
        actual = function(y, x, metrics=names, device="gpu")
        assert actual == pytest.approx(expected, rel=1e-8, abs=1e-10, nan_ok=True)
    cpu = spectral_metrics(x, metrics=SPECTRAL_METRICS, device="cpu")
    gpu = spectral_metrics(x, metrics=SPECTRAL_METRICS, device="cuda")
    for name in SPECTRAL_METRICS:
        np.testing.assert_allclose(cpu[name], gpu[name], rtol=1e-7, atol=1e-9)
    approximation = spectral_metrics(
        x, metrics=["effective_rank", "topk_energy_capture"], device="gpu", mode="topk", topk=4
    )
    assert approximation["status"] == "topk_approx"
    assert np.isnan(approximation["effective_rank"])
    assert 0 < approximation["topk_energy_capture"] <= 1
    basis = principal_basis(x.to("cuda"), rank=3)
    assert basis.is_cuda
    assert principal_angles(basis, basis).is_cuda
    approximate_basis = principal_basis(x.to("cuda"), rank=3, max_exact_elements=1)
    assert approximate_basis.is_cuda
    torch.testing.assert_close(
        approximate_basis.T @ approximate_basis, torch.eye(3, device="cuda", dtype=torch.float64)
    )
    from weight_analysis.metrics import basis as historical_basis
    from weight_analysis.metrics import spectral

    assert spectral(x, device="gpu", mode="topk", topk=4)["status"] == "topk_approx"
    historical, status = historical_basis(x, rank=3, device="gpu", max_exact_elements=1)
    assert historical.is_cuda and status == "topk_approx"
