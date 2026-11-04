"""Device routing and mathematical parity; real CUDA checks skip without hardware."""

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


def cuda(test):
    """Expose actual CUDA cases through pytest -m gpu and skip them without hardware."""
    return pytest.mark.gpu(
        pytest.mark.skipif(
            not torch.cuda.is_available(), reason="torch.cuda.is_available() is False"
        )(test)
    )


@pytest.fixture(params=list(range(torch.cuda.device_count())) or [0], ids=lambda i: f"cuda:{i}")
def cuda_device(request):
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")
    device = torch.device("cuda", request.param)
    yield device
    torch.cuda.synchronize(device)


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


METRIC_CASES = (
    [("drift", name) for name in DRIFT_METRICS]
    + [("pair", name) for name in PAIR_METRICS]
    + [("spectral", name) for name in SPECTRAL_METRICS]
)


@pytest.mark.parametrize("family,name", METRIC_CASES, ids=[f"{f}:{m}" for f, m in METRIC_CASES])
@cuda
def test_each_metric_executes_on_real_cuda(cuda_device, family, name):
    generator = torch.Generator().manual_seed(71)
    base = torch.randn(12, 8, generator=generator, dtype=torch.float64)
    final = 1.3 * base + base.sin() / 5
    if family == "spectral":
        expected = spectral_metrics(base, metrics=[name], device="cpu")[name]
        actual = spectral_metrics(base, metrics=[name], device=str(cuda_device))[name]
    else:
        function = drift_metrics if family == "drift" else pair_metrics
        expected = function(final, base, metrics=[name], device="cpu")[name]
        actual = function(final, base, metrics=[name], device=str(cuda_device))[name]
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@cuda
def test_cuda_noncontiguous_inputs_keep_device_and_promote_precision(cuda_device, dtype):
    from weight_analysis.metrics._validation import floating_tensor

    matrix = torch.arange(1, 49, dtype=torch.float64).reshape(6, 8).to(dtype).T
    assert not matrix.is_contiguous()
    resident = matrix.to(cuda_device)
    converted = floating_tensor(resident, name="matrix")
    assert converted.device == cuda_device
    assert converted.dtype == torch.float64
    expected = spectral_metrics(matrix, metrics=["matrix_fro", "normalized_weight_quantiles"])
    actual = spectral_metrics(resident, metrics=["matrix_fro", "normalized_weight_quantiles"])
    np.testing.assert_allclose(actual["matrix_fro"], expected["matrix_fro"], rtol=1e-8)
    np.testing.assert_allclose(
        actual["normalized_weight_quantiles"], expected["normalized_weight_quantiles"], rtol=1e-8
    )


@pytest.mark.parametrize("case", ["zero", "constant", "sparse", "opposite"])
@cuda
def test_cuda_degenerate_vector_cases_match_cpu(cuda_device, case):
    a = torch.tensor([0.0, -3.0, 3.0, 0.0, 1.0], dtype=torch.float64)
    b = a.clone()
    if case == "zero":
        a.zero_()
        b.zero_()
    elif case == "constant":
        a.fill_(2)
        b.fill_(4)
    elif case == "sparse":
        b = torch.tensor([0.0, 2.0, 0.0, 0.0, -1.0], dtype=torch.float64)
    else:
        b = -a
    expected = pair_metrics(a, b, metrics=PAIR_METRICS, device="cpu")
    actual = pair_metrics(a, b, metrics=PAIR_METRICS, device=str(cuda_device))
    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-10, nan_ok=True)
    assert topk_overlap(a, b, 0.4, device=str(cuda_device)) == pytest.approx(
        topk_overlap(a, b, 0.4), rel=1e-8, abs=1e-10, nan_ok=True
    )


@cuda
def test_cuda_rank_deficiency_zero_and_undefined_spectra(cuda_device):
    matrix = torch.zeros(6, 4, dtype=torch.float64, device=cuda_device)
    empty = principal_basis(matrix, rank=4)
    assert empty.device == cuda_device and empty.shape == (6, 0)
    values = spectral_metrics(matrix, metrics=["matrix_fro", "effective_rank"])
    assert values["status"] == "zero_matrix" and values["matrix_fro"] == 0
    assert np.isnan(values["effective_rank"])
    matrix[0, 0], matrix[1, 1] = 4.0, 2.0
    basis = principal_basis(matrix, rank=4)
    assert basis.device == cuda_device and basis.shape == (6, 2)
    torch.testing.assert_close(
        basis.T @ basis, torch.eye(2, dtype=torch.float64, device=cuda_device)
    )
    with pytest.raises(ValueError, match="orthonormal"):
        principal_angles(2 * basis, basis)


@cuda
def test_cuda_randomized_decomposition_preserves_cpu_and_cuda_rng(cuda_device):
    from weight_analysis.metrics.decomposition import truncated_left_basis

    matrix = torch.diag(torch.logspace(0, -6, 80, dtype=torch.float64)).to(cuda_device)
    cpu_before = torch.random.get_rng_state().clone()
    gpu_before = torch.cuda.get_rng_state(cuda_device).clone()
    first = truncated_singular_values(matrix, 4)
    second = truncated_singular_values(matrix, 4)
    left, values = truncated_left_basis(matrix, 4)
    assert first.device == left.device == values.device == cuda_device
    torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        first.cpu(), np.linalg.svd(matrix.cpu().numpy(), compute_uv=False)[:4], rtol=1e-6, atol=1e-9
    )
    assert torch.equal(cpu_before, torch.random.get_rng_state())
    assert torch.equal(gpu_before, torch.cuda.get_rng_state(cuda_device))


@pytest.mark.parametrize(
    "analysis", ["pretraining_drift", "spectral_fingerprints", "language_entity_decomposition"]
)
@cuda
def test_checkpoint_pipeline_calls_kernels_on_cuda(study, cuda_device, analysis, monkeypatch):
    checkpoints, originals = study
    calls = []
    name = "stack" if analysis == "language_entity_decomposition" else "vector_norm"
    namespace = torch if name == "stack" else torch.linalg
    original = getattr(namespace, name)

    def on_cuda(value, *args, **kwargs):
        tensors = list(value) if name == "stack" else [value]
        assert all(t.device == cuda_device for t in tensors), "Numerical kernel ran on CPU"
        calls.append(name)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(namespace, name, on_cuda)
    result = analyze(
        device=str(cuda_device),
        analysis=analysis,
        checkpoints=checkpoints,
        originals=originals,
        options=MetricOptions(spectral_mode="exact"),
    )
    assert calls and not result.data.empty


@cuda
def test_cuda_cache_and_parquet_roundtrip_do_not_recompute(
    study, cuda_device, tmp_path, monkeypatch
):
    from weight_analysis.core import WeightReader

    checkpoints, originals = study
    arguments = dict(
        analysis="pretraining_drift",
        checkpoints=checkpoints[:1],
        originals=originals,
        cache_dir=tmp_path / "cache",
    )
    gpu = analyze(device=str(cuda_device), **arguments)
    cpu = analyze(device="cpu", **arguments)
    assert gpu.unit_id != cpu.unit_id
    restored = read_result(tmp_path / "cache" / f"{gpu.unit_id}.parquet")
    pd.testing.assert_frame_equal(gpu.data, restored.data, check_dtype=False)
    assert restored.provenance == gpu.provenance

    def no_recomputation(*args, **kwargs):
        raise AssertionError("Valid CUDA cache unexpectedly reread a tensor")

    monkeypatch.setattr(WeightReader, "get", no_recomputation)
    reused = analyze(device=str(cuda_device), **arguments)
    assert reused.unit_id == gpu.unit_id


@cuda
def test_cuda_cli_uses_requested_gpu_and_saves_result(study, cuda_device, tmp_path):
    from dataclasses import asdict

    checkpoints, originals = study
    checkpoint = asdict(checkpoints[0])
    checkpoint["path"] = str(checkpoint["path"])
    output = tmp_path / "cli-result.parquet"
    configuration = dict(
        analysis="pretraining_drift",
        device=str(cuda_device),
        checkpoints=[checkpoint],
        originals={k: str(v) for k, v in originals.items()},
        output=str(output),
    )
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(configuration))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-m", "weight_analysis", "run", str(config_path)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    result = read_result(output)
    assert report["unit_id"] == result.unit_id and report["rows"] == len(result.data) > 0
    assert result.provenance["configuration"]["options"]["device"] == str(cuda_device)
    assert "cuda_device" in result.provenance["versions"]


@cuda
def test_two_cuda_devices_have_distinct_provenance(study):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA GPUs required")
    checkpoints, originals = study
    arguments = dict(analysis="pretraining_drift", checkpoints=checkpoints[:1], originals=originals)
    first = analyze(device="cuda:0", **arguments)
    second = analyze(device="cuda:1", **arguments)
    pd.testing.assert_frame_equal(first.data, second.data, rtol=1e-7, atol=1e-9)
    assert first.unit_id != second.unit_id
