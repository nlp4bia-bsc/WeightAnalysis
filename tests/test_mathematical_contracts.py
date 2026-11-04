"""Independent mathematical contracts for selective, CPU-only primitives."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import torch
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr

from weight_analysis.metrics import (
    DRIFT_METRICS,
    PAIR_METRICS,
    SPECTRAL_METRICS,
    VectorAccumulator,
    drift_metrics,
    pair_metrics,
    principal_angles,
    principal_basis,
    relative_frobenius,
    spectral_metrics,
    topk_overlap,
)


@pytest.mark.parametrize("seed", [0, 7, 29])
def test_all_drift_statistics_against_numpy(seed):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(4, 3))
    final = base + rng.normal(size=base.shape) / 7
    delta = final - base
    expected = {
        "relative_frobenius": np.linalg.norm(delta) / np.linalg.norm(base),
        "relative_l2": np.linalg.norm(delta) / np.linalg.norm(base),
        "delta_l2": np.linalg.norm(delta),
        "delta_rms": np.sqrt(np.mean(delta**2)),
        "delta_mae": np.mean(np.abs(delta)),
        "delta_l1": np.sum(np.abs(delta)),
        "delta_energy": np.sum(delta**2),
        "base_l2": np.linalg.norm(base),
        "final_l2": np.linalg.norm(final),
        "cosine_final_base": np.sum(base * final) / (np.linalg.norm(base) * np.linalg.norm(final)),
        "max_abs_delta": np.max(np.abs(delta)),
        "delta_mean": np.mean(delta),
    }
    result = drift_metrics(torch.from_numpy(final), torch.from_numpy(base), metrics=DRIFT_METRICS)
    assert set(result) == set(expected)
    for name, value in expected.items():
        assert result[name] == pytest.approx(value, abs=1e-13)
    assert relative_frobenius(torch.from_numpy(final), torch.from_numpy(base)) == pytest.approx(
        expected["relative_frobenius"]
    )


@pytest.mark.parametrize("seed", [1, 8, 31])
def test_all_pair_statistics_against_numpy_and_scipy(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(2, 15))
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    cosine = dot / (norm_a * norm_b)
    mask = (np.abs(a) > 0.25) & (np.abs(b) > 0.25)
    expected = {
        "cosine": cosine,
        "angle_over_pi": np.arccos(cosine) / np.pi,
        "pearson": pearsonr(a, b).statistic,
        "l2_distance": np.linalg.norm(a - b),
        "normalized_l2": np.linalg.norm(a - b) / np.sqrt(norm_a**2 + norm_b**2),
        "l1_mean_distance": np.mean(np.abs(a - b)),
        "rmse": np.sqrt(np.mean((a - b) ** 2)),
        "norm_a": norm_a,
        "norm_b": norm_b,
        "sign_agreement": np.mean(np.sign(a[mask]) == np.sign(b[mask])),
        "sign_conflict": np.mean(np.sign(a[mask]) != np.sign(b[mask])),
    }
    result = pair_metrics(
        torch.from_numpy(a), torch.from_numpy(b), metrics=PAIR_METRICS, sign_floor=0.25
    )
    assert set(result) == set(expected)
    for name, value in expected.items():
        assert result[name] == pytest.approx(value, abs=1e-13)


@pytest.mark.parametrize("seed", [3, 9, 22])
def test_spectral_statistics_against_numpy_svd(seed):
    matrix = np.random.default_rng(seed).normal(size=(8, 5))
    singular = np.linalg.svd(matrix, compute_uv=False)
    probability = singular / singular.sum()
    energy_probability = singular**2 / np.sum(singular**2)
    entropy = -np.sum(probability * np.log(probability))
    energy_entropy = -np.sum(energy_probability * np.log(energy_probability))
    norm = np.linalg.norm(matrix)
    result = spectral_metrics(torch.from_numpy(matrix), metrics=SPECTRAL_METRICS)
    expected = {
        "matrix_fro": norm,
        "spectral_norm": singular[0],
        "nuclear_norm": singular.sum(),
        "stable_rank": norm**2 / singular[0] ** 2,
        "effective_rank": np.exp(entropy),
        "effective_rank_fraction": np.exp(entropy) / 5,
        "energy_effective_rank": np.exp(energy_entropy),
        "spectral_entropy_sigma": entropy,
        "spectral_entropy_energy": energy_entropy,
        "top1_energy": energy_probability[0],
        "top5_energy": 1.0,
        "top10_energy": 1.0,
        "topk_energy_capture": 1.0,
    }
    assert result["status"] == "exact"
    for name, value in expected.items():
        assert result[name] == pytest.approx(value, abs=1e-12)
    for cutoff in (90, 95, 99):
        count = int(np.searchsorted(np.cumsum(energy_probability), cutoff / 100)) + 1
        assert result[f"rank{cutoff}_fraction"] == pytest.approx(count / 5)
    curve = np.interp(
        np.linspace(0, 1, 32), np.linspace(0, 1, 6), np.r_[0, np.cumsum(energy_probability)]
    )
    np.testing.assert_allclose(result["spectral_energy_curve"], curve, atol=1e-12)
    np.testing.assert_allclose(
        result["normalized_weight_quantiles"],
        np.quantile(matrix / np.sqrt(np.mean(matrix**2)), np.linspace(0, 1, 21)),
        atol=1e-12,
    )


def test_metric_selection_performs_no_unrequested_svd_or_quantiles(monkeypatch):
    module = importlib.import_module("weight_analysis.metrics.spectral")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unrequested decomposition or quantile computation")

    monkeypatch.setattr(torch.linalg, "svdvals", forbidden)
    monkeypatch.setattr(module, "svds", forbidden)
    monkeypatch.setattr(np, "quantile", forbidden)
    matrix = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    assert spectral_metrics(matrix, metrics=["matrix_fro"]) == {
        "status": "not_required",
        "matrix_fro": 5.0,
    }
    assert set(drift_metrics(matrix, matrix, metrics=["relative_frobenius"])) == {
        "relative_frobenius"
    }
    assert set(pair_metrics(matrix, matrix, metrics=["cosine"])) == {"cosine"}


def test_quantiles_only_do_not_decompose(monkeypatch):
    module = importlib.import_module("weight_analysis.metrics.spectral")

    def forbidden(*args, **kwargs):
        raise AssertionError("SVD must not run for entry quantiles")

    monkeypatch.setattr(torch.linalg, "svdvals", forbidden)
    monkeypatch.setattr(module, "svds", forbidden)
    result = spectral_metrics(torch.eye(3), metrics=["normalized_weight_quantiles"])
    assert result["status"] == "not_required"
    assert len(result["normalized_weight_quantiles"]) == 21


def test_joint_spectral_selection_reuses_one_decomposition(monkeypatch):
    original = torch.linalg.svdvals
    calls = []

    def counted(matrix):
        calls.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(torch.linalg, "svdvals", counted)
    spectral_metrics(torch.eye(4), metrics=["effective_rank", "nuclear_norm", "spectral_norm"])
    assert calls == [torch.Size([4, 4])]


def test_truncation_is_deterministic_and_keeps_exact_only_metrics_undefined():
    matrix = torch.from_numpy(np.random.default_rng(9).normal(size=(9, 7)))
    names = ["spectral_norm", "effective_rank", "nuclear_norm", "spectral_energy_curve"]
    first = spectral_metrics(matrix, metrics=names, mode="topk", topk=2)
    second = spectral_metrics(matrix, metrics=names, mode="topk", topk=2)
    assert first["status"] == second["status"] == "topk_approx"
    assert first["spectral_norm"] == pytest.approx(second["spectral_norm"], abs=1e-12)
    assert first["spectral_norm"] == pytest.approx(
        np.linalg.svd(matrix.numpy(), compute_uv=False)[0]
    )
    assert math.isnan(first["effective_rank"]) and math.isnan(first["nuclear_norm"])
    assert first["spectral_energy_curve"] is None


def test_zero_and_constant_denominators_have_explicit_null_convention():
    zeros = torch.zeros(3)
    assert math.isnan(relative_frobenius(torch.ones(3), zeros))
    pair = pair_metrics(zeros, zeros, metrics=PAIR_METRICS)
    assert pair["l2_distance"] == pair["rmse"] == 0
    assert math.isnan(pair["cosine"]) and math.isnan(pair["pearson"])
    assert math.isnan(pair["sign_agreement"]) and math.isnan(pair["normalized_l2"])
    spectrum = spectral_metrics(torch.zeros(3, 3), metrics=SPECTRAL_METRICS)
    assert spectrum["status"] == "zero_matrix" and spectrum["matrix_fro"] == 0
    assert math.isnan(spectrum["effective_rank"])
    assert spectrum["spectral_energy_curve"] is None


@pytest.mark.parametrize("function", [drift_metrics, pair_metrics])
@pytest.mark.parametrize(
    "bad",
    [torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.tensor([1]), torch.empty(0)],
)
def test_vector_metrics_reject_invalid_inputs(function, bad):
    with pytest.raises(ValueError):
        function(bad, torch.ones_like(bad))


@pytest.mark.parametrize(
    "function,metrics",
    [(drift_metrics, []), (pair_metrics, ["not_a_metric"]), (spectral_metrics, "effective_rank")],
)
def test_metric_names_are_validated_before_computation(function, metrics):
    with pytest.raises(ValueError):
        if function is spectral_metrics:
            function(torch.eye(3), metrics=metrics)
        else:
            function(torch.ones(3), torch.ones(3), metrics=metrics)


@pytest.mark.parametrize(
    "kwargs", [{"mode": "unknown"}, {"topk": 0}, {"quantile_points": 1}, {"device": "tpu"}]
)
def test_spectral_configuration_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        spectral_metrics(torch.eye(4), **kwargs)


def test_numerical_rank_basis_excludes_arbitrary_nullspace():
    matrix = torch.diag(torch.tensor([4.0, 0.0, 0.0], dtype=torch.float64))
    result = principal_basis(matrix, rank=3)
    assert result.shape == (3, 1)
    np.testing.assert_allclose((result @ result.T).numpy(), np.diag([1.0, 0.0, 0.0]))
    assert principal_basis(torch.zeros(4, 3), rank=3).shape == (4, 0)


def test_principal_angles_against_independent_scipy_and_basis_rotation():
    rng = np.random.default_rng(6)
    a, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    b, _ = np.linalg.qr(rng.normal(size=(8, 2)))
    angles = principal_angles(torch.from_numpy(a), torch.from_numpy(b))
    np.testing.assert_allclose(np.sort(angles.numpy()), np.sort(subspace_angles(a, b)), atol=1e-12)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    np.testing.assert_allclose(
        principal_angles(torch.from_numpy(a @ rotation), torch.from_numpy(b)), angles, atol=1e-12
    )


def test_principal_angles_reject_nonorthonormal_and_incompatible_bases():
    with pytest.raises(ValueError, match="orthonormal"):
        principal_angles(torch.ones(4, 2), torch.ones(4, 2))
    with pytest.raises(ValueError, match="coordinate"):
        principal_angles(torch.eye(4)[:, :2], torch.eye(5)[:, :2])
    with pytest.raises(ValueError, match="nonempty"):
        principal_angles(torch.empty(4, 0), torch.empty(4, 0))


def test_centered_pearson_avoids_legacy_raw_moment_cancellation():
    a = torch.tensor([1e12 + n for n in range(1, 6)], dtype=torch.float64)
    b = torch.tensor([1e12 + 2 * n for n in range(1, 6)], dtype=torch.float64)
    assert pair_metrics(a, b, metrics=["pearson"])["pearson"] == pytest.approx(1)
    historical = VectorAccumulator()
    historical.update(a, b)
    assert not math.isclose(historical.result()["pearson"], 1, rel_tol=1e-7)


def test_overlap_sparse_support_and_scaling_invariance():
    a = torch.tensor([5.0, -3.0, 0.0, 0.0])
    b = torch.tensor([4.0, 2.0, 0.0, 0.0])
    result = topk_overlap(a, b, fraction=1)
    assert result["topk_k_a"] == result["topk_k_b"] == result["topk_intersection"] == 2
    assert result["topk_jaccard"] == 1 and result["topk_sign_agreement"] == 0.5
    assert topk_overlap(a * 7, b * 7, 1)["weighted_jaccard_abs"] == pytest.approx(
        result["weighted_jaccard_abs"]
    )


def test_spectral_scale_and_orthogonal_invariants():
    matrix = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    names = ["matrix_fro", "effective_rank", "stable_rank"]
    first = spectral_metrics(matrix, metrics=names)
    changed = spectral_metrics(7 * rotation @ matrix, metrics=names)
    assert changed["matrix_fro"] == pytest.approx(7 * first["matrix_fro"])
    assert changed["effective_rank"] == pytest.approx(first["effective_rank"])
    assert changed["stable_rank"] == pytest.approx(first["stable_rank"])
