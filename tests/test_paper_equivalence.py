"""Numerical validations tied to published equations and author source files.

These tests require no network and compare against independent oracle formulas.
They do NOT claim to execute the external repositories. For that, use
`tests/test_external_oracles.py` with an approved local reference checkout.
"""
from __future__ import annotations
import math
import numpy as np
import pytest
import torch
from scipy import stats
from scipy.stats import wasserstein_distance

from weightspace.metrics import (
    drift, VectorAccumulator, spectral, basis, subspaces, topk_overlap,
    paper_trace_similarity, paper_subspace_layer_alignment,
    paper_subspace_hierarchical_alignment, quantile_w1,
)

# Task Arithmetic, Ilharco et al., ICLR 2023. Official:
# https://github.com/mlfoundations/task_vectors/blob/main/src/task_vectors.py
# Original code, lines 18–21: vector[key] = finetuned[key] - pretrained[key].
def test_task_arithmetic_delta_against_official_formula():
    original = torch.tensor([[1., 2.], [-4., 3.]])
    tuned = torch.tensor([[3., 1.], [-2., 7.]])
    reference = tuned - original
    got = drift(tuned, original)
    np.testing.assert_allclose(got['delta_l2'], torch.linalg.vector_norm(reference).item(), atol=1e-12)
    np.testing.assert_allclose(got['delta_l1'], reference.abs().sum().item(), atol=1e-12)
    np.testing.assert_allclose(got['delta_rms'], reference.square().mean().sqrt().item(), atol=1e-12)


# Group then Scale, Li et al., Findings ACL 2025. Official:
# https://github.com/ZNLP/DMoE/blob/main/src/delta_smi/bloom.py
# Original lines 45–63: base - tuned, stack and F.cosine_similarity(...).
# Reversing BOTH deltas preserves their cosine, but not the delta sign itself.
def test_dmoe_language_similarity_official_torch_formula():
    base = torch.tensor([3., -2., 5., 7.], dtype=torch.float64)
    a = torch.tensor([2., 2., 4., 8.], dtype=torch.float64)
    b = torch.tensor([1., 3., 7., 6.], dtype=torch.float64)
    official = torch.nn.functional.cosine_similarity((base-a)[None], (base-b)[None]).item()
    acc = VectorAccumulator()
    acc.update(a-base,b-base,threshold=0.)
    np.testing.assert_allclose(acc.result()['cosine'],official,atol=1e-12)


# Roy & Vetterli, EUSIPCO 2007, effective rank: H(p) with p=sigma/sum(sigma).
# Our squared-singular-value alternative is NOT their effective rank.
def test_roy_vetterli_effective_rank_and_energy_alternative():
    m = torch.diag(torch.tensor([3.,1.,0.],dtype=torch.float64))
    found=spectral(m,mode='exact')
    p=np.array([3/4,1/4]); pe=np.array([9/10,1/10])
    expected=np.exp(-(p*np.log(p)).sum())
    expected_e=np.exp(-(pe*np.log(pe)).sum())
    np.testing.assert_allclose(found['effective_rank'],expected,atol=1e-12)
    np.testing.assert_allclose(found['energy_effective_rank'],expected_e,atol=1e-12)
    assert found['effective_rank']!=pytest.approx(found['energy_effective_rank'])


# Martin & Mahoney (JMLR 2021); weight-space spectral norms are classical:
# ||W||_F² = tr(WᵀW) = sum_i sigma_i², ||W||_2 = sigma_max.
# This does NOT reproduce WeightWatcher's power-law/RMT alpha fitting.
def test_spectral_reference_numpy_eigen_norms():
    x=np.array([[1.,-3.,4.],[0.,2.,5.],[1.,0.,-1.]])
    s=np.linalg.svd(x,compute_uv=False)
    d=spectral(torch.from_numpy(x),mode='exact')
    assert d['matrix_fro']==pytest.approx(float(np.linalg.norm(x,'fro')))
    assert d['matrix_fro']**2==pytest.approx(float(np.trace(x.T@x)))
    assert d['spectral_norm']==pytest.approx(float(s[0]))
    assert d['nuclear_norm']==pytest.approx(float(s.sum()))
    assert d['stable_rank']==pytest.approx(float(np.square(s).sum()/s[0]**2))


# TRACE Chen, Shang & Liu (COLM 2026), official source trace.py Eq. 4–5.
# https://github.com/OPTML-Group/LLM-Biometrics/blob/main/llm_biometrics/trace.py
# Independent oracle: scipy.stats.pearsonr after np.interp to max depth.
def test_trace_exact_matches_independent_scipy_oracle():
    a={'q':[1.,3.,2.,6.], 'v':[2.,4.,8.], 'o':[5.,5.,5.]}
    b={'q':[8.,5.,7.], 'v':[7.,3.,2.,9.], 'o':[9.,9.,9.]}
    result=paper_trace_similarity(a,b)
    def oracle(x,y):
        t=max(len(x),len(y))
        xx=np.interp(np.linspace(0,1,t),np.linspace(0,1,len(x)),x)
        yy=np.interp(np.linspace(0,1,t),np.linspace(0,1,len(y)),y)
        return float(stats.pearsonr(xx,yy).statistic)
    scores=[oracle(a[c],b[c]) for c in ('q','v')]
    np.testing.assert_allclose(result['trace_similarity'],np.mean(scores),atol=1e-12)
    assert result['trace_n_components']==2
    assert set(result['trace_per_component'])=={'q','v'}
    # Trace similarity is not normalized spectral-energy-curve L1.
    assert paper_trace_similarity({'q':[1,2,3]},{'q':[3,2,1]})['trace_similarity']==pytest.approx(-1.)


# Paper TRACE requires >=2 nonconstant aligned layers.
def test_trace_undefined_profiles_skip_correctly():
    a={'q':[1.,1.,1.],'k':[1.]}
    b={'q':[2.,2.,2.],'k':[4.]}
    v=paper_trace_similarity(a,b)
    assert math.isnan(v['trace_similarity']) and v['trace_n_components']==0


# Chen et al. COLM 2026 subspace.py Eq. (8): average J SMALLEST
# principal cosines, not global mean cosine or chordal distance.
def test_colm_subspace_worst_j_cosines_not_mean():
    # Three 1D angles: 0 deg, 60 deg, 90 deg in independent coordinate planes.
    a=torch.zeros(6,3,dtype=torch.float64)
    b=torch.zeros(6,3,dtype=torch.float64)
    for i,ang in enumerate((0.,math.pi/3,math.pi/2)):
        a[2*i,i]=1
        b[2*i,i]=math.cos(ang)
        b[2*i+1,i]=math.sin(ang)
    assert paper_subspace_layer_alignment(a,b,j=1)==pytest.approx(0.,abs=1e-12)
    assert paper_subspace_layer_alignment(a,b,j=2)==pytest.approx(.25,abs=1e-12)
    assert paper_subspace_layer_alignment(a,b,j=3)==pytest.approx(.5,abs=1e-12)
    our_other_metric=subspaces((a,'test'),(b,'test'))
    assert our_other_metric['subspace_mean_cos']==pytest.approx(.5,abs=1e-12)
    # Aggregation of least-aligned layers (K_layer=2), then components equally.
    scores=paper_subspace_hierarchical_alignment({'q':[.5,.1,.9], 'k':[.7,.8,.2]}, n_bottom=2)
    assert scores['subspace_per_component']['q']==pytest.approx(.3)
    assert scores['subspace_per_component']['k']==pytest.approx(.45)
    assert scores['subspace_similarity']==pytest.approx(.375)


# TIES Merging (Yadav et al., NeurIPS 2023), official merge_utils.py:
# https://github.com/prateeky2806/ties-merging/blob/main/src/utils/merge_utils.py
# TIES trim uses kthvalue and >=; it can retain >k due to tied magnitudes.
# Our EXACT-cardinality topk Jaccard (positive changes only, stable ties)
# is a separate research metric and MUST NEVER be labeled 'TIES trim'.
def test_ties_topk_tie_behavior_intentionally_not_equivalent():
    a=torch.tensor([3.,3.,3.,1.]); b=torch.tensor([3.,3.,1.,3.])
    ours=topk_overlap(a,b,fraction=.5)
    assert ours['topk_k_a']==2 and ours['topk_k_b']==2
    # A threshold-based version can include 3 ties at magnitude 3;
    # proves reproduction of TIES trim cannot be inferred from our Jaccard.
    threshold=torch.kthvalue(a.abs(), 2).values
    selected=(a.abs()>=threshold)
    assert int(selected.sum())!=ours['topk_k_a']


# Classical Wasserstein-1 and quantile approximation. Our quantile_w1
# approximates the integral of |F_a^-1-F_b^-1| by trapezoidal quadrature,
# not scipy.stats.wasserstein_distance on original empirical samples.
def test_quantile_approximation_bound_and_perfect_shift():
    a=np.linspace(-1.,1.,101)
    b=a+3.5
    qa=np.quantile(a,np.linspace(0,1,21))
    qb=np.quantile(b,np.linspace(0,1,21))
    assert quantile_w1(qa.tolist(),qb.tolist())==pytest.approx(3.5,abs=1e-12)
    assert wasserstein_distance(a,b)==pytest.approx(3.5,abs=1e-12)


# Truncated SVD cannot determine entropy, nuclear norm or exact energy CDF.
def test_truncated_svd_never_mislabels_exact_entropy():
    torch.manual_seed(13)
    x=torch.randn(25,20,dtype=torch.float64)
    out=spectral(x,mode='auto',max_exact_elements=3,topk=4)
    assert out['status']=='topk_approx'
    assert math.isnan(out['effective_rank'])
    assert math.isnan(out['energy_effective_rank'])
    assert math.isnan(out['nuclear_norm'])
    assert out['spectral_energy_curve'] is None


# A paper-inspired ranking must never be computed on NaN/Inf silently.
def test_overlap_rejects_nonfinite_inputs():
    with pytest.raises(ValueError,match='nonfinite'):
        topk_overlap(torch.tensor([1.,float('nan')]),torch.tensor([1.,2.]),.5)
