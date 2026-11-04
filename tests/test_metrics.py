from __future__ import annotations
import math
import numpy as np
import pytest
import torch
from weightspace.metrics import VectorAccumulator,drift,topk_overlap,spectral,basis,subspaces,quantile_w1


def test_vectors():
    x=torch.tensor([1.,2.,3.]); z=torch.zeros(3)
    v=VectorAccumulator();v.update(x,x);s=v.result()
    assert s['cosine']==pytest.approx(1) and s['l2_distance']==pytest.approx(0)
    v=VectorAccumulator();v.update(x,-x);assert v.result()['cosine']==pytest.approx(-1)
    v=VectorAccumulator();v.update(torch.tensor([1.,0.]),torch.tensor([0.,1.]));assert v.result()['cosine']==pytest.approx(0)
    v=VectorAccumulator();v.update(z,x); assert math.isnan(v.result()['cosine'])
    with pytest.raises(ValueError):VectorAccumulator().update(x,torch.tensor([1.]))
    assert drift(x,x)['delta_l2']==0
    assert drift(x,x)['cosine_final_base']==pytest.approx(1)
    with pytest.raises(ValueError):drift(x,torch.tensor([1.]))


def test_ties():
    x=torch.tensor([9.,8.,1.,0.]); y=torch.tensor([9.,8.,0.,1.]);z=torch.tensor([0.,1.,9.,8.])
    assert topk_overlap(x,x,.5)['topk_jaccard']==1
    assert topk_overlap(x,z,.5)['topk_jaccard']==0
    assert topk_overlap(x,y,.5)['topk_expected_intersection']==1
    assert topk_overlap(torch.ones(4),torch.ones(4),.25)['topk_k']==1


def test_spectra():
    m=torch.diag(torch.tensor([4.,2.,1.]))
    v=spectral(m,mode='exact');assert v['status']=='exact'
    assert v['spectral_norm']==pytest.approx(4)
    assert v['stable_rank']==pytest.approx(21/16)
    assert v['effective_rank']==pytest.approx(math.exp(-sum(p*math.log(p) for p in [4/7,2/7,1/7])))
    assert v['spectral_energy_curve'][-1]==pytest.approx(1)
    assert spectral(torch.zeros(4,4),mode='exact')['status']=='zero_matrix'
    assert spectral(torch.ones(3),mode='exact')['status']=='not_matrix'
    tri=spectral(torch.randn(10,9),mode='topk',topk=2)
    assert tri['status']=='topk_approx' and math.isnan(tri['effective_rank'])
    assert tri['spectral_energy_curve'] is None
    rng=np.random.default_rng(2);matrix=torch.from_numpy(rng.standard_normal((7,4)))
    got=spectral(matrix,mode='exact')['spectral_norm']
    assert got==pytest.approx(np.linalg.svd(matrix.numpy(),compute_uv=False)[0],rel=1e-10)


def test_subspace():
    x=torch.diag(torch.tensor([4.,2.,1.]))
    assert subspaces(basis(x,2),basis(x,2))['subspace_chordal_normalized']==pytest.approx(0,abs=1e-7)
    a=torch.tensor([[1.,0.],[0.,0.],[0.,0.]],dtype=torch.float64)
    b=torch.tensor([[0.,0.],[1.,0.],[0.,0.]],dtype=torch.float64)
    assert subspaces(basis(a,1),basis(b,1))['subspace_mean_cos2']==pytest.approx(0,abs=1e-12)
    assert subspaces(basis(a,1),basis(torch.zeros(4,4),1))['subspace_status'].find('zero_matrix')>=0
    assert quantile_w1([0,1,2],[1,2,3])==pytest.approx(1)


def test_spectral_orthogonal_transform_invariance():
    # Singular values invariant under orthogonal transforms; vector cosines need not be.
    a=torch.tensor([[3.,0.],[0.,1.]])
    rot=torch.tensor([[0.,1.],[-1.,0.]])
    assert spectral(a,mode='exact')['effective_rank']==pytest.approx(spectral(rot@a,mode='exact')['effective_rank'])
    assert spectral(a,mode='exact')['matrix_fro']==pytest.approx(spectral(rot@a,mode='exact')['matrix_fro'])


def test_exact_subspace_basis_rotation():
    x=torch.tensor([[4.,0.,0.],[0.,3.,0.],[0.,0.,2.]],dtype=torch.float64)
    u,_=basis(x,2)
    rot=torch.tensor([[0.,1.],[-1.,0.]],dtype=torch.float64)
    from weightspace.metrics import subspaces
    stat=subspaces((u,'exact'),(u@rot,'exact'))
    assert stat['subspace_mean_cos2']==pytest.approx(1,abs=1e-12)


def test_standard_bert_attention_head_views():
    from weightspace.checkpoints import head_views
    conf=dict(model_type='bert',num_attention_heads=2)
    a=torch.arange(16).reshape(4,4)
    q=list(head_views(a,'q',conf))
    o=list(head_views(a,'o',conf))
    assert len(q)==len(o)==2
    assert torch.equal(q[0][1],a[:2,:])
    assert torch.equal(o[1][1],a[:,2:])
    assert list(head_views(a,'q',dict(model_type='unknown',num_attention_heads=2)))==[]


def test_weighted_jaccard():
    a=torch.tensor([1.,2.,3.]); b=torch.tensor([1.,2.,3.]); neg=-a
    assert topk_overlap(a,b,1.)['weighted_jaccard_abs']==pytest.approx(1.)
    assert topk_overlap(a,neg,1.)['topk_sign_agreement']==pytest.approx(0.)
    assert math.isnan(topk_overlap(torch.zeros(2),torch.zeros(2),1.)['weighted_jaccard_abs'])


def test_zero_topk_not_artificially_identical():
    z=torch.zeros(10)
    assert math.isnan(topk_overlap(z,z,.1)['topk_jaccard'])
    assert topk_overlap(z,z,.1)['topk_k_a']==0
    a=torch.tensor([0.,0.,3.,2.]);b=torch.tensor([0.,0.,3.,-2.])
    assert topk_overlap(a,b,1.)['topk_jaccard']==pytest.approx(1.)
    assert topk_overlap(a,b,1.)['topk_sign_agreement']==pytest.approx(.5)
