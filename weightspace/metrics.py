"""Pure weight-only mathematical metrics with traceable citations.

Standard norms/statistics are classical; articles below motivate their use
for fine-tuned neural networks rather than claiming their invention.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import wasserstein_distance
import torch

EPS=1e-14

# Classical Euclidean/Frobenius norms and inner products; weight-drift context:
# Radiya-Dixit & Wang, AISTATS 2020, https://proceedings.mlr.press/v108/radiya-dixit20a.html
# Task-vector adaptation: Ilharco et al., ICLR 2023, https://openreview.net/forum?id=6t0Kwf8-jrj
# Undefined zero norms return NaN (converted to Parquet null by the writer).
def drift(final:torch.Tensor, base:torch.Tensor)->dict:
    if final.shape != base.shape or not final.is_floating_point(): raise ValueError('incompatible drift')
    a = final.reshape(-1).to(torch.float64)
    b = base.reshape(-1).to(torch.float64)
    d = a-b
    n = a.numel()
    if n == 0: raise ValueError('empty parameter')
    dn = torch.linalg.vector_norm(d).item(); bn = torch.linalg.vector_norm(b).item()
    fn = torch.linalg.vector_norm(a).item()
    return dict(n_parameters=n,delta_l2=dn,base_l2=bn,final_l2=fn,
        relative_l2=(dn/bn if bn>EPS else math.nan),delta_rms=dn/math.sqrt(n),
        delta_mae=float(d.abs().mean()),max_abs_delta=float(d.abs().max()),delta_mean=float(d.mean()),
        cosine_final_base=float(torch.dot(a,b)/(fn*bn)) if fn>EPS and bn>EPS else math.nan,
        delta_l1=float(d.abs().sum()),delta_energy=dn*dn)


# Classical cosine/Pearson/L1/L2 statistics; use on Δθ follows Ilharco et al.
# (ICLR 2023) and Group then Scale, Li et al. (Findings ACL 2025)
# https://aclanthology.org/2025.findings-acl.87/
@dataclass
class VectorAccumulator:
    n:int=0
    xx:float=0.
    yy:float=0.
    xy:float=0.
    sx:float=0.
    sy:float=0.
    d2:float=0.
    l1:float=0.
    agree:int=0
    both:int=0
    conflict:int=0

    def update(self,x:torch.Tensor,y:torch.Tensor,threshold:float=1e-8):
        if x.shape != y.shape: raise ValueError('different vector shape')
        a=x.reshape(-1).to(torch.float64); b=y.reshape(-1).to(torch.float64)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('nonfinite vector')
        self.n+=a.numel(); self.xx+=float(torch.dot(a,a)); self.yy+=float(torch.dot(b,b)); self.xy+=float(torch.dot(a,b))
        self.sx+=float(a.sum()); self.sy+=float(b.sum())
        d=a-b; self.d2+=float(torch.dot(d,d)); self.l1+=float(d.abs().sum())
        mask=(a.abs()>threshold)&(b.abs()>threshold)
        self.both+=int(mask.sum()); self.agree+=int(((a*b>0)&mask).sum()); self.conflict+=int(((a*b<0)&mask).sum())

    def result(self):
        if self.n==0: raise ValueError('empty accumulator')
        xn=math.sqrt(self.xx); yn=math.sqrt(self.yy)
        cosine=float(np.clip(self.xy/(xn*yn),-1,1)) if xn>EPS and yn>EPS else math.nan
        cx=self.xx-self.sx**2/self.n; cy=self.yy-self.sy**2/self.n
        pearson=float(np.clip((self.xy-self.sx*self.sy/self.n)/math.sqrt(cx*cy),-1,1)) if cx>EPS and cy>EPS else math.nan
        return dict(n_parameters=self.n,cosine=cosine,angle_over_pi=math.acos(cosine)/math.pi if math.isfinite(cosine) else math.nan,
            pearson=pearson,l2_distance=math.sqrt(self.d2),
            normalized_l2=math.sqrt(self.d2/(self.xx+self.yy)) if self.xx+self.yy>EPS else math.nan,
            l1_mean_distance=self.l1/self.n,rmse=math.sqrt(self.d2/self.n),
            norm_a=xn,norm_b=yn,sign_agreement=self.agree/self.both if self.both else math.nan,
            sign_conflict=self.conflict/self.both if self.both else math.nan,sign_compared=self.both)


# Classical Jaccard/top-k set intersection; TIES-Merging motivation for large
# Δ-weight coordinates: Yadav et al., NeurIPS 2023.
# https://papers.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html
# Our stable lexicographic tie-break and overlap null model are derived diagnostics.
def topk_overlap(a:torch.Tensor,b:torch.Tensor,fraction:float)->dict:
    if a.shape != b.shape or not 0<fraction<=1: raise ValueError('invalid top-k inputs')
    aa=a.detach().cpu().reshape(-1).abs().numpy(); bb=b.detach().cpu().reshape(-1).abs().numpy()
    n=aa.size
    if not n: raise ValueError('empty tensor')
    if not (np.isfinite(aa).all() and np.isfinite(bb).all()):
        raise ValueError('nonfinite top-k inputs')
    k=max(1,math.ceil(n*fraction))
    # Stable ties: descending magnitude, then increasing coordinate index.
    # Do not count unchanged zero-valued coordinates as selected parameters.
    # For sparse updates, the effective top-k set can be smaller than k.
    ia=np.lexsort((np.arange(n),-aa))
    ib=np.lexsort((np.arange(n),-bb))
    ia=ia[aa[ia]>0][:k]
    ib=ib[bb[ib]>0][:k]
    common=int(np.intersect1d(ia,ib,assume_unique=True).size)
    union=len(ia)+len(ib)-common
    # Weighted Ruzicka/Jaccard on absolute updates is a classical generalized
    # Jaccard coefficient (intersection=min, union=max), applied here to Δθ.
    small=float(np.minimum(aa,bb).sum(dtype=np.float64))
    large=float(np.maximum(aa,bb).sum(dtype=np.float64))
    shared_sign=(np.sign(a.detach().cpu().reshape(-1).numpy())==
                 np.sign(b.detach().cpu().reshape(-1).numpy()))
    intersect_mask=np.zeros(n,dtype=bool)
    intersect_mask[np.intersect1d(ia,ib,assume_unique=True)]=True
    sign_consistent=int(np.sum(shared_sign & intersect_mask))
    expected=len(ia)*len(ib)/n
    return dict(topk_fraction=fraction,topk_k=k,topk_k_a=len(ia),topk_k_b=len(ib),topk_intersection=common,
        topk_jaccard=common/union if union else math.nan,
        topk_overlap_coefficient=common/min(len(ia),len(ib)) if len(ia) and len(ib) else math.nan,
        topk_expected_intersection=expected,
        topk_excess_intersection=common-expected,
        weighted_jaccard_abs=small/large if large>EPS else math.nan,
        topk_sign_agreement=sign_consistent/common if common else math.nan)


# SVD, nuclear/spectral/Frobenius norm: classical numerical linear algebra.
# Roy & Vetterli, EUSIPCO 2007, definition of eRank: p_i=σ_i/Σσ.
# https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf
# Martin & Mahoney, JMLR 2021, spectral analysis of trained neural weights.
# https://www.jmlr.org/papers/v22/20-410.html
# Chen et al. (2026), spectral weight fingerprints: https://arxiv.org/abs/2608.07786
# Energy effective rank p_i=σ²_i/Σσ² is EXPLICITLY a derived distinct metric.
def spectral(matrix:torch.Tensor,mode:str='auto',max_exact_elements:int=3_000_000,
             topk:int=32,device:str='cpu',curve_points:int=32,quantile_points:int=21)->dict:
    if matrix.ndim!=2 or min(matrix.shape)<2:
        return dict(status='not_matrix')
    if not torch.isfinite(matrix).all(): raise ValueError('nonfinite spectral matrix')
    r,c=matrix.shape; rank=min(r,c)
    if mode=='skip': return dict(status='disabled')
    full=(mode=='exact' or mode=='auto' and matrix.numel()<=max_exact_elements or rank<=2)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    norm=float(torch.linalg.vector_norm(matrix.to(torch.float64)))
    if norm<=EPS:
        return dict(status='zero_matrix',matrix_rows=r,matrix_cols=c,matrix_fro=0.,
                    stable_rank=math.nan,effective_rank=math.nan,energy_effective_rank=math.nan,
                    spectral_energy_curve=None)
    with torch.no_grad():
        if full:
            inp=matrix.to(device=device,dtype=torch.float64)
            s=torch.linalg.svdvals(inp).detach().cpu().numpy()
            del inp
            status='exact'
        else:
            k=min(max(1,topk),rank-1)
            if k<1: raise ValueError('invalid truncated rank')
            # scipy svds returns unordered singular values; explicit RNG fixed via v0.
            inp=matrix.detach().cpu().numpy()
            try: s=svds(inp,k=k,return_singular_vectors=False,v0=np.ones(min(inp.shape)))
            except Exception as ex: raise RuntimeError(f'Truncated SVD failed for {r}x{c}: {ex}') from ex
            status='topk_approx'
    s=np.sort(np.maximum(np.asarray(s,dtype=np.float64),0))[::-1]
    energy=s*s
    capture=float(energy.sum()/(norm*norm))
    capture=float(np.clip(capture,0,1))
    sigma_p=s/s.sum() if full else None
    eng_p=energy/energy.sum() if full else None
    erank=float(np.exp(-np.sum(sigma_p[sigma_p>0]*np.log(sigma_p[sigma_p>0])))) if full else math.nan
    e_erank=float(np.exp(-np.sum(eng_p[eng_p>0]*np.log(eng_p[eng_p>0])))) if full else math.nan
    cdf=np.cumsum(eng_p) if full else None
    if full:
        x=np.linspace(0,1,len(s)+1)
        curve=np.interp(np.linspace(0,1,curve_points),x,np.r_[0,cdf]).tolist()
    else:
        curve=None
    # Normalized weight distributions: classical quantiles, *not* exact TRACE.
    # May be expensive for large matrices but does not retain tensor copies.
    flat=matrix.detach().cpu().numpy().ravel()
    rms=norm/math.sqrt(flat.size)
    wq=np.quantile(flat/rms,np.linspace(0,1,quantile_points)).tolist()
    return dict(status=status,matrix_rows=r,matrix_cols=c,max_possible_rank=rank,
      n_singular_computed=len(s),matrix_fro=norm,spectral_norm=float(s[0]),
      nuclear_norm=float(s.sum()) if full else math.nan,
      stable_rank=norm*norm/(s[0]*s[0]) if s[0]>EPS else math.nan,
      top1_energy=float(energy[:1].sum()/norm**2),top5_energy=float(energy[:5].sum()/norm**2),
      top10_energy=float(energy[:10].sum()/norm**2),topk_energy_capture=capture,
      effective_rank=erank,effective_rank_fraction=erank/rank if full else math.nan,
      energy_effective_rank=e_erank,
      spectral_entropy_sigma=math.log(erank) if full else math.nan,
      spectral_entropy_energy=math.log(e_erank) if full else math.nan,
      rank90_fraction=(float((np.searchsorted(cdf,.90)+1)/rank) if full else math.nan),
      rank95_fraction=(float((np.searchsorted(cdf,.95)+1)/rank) if full else math.nan),
      rank99_fraction=(float((np.searchsorted(cdf,.99)+1)/rank) if full else math.nan),
      spectral_energy_curve=curve,normalized_weight_quantiles=wq)


# Principal angles: Golub & Van Loan, Matrix Computations (classical).
# Motivation for weight-space subspaces: Chen et al., COLM 2026,
# https://arxiv.org/abs/2608.07786 . Only meaningful in shared coordinates.
def basis(matrix:torch.Tensor,rank:int,device:str='cpu',max_exact_elements:int=3_000_000):
    if matrix.ndim!=2 or min(matrix.shape)<2: return None,'not_matrix'
    if not torch.isfinite(matrix).all(): raise ValueError('nonfinite subspace matrix')
    if float(torch.linalg.vector_norm(matrix))<=EPS: return None,'zero_matrix'
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    k=min(rank,min(matrix.shape))
    if k<1: return None,'disabled'
    if matrix.numel()<=max_exact_elements or k==min(matrix.shape):
        u=torch.linalg.svd(matrix.to(device=device,dtype=torch.float64),full_matrices=False).U[:,:k]
        return u.detach().cpu(), 'exact_svd'
    kk=min(k,min(matrix.shape)-1)
    # SciPy svds returns largest kk when which='LM'; singular vectors sorted.
    u,s,_=svds(matrix.detach().cpu().numpy(),k=kk,which='LM',v0=np.ones(min(matrix.shape)))
    u=u[:,np.argsort(s)[::-1]].copy()
    return torch.from_numpy(u.astype(np.float64)), 'topk_approx'


# Principal angle-based chordal/projection distances: classical Grassmannian
# geometry. Undefined across mismatched coordinate spaces; never remap silently.
def subspaces(a,b)->dict:
    u,st_a=a; v,st_b=b
    if u is None or v is None: return dict(subspace_status=f'{st_a}|{st_b}')
    if u.shape[0]!=v.shape[0] or u.shape[1]!=v.shape[1]: return dict(subspace_status='incompatible_coordinate_or_rank')
    vals=torch.linalg.svdvals(u.T@v).clamp(0.,1.)
    rank=u.shape[1]
    return dict(subspace_status=f'{st_a}|{st_b}',subspace_rank=rank,
       subspace_mean_cos2=float(vals.square().mean()),subspace_mean_cos=float(vals.mean()),
       subspace_chordal_normalized=float(torch.sqrt((1-vals.square()).clamp_min(0).sum())/math.sqrt(rank)),
       subspace_angle_mean_rad=float(torch.acos(vals).mean()))


# 1D Wasserstein distance: classical optimal transport, e.g. Villani (2009),
# Optimal Transport: Old and New. The standardized quantile curve is a derived
# structural weight-only fingerprint; not a reproduction of TRACE/COLM 2026.
def quantile_w1(qa:list[float],qb:list[float])->float:
    a=np.asarray(qa,float); b=np.asarray(qb,float)
    if a.shape!=b.shape or a.ndim!=1 or a.size<2: raise ValueError('quantile shape mismatch')
    return float(np.trapezoid(np.abs(a-b),dx=1/(a.size-1))) if hasattr(np,'trapezoid') else float(np.trapz(np.abs(a-b),dx=1/(a.size-1)))


# PAPER-EXACT METRIC (Eq. 2–5): Chen, Shang & Liu (2026),
# Who Built This Model? Tracing LLM Lineage via Spectral Fingerprints in Weight Space,
# COLM 2026, https://arxiv.org/abs/2608.07786
# Authors' implementation:
# https://github.com/OPTML-Group/LLM-Biometrics/blob/main/llm_biometrics/trace.py
# A per-component profile is t(W_l)=sqrt(tr(W_l^T W_l))=||W_l||_F.
# Resample each profile to max(layer counts), zscore, take Pearson r per
# component, and average valid component scores. Unlike the older derived
# energy-CDF L1 metric, this implements their Eq. (5) on supplied profiles.
# EXCEPTION: only weights/components chosen by this framework are included.
# Authors' original code expects distinct transformer layer projections;
# our profile extraction excludes generic norm/bias/embeddings and may differ
# for non-standard architectures, which requires reporting component coverage.
def paper_trace_similarity(
    fingerprint_a: dict[str, list[float] | np.ndarray],
    fingerprint_b: dict[str, list[float] | np.ndarray],
) -> dict:
    """TRACE Eq. (5), scalar and component-level Pearson correlations.

    Undefined for shared components with fewer than 2 layers, nonfinite values,
    or constant magnitude across depth. Returned NaN is not a zero similarity.
    """
    scores: dict[str, float] = {}
    shared = sorted(set(fingerprint_a) & set(fingerprint_b))
    for component in shared:
        a = np.asarray(fingerprint_a[component], dtype=np.float64)
        b = np.asarray(fingerprint_b[component], dtype=np.float64)
        if a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < 2:
            continue
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        length = max(len(a), len(b))
        axis = np.linspace(0., 1., length)
        aa = np.interp(axis, np.linspace(0., 1., len(a)), a)
        bb = np.interp(axis, np.linspace(0., 1., len(b)), b)
        asd, bsd = np.std(aa), np.std(bb)
        if asd < 1e-10 or bsd < 1e-10:
            continue
        az = (aa - aa.mean()) / asd
        bz = (bb - bb.mean()) / bsd
        score = np.dot(az, bz) / length
        scores[component] = float(np.clip(score, -1., 1.))
    return {
        'trace_similarity': float(np.mean(list(scores.values()))) if scores else math.nan,
        'trace_n_components': len(scores),
        'trace_per_component': scores,
    }


# PAPER-EXACT PER-LAYER METRIC (Eq. 7–8): Chen, Shang & Liu (2026), COLM.
# https://github.com/OPTML-Group/LLM-Biometrics/blob/main/llm_biometrics/subspace.py
# For corresponding subspaces (same coordinate system), take J smallest
# principal cosines, NOT the mean over all k cosines. Aggregation over the
# K least-aligned layers must be done separately by the analysis caller.
def paper_subspace_layer_alignment(a: torch.Tensor, b: torch.Tensor, j: int = 3) -> float:
    """Paper's mean of J least-aligned principal directions, Eq. (8)."""
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError('TRACE subspace comparisons require aligned shapes')
    if not 1 <= j:
        raise ValueError('j must be positive')
    aa = a.to(torch.float64)
    bb = b.to(torch.float64)
    # Matrices are already orthonormal singular-vector bases.
    cosines = torch.linalg.svdvals(aa.T @ bb).clamp(0., 1.)
    return float(torch.sort(cosines)[0][:min(j,cosines.numel())].mean())


# PAPER-EXACT HIERARCHICAL AGGREGATION: Chen, Shang & Liu (2026), Alg. A2.
# Component layer scores are reduced to n_bottom least-aligned layers,
# then equally weighted across components, not over all layers at once.
def paper_subspace_hierarchical_alignment(
    layer_scores: dict[str, list[float] | np.ndarray], n_bottom: int = 3
) -> dict:
    """Reduce per-layer subspace scores exactly as Alg. A2."""
    if n_bottom < 1:
        raise ValueError('n_bottom must be positive')
    per = {}
    for component, values in sorted(layer_scores.items()):
        scores = np.asarray(values, dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size:
            per[component] = float(np.sort(scores)[:min(n_bottom,scores.size)].mean())
    return {'subspace_similarity': float(np.mean(list(per.values()))) if per else math.nan,
            'subspace_n_components': len(per),'subspace_per_component': per}
