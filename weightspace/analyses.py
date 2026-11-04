"""Weight-only experiments A/B. No forward passes, labels, gradients or predictions."""
from __future__ import annotations
from collections import defaultdict
from contextlib import ExitStack
from itertools import product
import math
import logging
import numpy as np
import pandas as pd
import torch
from .checkpoints import Checkpoint, WeightReader, audit_alignment, get_delta, identify, SPECTRAL_COMPONENTS, config_metadata, head_views
from .metrics import VectorAccumulator, drift, spectral, basis, subspaces, topk_overlap, quantile_w1, paper_trace_similarity, paper_subspace_layer_alignment, paper_subspace_hierarchical_alignment, EPS
LOG=logging.getLogger(__name__)


def common(cp: Checkpoint):
    return dict(checkpoint=cp.uid,trained_path=str(cp.path),backbone=cp.backbone,
                language=cp.language,entity=cp.entity,fold=cp.fold)


def scope_rows(info:dict,metric:dict,level:str='tensor',**extra)->dict:
    return dict(level=level,**info,**metric,**extra)


def describe_checkpoint(cp:Checkpoint,base_path,settings,base_spectral_cache:dict|None=None,*,compute_drift:bool=True,compute_spectra:bool=True):
    """A1: compute delta drift; build separate spectral fingerprints for base/final/delta."""
    drift_rows=[]; spectra=[]
    aggregate=defaultdict(lambda: dict(n=0,ds=0.,bs=0.,fs=0.,dot=0.,l1=0.,maximum=0.))
    head_config=config_metadata(base_path)
    with WeightReader(cp.path) as r,WeightReader(base_path) as b:
        keys,status=audit_alignment(r,b,settings.min_aligned_fraction)
        LOG.info('A1 %s matched %.2f%% of encoder elements, %d tensors',cp.uid,100*status['element_coverage'],len(keys))
        layers=[identify(k,tuple(r.get(k).shape))[0] for k in keys]
        recognized_elements=sum(r.get(k).numel() for k in keys
            if identify(k,tuple(r.get(k).shape))[1] not in ('other','norm','embeddings'))
        recognized_fraction=recognized_elements/status['matched_elements'] if status['matched_elements'] else 0.
        if recognized_fraction<.35:
            LOG.warning('Only %.2f%% of encoder elements are recognized attention/FFN projections for %s; spectral pair comparisons may be unavailable',100*recognized_fraction,cp.uid)
        maxlayer=max((z for z in layers if z is not None),default=0)
        for key in keys:
            fin,orig,d=get_delta(r,b,key)
            layer,comp,_=identify(key,tuple(fin.shape))
            loc=dict(**common(cp),original_path=str(base_path),layer=layer,component=comp,tensor=key,
                n_aligned_elements=status['matched_elements'],element_coverage=status['element_coverage'],recognized_projection_fraction=recognized_fraction)
            if compute_drift:
                stats=drift(fin,orig)
                drift_rows.append(scope_rows(loc,stats))
                # Per-head comparisons for standard BERT-family projection matrices.
                for (head_i,wh),(head_j,bh) in zip(head_views(fin,comp,head_config),head_views(orig,comp,head_config)):
                    if head_i!=head_j:raise ValueError('Attention head index mismatch')
                    drift_rows.append(scope_rows({**loc,'tensor':key+f'::head{head_i}','head_index':head_i},
                                                  drift(wh,bh),level='head'))
            if compute_spectra and d.ndim==2 and comp in SPECTRAL_COMPONENTS:
                for source,tensor in [('base',orig),('final',fin),('delta',d)]:
                    cache_key=(cp.backbone,key)
                    if source=='base' and base_spectral_cache is not None and cache_key in base_spectral_cache:
                        metrics=base_spectral_cache[cache_key]
                    else:
                        metrics=spectral(tensor,settings.spectral_mode,settings.max_svd_elements,
                                         settings.topk,settings.device)
                        if source=='base' and base_spectral_cache is not None:
                            base_spectral_cache[cache_key]=metrics
                    spectra.append(scope_rows(loc,metrics,source=source,relative_depth=(layer/maxlayer if layer is not None and maxlayer>0 else None)))
            if compute_drift:
                for scope,lay,section in [('encoder',None,None),('layer',layer,None),('component',None,comp),('layer_component',layer,comp)]:
                    a=aggregate[scope,lay,section]
                    a['n']+=d.numel();a['ds']+=stats['delta_l2']**2;a['bs']+=stats['base_l2']**2
                    a['fs']+=stats['final_l2']**2;a['dot']+=float(torch.dot(fin.flatten(),orig.flatten()))
                    a['l1']+=stats['delta_l1'];a['maximum']=max(a['maximum'],stats['max_abs_delta'])
        if compute_drift:
            for (scope,layer,comp),a in aggregate.items():
                ds=math.sqrt(a['ds']);bs=math.sqrt(a['bs']);fs=math.sqrt(a['fs'])
                drift_rows.append(dict(**common(cp),original_path=str(base_path),level=scope,layer=layer,component=comp,tensor=None,
                    n_parameters=a['n'],n_aligned_elements=status['matched_elements'],element_coverage=status['element_coverage'],recognized_projection_fraction=recognized_fraction,
                    delta_l2=ds,base_l2=bs,final_l2=fs,delta_energy=a['ds'],
                    relative_l2=(ds/bs if bs>EPS else math.nan),delta_rms=ds/math.sqrt(a['n']),
                    delta_mae=a['l1']/a['n'],delta_l1=a['l1'],max_abs_delta=a['maximum'],
                    cosine_final_base=a['dot']/(fs*bs) if fs>EPS and bs>EPS else math.nan))
    return drift_rows,spectra


def compare_related(a:Checkpoint,b:Checkpoint,base_path,settings,subspace_cache:dict|None=None):
    """A2-A4: pair aligned deltas + raw final weights (separate comparison_kind)."""
    rows=[]; aggregators=defaultdict(VectorAccumulator)
    # Scores per layer/component for Chen et al. (COLM 2026), Eq. 8, Alg A2.
    paper_layer_scores=defaultdict(list)
    head_config=config_metadata(base_path)
    with WeightReader(a.path) as ra,WeightReader(b.path) as rb,WeightReader(base_path) as original:
        ka,sa=audit_alignment(ra,original,settings.min_aligned_fraction)
        kb,sb=audit_alignment(rb,original,settings.min_aligned_fraction)
        matching=sorted(set(ka)&set(kb))
        a_els=sum(ra.get(k).numel() for k in ka)
        b_els=sum(rb.get(k).numel() for k in kb)
        covered=sum(ra.get(k).numel() for k in matching)
        fraction=covered/max(a_els,b_els)
        if fraction<settings.min_aligned_fraction:
            raise ValueError(f'Pair has poor matched-element coverage: {a.uid} {b.uid} {fraction:.3f}')
        shared=dict(checkpoint_a=a.uid,checkpoint_b=b.uid,original_path=str(base_path),
            trained_path_a=str(a.path),trained_path_b=str(b.path),backbone=a.backbone,
            language_a=a.language,language_b=b.language,entity_a=a.entity,entity_b=b.entity,
            fold_a=a.fold,fold_b=b.fold,pair_element_coverage=fraction)
        for key in matching:
            fa,ini,da=get_delta(ra,original,key)
            fb,ini2,db=get_delta(rb,original,key)
            if not torch.equal(ini,ini2): raise ValueError('Base tensor changed between pair reads')
            layer,comp,_=identify(key,tuple(fa.shape))
            for kind,ma,mb in [('delta',da,db),('final',fa,fb)]:
                for scope,lay,section,tensor in [('tensor',layer,comp,key),('layer',layer,None,None),
                        ('component',None,comp,None),('layer_component',layer,comp,None),('encoder',None,None,None)]:
                    aggregators[kind,scope,lay,section,tensor].update(ma,mb,settings.sign_floor)
                for (hi,ah),(hj,bh) in zip(head_views(ma,comp,head_config),head_views(mb,comp,head_config)):
                    if hi!=hj:raise ValueError('Attention head index mismatch')
                    aggregators[kind,'head',layer,comp,f'{key}::head{hi}'].update(ah,bh,settings.sign_floor)
            if da.ndim==2 and comp in SPECTRAL_COMPONENTS and settings.subspace_rank>0:
                # Cache singular vectors for this backbone's checkpoints only.
                # Correct shared coordinate system is guaranteed by alignment above.
                if subspace_cache is None:
                    left=basis(da,settings.subspace_rank,settings.device,settings.max_svd_elements)
                    right=basis(db,settings.subspace_rank,settings.device,settings.max_svd_elements)
                else:
                    for ck,tensor in ((a,da),(b,db)):
                        ident=(ck.uid,key)
                        if ident not in subspace_cache:
                            subspace_cache[ident]=basis(tensor,settings.subspace_rank,settings.device,settings.max_svd_elements)
                    left=subspace_cache[a.uid,key]
                    right=subspace_cache[b.uid,key]
                subspace_stats=subspaces(left,right)
                if left[0] is not None and right[0] is not None and left[0].shape==right[0].shape:
                    paper_score=paper_subspace_layer_alignment(left[0],right[0],j=3)
                    subspace_stats['paper_colm26_worst3_cosines']=paper_score
                    if layer is not None:
                        paper_layer_scores[(comp,layer)].append(paper_score)
                rows.append(dict(**shared,level='tensor',layer=layer,component=comp,tensor=key,
                          comparison_kind='delta',metric_family='subspace',**subspace_stats))
            if settings.topk_fraction>0 and da.numel()<=settings.max_topk_elements:
                rows.append(dict(**shared,level='tensor',layer=layer,component=comp,tensor=key,
                          comparison_kind='delta',metric_family='overlap',**topk_overlap(da,db,settings.topk_fraction)))
        for (kind,scope,layer,comp,key),metric in aggregators.items():
            rows.append(dict(**shared,level=scope,layer=layer,component=comp,tensor=key,
                        comparison_kind=kind,metric_family='vector',**metric.result()))
        if paper_layer_scores:
            # Aggregate duplicate matrices in the same layer/component (if any)
            # before selecting least-aligned layers, as in Chen et al. Alg A2.
            per_component=defaultdict(list)
            for (comp,layer),vals in sorted(paper_layer_scores.items()):
                per_component[comp].append(float(np.mean(vals)))
            summary=paper_subspace_hierarchical_alignment(per_component,n_bottom=3)
            rows.append(dict(**shared,level='encoder',layer=None,component=None,
                tensor=None,comparison_kind='delta',metric_family='subspace_paper_colm26',
                paper_colm26_hierarchical_similarity=summary['subspace_similarity'],
                paper_colm26_n_components=summary['subspace_n_components'],
                paper_colm26_component_scores=str(summary['subspace_per_component'])))
    return rows


def _profile(df:pd.DataFrame,source:str,comp:str,column:str,grid_size:int=16):
    """Interpolate depth-aligned intrinsic profiles; not neural alignment."""
    if df.empty or column not in df.columns or 'status' not in df.columns:
        return None
    sub=df[(df.component==comp)&(df.source==source)&df.relative_depth.notna()]
    if column=='spectral_energy_curve':
        sub=sub[(sub.status=='exact') & sub[column].notna()]
    else:
        sub=sub[sub[column].notna()]
    if sub.empty:return None
    pairs=[]
    for r in sub.itertuples(index=False):
        arr=np.asarray(getattr(r,column),dtype=float)
        if arr.ndim==0: arr=arr.reshape(1)
        if np.isfinite(arr).all(): pairs.append((float(r.relative_depth),arr))
    if not pairs: return None
    sizes={v.size for _,v in pairs}
    if len(sizes)!=1: return None
    xs=np.array([p[0] for p in pairs]); ys=np.stack([p[1] for p in pairs])
    ux=np.unique(xs)
    ys=np.stack([ys[xs==x].mean(axis=0) for x in ux])
    if len(ux)==1: return np.repeat(ys,grid_size,axis=0)
    return np.stack([np.interp(np.linspace(0,1,grid_size),ux,ys[:,j]) for j in range(ys.shape[1])],axis=1)


def compare_backbones(a:Checkpoint,b:Checkpoint,spectral_df:pd.DataFrame,drift_df:pd.DataFrame,min_components:int=3):
    """B: exact-paper TRACE plus independent intrinsic fingerprint distances."""
    sa=spectral_df[spectral_df.checkpoint==a.uid];sb=spectral_df[spectral_df.checkpoint==b.uid]
    da=drift_df[(drift_df.checkpoint==a.uid)&(drift_df.level=='tensor')]
    db=drift_df[(drift_df.checkpoint==b.uid)&(drift_df.level=='tensor')]
    row=dict(checkpoint_a=a.uid,checkpoint_b=b.uid,trained_path_a=str(a.path),trained_path_b=str(b.path),
             backbone_a=a.backbone,backbone_b=b.backbone,
             language=a.language,entity=a.entity,fold=1,comparison_kind='heterogeneous_fingerprint')
    for source in ('base','final','delta'):
        # Exact TRACE Eq. (5) on per-layer Frobenius norm fingerprints.
        # Chen, Shang & Liu (COLM 2026), official implementation:
        # https://github.com/OPTML-Group/LLM-Biometrics/blob/main/llm_biometrics/trace.py
        # This is a different measure from our normalized spectral-CDF L1.
        def trace_fingerprint(frame):
            if frame.empty or 'matrix_fro' not in frame.columns:
                return {}
            matching = frame[(frame.source == source) & (frame.layer.notna()) &
                             frame.component.isin(SPECTRAL_COMPONENTS) &
                             (frame.matrix_fro.notna()) &
                             (frame.level == 'tensor')]
            result = {}
            for comp, group in matching.groupby('component'):
                series = group.groupby('layer')['matrix_fro'].mean().sort_index()
                result[comp] = series.to_numpy(dtype=float)
            return result
        trace = paper_trace_similarity(trace_fingerprint(sa), trace_fingerprint(sb))
        row[f'trace_similarity_{source}'] = trace['trace_similarity']
        row[f'trace_distance_{source}'] = (1.-trace['trace_similarity']
            if math.isfinite(trace['trace_similarity']) else math.nan)
        row[f'n_components_trace_{source}'] = trace['trace_n_components']
        for comp, score in trace['trace_per_component'].items():
            row[f'trace_{comp}_{source}'] = score
        for metric,col in [('curve_l1','spectral_energy_curve'),('quantile_w1','normalized_weight_quantiles')]:
            measures=[]
            for comp in SPECTRAL_COMPONENTS:
                x=_profile(sa,source,comp,col);y=_profile(sb,source,comp,col)
                if x is None or y is None or x.shape!=y.shape: continue
                if metric=='curve_l1': measures.append(float(np.abs(x-y).mean()))
                else: measures.append(float(np.mean([quantile_w1(u,v) for u,v in zip(x,y)])))
            row[f'{metric}_{source}']=float(np.mean(measures)) if len(measures)>=min_components else math.nan
            row[f'n_components_{metric}_{source}']=len(measures)
        # Norm profile distance explicitly scale-sensitive, use normalized delta profiles below.
    measures=[]
    for comp in SPECTRAL_COMPONENTS:
        def xprofile(frame):
            t=frame[(frame.component==comp)&frame.layer.notna()]
            if t.empty:return None
            x=t.groupby('layer',as_index=False)['relative_l2'].mean().sort_values('layer').relative_l2.to_numpy()
            if not np.isfinite(x).all():return None
            return np.interp(np.linspace(0,1,16),np.linspace(0,1,len(x)),np.log1p(x))
        x=xprofile(da);y=xprofile(db)
        if x is not None and y is not None: measures.append(float(np.mean(np.abs(x-y))))
    row['adaptation_profile_l1']=float(np.mean(measures)) if len(measures)>=min_components else math.nan
    row['n_components_adaptation']=len(measures)
    row['comparability']=('intrinsic_profiles_only_not_functional_equivalence' if len(measures)>=min_components else 'insufficient_common_components')
    return row


def ranks(df:pd.DataFrame):
    """Independent per-metric ranks; never average incomparable distances."""
    if df.empty:return df
    for metric in ['trace_distance_base','trace_distance_final','trace_distance_delta',
                   'curve_l1_base','curve_l1_final','curve_l1_delta',
                   'quantile_w1_base','quantile_w1_final','quantile_w1_delta','adaptation_profile_l1']:
        if metric in df:
            df['rank_'+metric]=df.groupby(['language','entity'])[metric].rank(method='min',ascending=True)
            df['n_pairs_'+metric]=df.groupby(['language','entity'])[metric].transform('count').astype('int64')
    return df


# Classical balanced two-factor ANOVA decomposition of sums of squares; only
# descriptive weight-vector decomposition, NOT F-tests with one cell replicate.
# Fisher (1925), Statistical Methods for Research Workers; contextual use of
# task deltas: Ilharco et al., ICLR 2023, https://openreview.net/forum?id=6t0Kwf8-jrj
# A complete balanced L×E grid is required. Interaction absorbs train noise.
def factorial(checkpoints:list[Checkpoint],base_path,settings):
    ls=sorted({x.language for x in checkpoints});es=sorted({x.entity for x in checkpoints})
    if len(ls)<2 or len(es)<2:return []
    lookup={(c.language,c.entity):c for c in checkpoints}
    if len(lookup)!=len(checkpoints) or any((l,e) not in lookup for l,e in product(ls,es)):
        LOG.warning('Factorial skipped for %s: full balanced language × entity grid required',checkpoints[0].backbone)
        return []
    rows=[]; accumulated=defaultdict(lambda:np.zeros(5,float))
    with ExitStack() as stack:
        b=stack.enter_context(WeightReader(base_path))
        readers={(l,e):stack.enter_context(WeightReader(lookup[l,e].path)) for l,e in product(ls,es)}
        keys=[]
        for r in readers.values():
            current,_=audit_alignment(r,b,settings.min_aligned_fraction)
            keys.append(set(current))
        commonkeys=sorted(set.intersection(*keys))
        common_elements=sum(b.get(k).numel() for k in commonkeys)
        matched_cells=[sum(r.get(k).numel() for k in keyset)
                       for r,keyset in zip(readers.values(),keys)]
        coverage=common_elements/max(matched_cells)
        if coverage<settings.min_aligned_fraction:
            raise ValueError(f'Factorial lacks common coordinates across all cells: {coverage:.4%}')
        for key in commonkeys:
            orig=b.get(key)  # hold only one parameter tensor from original
            layer,comp,_=identify(key,tuple(orig.shape))
            # Cache ONE tensor from each checkpoint for this key. Even huge
            # embeddings are read once, not once per 30k-element slice.
            cached={(l,e):readers[l,e].get(key).reshape(-1) for l,e in product(ls,es)}
            ss=np.zeros(5,float)
            for start in range(0,orig.numel(),settings.factorial_chunk_elements):
                end=min(orig.numel(),start+settings.factorial_chunk_elements)
                w0=orig.reshape(-1)[start:end].to(torch.float64)
                z=torch.stack([torch.stack([(cached[l,e][start:end].to(torch.float64)-w0)
                                 for e in es]) for l in ls])
                if not torch.isfinite(z).all(): raise ValueError(f'Nonfinite factorial input {key}')
                mu=z.mean((0,1)); lf=z.mean(1)-mu;ef=z.mean(0)-mu
                interaction=z-mu-lf[:,None,:]-ef[None,:,:]
                ss+=np.array([len(es)*float(lf.square().sum()),len(ls)*float(ef.square().sum()),
                              float(interaction.square().sum()),float((z-mu).square().sum()),z.shape[-1]],float)
            cached.clear()
            if not np.isclose(ss[:3].sum(),ss[3],rtol=2e-9,atol=1e-10):
                raise ValueError(f'Factorial identity broken for {key}: {ss}')
            for level,lay,section,tensor in [('tensor',layer,comp,key),('layer',layer,None,None),
                                               ('component',None,comp,None),('encoder',None,None,None)]:
                if level=='tensor':
                    rows.append(_factorial_row(checkpoints[0].backbone,ls,es,level,lay,section,tensor,ss))
                else:accumulated[level,lay,section]+=ss
        for (level,layer,comp),ss in accumulated.items():
            rows.append(_factorial_row(checkpoints[0].backbone,ls,es,level,layer,comp,None,ss))
    return rows


def _factorial_row(backbone,ls,es,level,layer,comp,key,ss):
    a,b,c,total,n=ss
    return dict(backbone=backbone,fold=1,n_languages=len(ls),n_entities=len(es),
                level=level,layer=layer,component=comp,tensor=key,n_parameters=int(n),
                ss_language=a,ss_entity=b,ss_interaction=c,ss_total=total,
                fraction_language=a/total if total>EPS else math.nan,
                fraction_entity=b/total if total>EPS else math.nan,
                fraction_interaction=c/total if total>EPS else math.nan,
                interpretation='descriptive_one_checkpoint_per_cell_no_inference')

# Classical graph-distance summary; descriptive *mean pairwise distance*, not
# a proof of common functions or an exact shortest-path centrality algorithm.
# Structural model-lineage motivation: Chen et al., COLM 2026,
# https://arxiv.org/abs/2608.07786
# We require comparable nonmissing metric values for ALL model pairs in a stratum.
def model_distance_summary(pair_df:pd.DataFrame)->pd.DataFrame:
    if pair_df.empty:return pd.DataFrame()
    rows=[]
    dist_metrics=('trace_distance_delta','trace_distance_final','curve_l1_delta','quantile_w1_delta','adaptation_profile_l1',
                  'curve_l1_final','quantile_w1_final')
    for (language,entity),group in pair_df.groupby(['language','entity']):
        models=sorted(set(group.backbone_a)|set(group.backbone_b))
        for metric in dist_metrics:
            if metric not in group:continue
            for model in models:
                sub=group[(group.backbone_a==model)|(group.backbone_b==model)]
                vals=[]
                for q in sub.itertuples(index=False):
                    v=getattr(q,metric)
                    if np.isfinite(v):
                        other=q.backbone_b if q.backbone_a==model else q.backbone_a
                        vals.append((other,float(v)))
                n_expected=len(models)-1
                values=np.array([v for _,v in vals],float)
                if values.size:
                    nearest=min(vals,key=lambda p:(p[1],p[0]))
                    furthest=max(vals,key=lambda p:(p[1],p[0]))
                else:
                    nearest=(None,math.nan);furthest=(None,math.nan)
                rows.append(dict(language=language,entity=entity,backbone=model,
                    metric=metric,n_neighbors=len(vals),n_possible_neighbors=n_expected,
                    complete_comparison=len(vals)==n_expected,
                    mean_distance=float(values.mean()) if len(vals) else math.nan,
                    median_distance=float(np.median(values)) if len(vals) else math.nan,
                    nearest_backbone=nearest[0],nearest_distance=nearest[1],
                    furthest_backbone=furthest[0],furthest_distance=furthest[1],
                    interpretation='intrinsic_fingerprint_distance_not_functional_similarity'))
    df=pd.DataFrame(rows)
    if df.empty:return df
    df['rank_most_central']=float('nan')
    valid=df.complete_comparison
    if valid.any():
        df.loc[valid,'rank_most_central']=df[valid].groupby(['language','entity','metric'])['mean_distance'].rank(method='min')
    return df


def rank_related_encoder_pairs(frame:pd.DataFrame,mode:str)->pd.DataFrame:
    """Rank A2/A3 pairs only within a common backbone and entity or language."""
    if frame.empty:return frame
    frame=frame.copy()
    frame['rank_delta_cosine']=float('nan')
    mask=(frame.level=='encoder')&(frame.comparison_kind=='delta')&(frame.metric_family=='vector')
    grouping=['backbone','entity_a'] if mode=='cross_language' else ['backbone','language_a']
    if mask.any() and set(grouping)<=set(frame.columns):
        frame.loc[mask,'rank_delta_cosine']=frame[mask].groupby(grouping)['cosine'].rank(method='min',ascending=False)
    return frame
