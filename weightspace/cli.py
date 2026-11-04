"""CLI orchestrating checkpoint-only A1-A5 and B1-B5 experiments."""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import combinations
import logging
from pathlib import Path
import math
import pandas as pd
import torch
from .checkpoints import discover,WeightReader,config_metadata,audit_alignment
from .analyses import describe_checkpoint,compare_related,compare_backbones,ranks,factorial,model_distance_summary,rank_related_encoder_pairs
from .storage import write_parquet,load_performance,with_performance
from .performance import summarize_f1
LOG=logging.getLogger('weightspace')


def parser():
    p=argparse.ArgumentParser(description='MultiClinAI weight-only encoder analysis (1 saved fold)')
    p.add_argument('--originals',type=Path,default=Path('~/ner_lab/TFG_Aziz/models').expanduser())
    p.add_argument('--trained',type=Path,default=Path('/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weights'))
    p.add_argument('--output',type=Path,default=Path('/mnt/fscratch/users/gaias_001_uma/fgd/ner_lab/TFG_Aziz/weight_analysis'))
    p.add_argument('--performance',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--spectral-mode',choices=['auto','exact','topk','skip'],default='auto')
    p.add_argument('--max-svd-elements',type=int,default=3_000_000)
    p.add_argument('--topk',type=int,default=32)
    p.add_argument('--subspace-rank',type=int,default=8)
    p.add_argument('--topk-fraction',type=float,default=.01)
    p.add_argument('--max-topk-elements',type=int,default=2_000_000)
    p.add_argument('--sign-floor',type=float,default=1e-8)
    p.add_argument('--min-fingerprint-components',type=int,default=3)
    p.add_argument('--min-aligned-fraction',type=float,default=.99)
    p.add_argument('--factorial-chunk-elements',type=int,default=30_000)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--deep-dry-run',action='store_true',help='Read tensor shapes/keys to audit alignment without SVD or Parquet')
    p.add_argument('--verbose',action='store_true')
    return p


def run(argv=None):
    args=parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if not 0<args.min_aligned_fraction<=1:raise ValueError('min-aligned-fraction must be in (0,1]')
    if not 0<=args.topk_fraction<=1:raise ValueError('topk-fraction outside [0,1]')
    if args.topk<1 or args.subspace_rank<0 or args.min_fingerprint_components<1 or args.max_svd_elements<1 or args.factorial_chunk_elements<1:
        raise ValueError('invalid rank / matrix threshold / factorial chunk')
    if args.device=='cuda' and not torch.cuda.is_available() and not (args.dry_run or args.deep_dry_run):
        raise RuntimeError('CUDA explicitly requested but not available')
    bases,cps=discover(args.originals.expanduser().resolve(),args.trained.expanduser().resolve())
    LOG.info('%s originals, %s trained checkpoints; languages=%s entities=%s',len(bases),len(cps),
             sorted({x.language for x in cps}),sorted({x.entity for x in cps}))
    for cp in cps:
        meta=config_metadata(cp.path); base_meta=config_metadata(bases[cp.backbone])
        if meta.get('model_type') and base_meta.get('model_type') and meta['model_type']!=base_meta['model_type']:
            raise ValueError(f'Config architecture mismatch: {cp.uid} base={base_meta["model_type"]} fin={meta["model_type"]}')
        for field in ('hidden_size','num_hidden_layers','num_attention_heads','vocab_size'):
            if meta.get(field) is not None and base_meta.get(field) is not None and meta[field]!=base_meta[field]:
                raise ValueError(f'Base/final config {field} differs for {cp.uid}: {base_meta[field]} vs {meta[field]}')
        LOG.info('model=%s original=%s config-type=%s',cp.uid,bases[cp.backbone],meta.get('model_type'))
        if args.deep_dry_run:
            with WeightReader(cp.path) as r,WeightReader(bases[cp.backbone]) as b:
                keys,status=audit_alignment(r,b,args.min_aligned_fraction)
                LOG.info('  matched %d tensors, element_coverage=%.5f, missing=%s mismatch=%s',len(keys),status['element_coverage'],status['missing_keys'][:4],status['mismatched_keys'][:4])
    if args.dry_run or args.deep_dry_run:
        LOG.info('DRY-RUN complete. No files written.');return
    # Fail FAST on Parquet dependency before multi-hour analysis.
    try:import pyarrow
    except ImportError as e:raise RuntimeError('pyarrow unavailable; install requirements before running') from e
    scores=load_performance(args.performance)
    out=args.output.expanduser().resolve()
    all_drift=[];all_spectra=[]
    base_spectral_cache={}
    for idx,cp in enumerate(cps,1):
        LOG.info('Extracting A1 %d/%d %s',idx,len(cps),cp.uid)
        d,s=describe_checkpoint(cp,bases[cp.backbone],args,base_spectral_cache)
        all_drift.extend(d);all_spectra.extend(s)
    drift=pd.DataFrame(all_drift); spectra=pd.DataFrame(all_spectra)
    drift_with_f1=with_performance(drift,scores)
    write_parquet(drift_with_f1,out/'01_pretraining_drift.parquet')
    write_parquet(spectra,out/'02_spectral_fingerprints.parquet')
    del all_drift,all_spectra
    buckets=defaultdict(list)
    subspace_cache={}
    cached_backbone=None
    for idx,(a,b) in enumerate(combinations(cps,2),1):
        if a.backbone != b.backbone:continue
        if a.backbone!=cached_backbone:
            subspace_cache.clear()
            cached_backbone=a.backbone
        if a.language==b.language and a.entity==b.entity:continue
        category=('cross_language' if a.entity==b.entity else
                  'cross_entity' if a.language==b.language else 'cross_language_entity')
        LOG.info('Comparing A %s %s <-> %s',category,a.uid,b.uid)
        buckets[category].extend(compare_related(a,b,bases[a.backbone],args,subspace_cache))
    for kind,name in [('cross_language','03_cross_language.parquet'),
                      ('cross_entity','04_cross_entity.parquet'),
                      ('cross_language_entity','05_cross_language_entity.parquet')]:
        frame=pd.DataFrame(buckets.pop(kind,[]))
        if kind in ('cross_language','cross_entity'):
            frame=rank_related_encoder_pairs(frame,kind)
        write_parquet(with_performance(frame,scores),out/name)
    # By design we have no 06_cross_fold.parquet.
    cross=[]
    for a,b in combinations(cps,2):
        if a.backbone==b.backbone or a.language!=b.language or a.entity!=b.entity:continue
        cross.append(compare_backbones(a,b,spectra,drift,args.min_fingerprint_components))
    cross_df=ranks(pd.DataFrame(cross))
    cross_with_f1=with_performance(cross_df,scores)
    write_parquet(cross_with_f1,out/'07_cross_backbone.parquet')
    write_parquet(model_distance_summary(cross_df),out/'11_model_centrality.parquet')
    if not scores.empty:
        associations=summarize_f1(drift_with_f1,cross_with_f1)
        write_parquet(associations,out/'10_performance_associations.parquet')
    if not cross_df.empty:
        metrics=[c for c in ('trace_distance_base','trace_distance_final','trace_distance_delta',
                 'curve_l1_base','curve_l1_final','curve_l1_delta',
                 'quantile_w1_base','quantile_w1_final','quantile_w1_delta','adaptation_profile_l1') if c in cross_df]
        aggregated=cross_df.groupby(['backbone_a','backbone_b','entity'],as_index=False).agg(
            n_languages=('language','nunique'),**{m:(m,'mean') for m in metrics})
        for col in metrics:
            aggregated['rank_'+col]=aggregated.groupby('entity')[col].rank(method='min')
        aggregated['aggregation']='mean_across_languages_descriptive_not_independent'
        write_parquet(aggregated,out/'09_cross_backbone_by_entity.parquet')
    factor=[]
    for base in sorted(bases):
        subset=[x for x in cps if x.backbone==base]
        factor.extend(factorial(subset,bases[base],args))
    write_parquet(factor,out/'08_language_entity_decomposition.parquet')
    LOG.info('Complete. Only fold 1 exists; cross-fold weight stability deliberately NOT computed.')


def main():run()
if __name__=='__main__':main()
