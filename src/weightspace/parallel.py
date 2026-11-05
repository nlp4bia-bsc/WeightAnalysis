"""Resumable, sharded SLURM execution of the unchanged scientific analyses.

Each final analysis has exactly one output Parquet. Intermediate shard Parquets
are atomic, private to that analysis, and deleted only on successful finalization.
The traditional `python -m weightspace` pipeline remains available unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .analyses import (
    compare_backbones,
    compare_related,
    describe_checkpoint,
    factorial,
    model_distance_summary,
    rank_related_encoder_pairs,
    ranks,
)
from .checkpoints import config_metadata, discover
from .cli import parser as original_parser
from .performance import summarize_f1
from .storage import METADATA, load_performance, with_performance, write_parquet

LOG = logging.getLogger('weightspace.parallel')
NAMES = {
    '01':'01_pretraining_drift.parquet',
    '02':'02_spectral_fingerprints.parquet',
    '03':'03_cross_language.parquet',
    '04':'04_cross_entity.parquet',
    '05':'05_cross_language_entity.parquet',
    '07':'07_cross_backbone.parquet',
    '08':'08_language_entity_decomposition.parquet',
    '09':'09_cross_backbone_by_entity.parquet',
    '10':'10_performance_associations.parquet',
    '11':'11_model_centrality.parquet',
}
PARTITIONED = {'01','02','03','04','05','07','08'}
PAIR_KINDS = {'03':'cross_language', '04':'cross_entity', '05':'cross_language_entity'}


def parser():
    p = original_parser()
    p.description = 'Parallel MultiClinAI analysis: one final Parquet per experiment'
    p.add_argument('--analysis', choices=sorted(NAMES), required=True)
    p.add_argument('--shard', type=int, help='Zero-based shard index, defaults to SLURM_ARRAY_TASK_ID')
    p.add_argument('--splits-per-backbone', type=int, default=3,
                   help='For A2-A4, split each backbone pair list into this many shards')
    p.add_argument('--finalize', action='store_true', help='Merge all successful shards atomically')
    p.add_argument('--stage-dir', type=Path, help='Optional staging directory (default OUTPUT/.weightspace_stage_v4)')
    p.add_argument('--force', action='store_true', help='Recompute a completed shard; normally existing shards are reused')
    return p


def shard_count(analysis, bases, checkpoints, splits):
    if analysis in {'01','02','08'}:
        return len(bases)
    if analysis in PAIR_KINDS:
        return len(bases)*splits
    if analysis == '07':
        return len({(c.language,c.entity) for c in checkpoints})
    raise ValueError(f'{analysis} is not sharded')


def _validate_cp_configs(checkpoints, bases):
    for cp in checkpoints:
        fin = config_metadata(cp.path); base = config_metadata(bases[cp.backbone])
        if fin.get('model_type') and base.get('model_type') and fin['model_type'] != base['model_type']:
            raise ValueError(f'Architecture mismatch for {cp.uid}')
        for key in ('hidden_size','num_hidden_layers','num_attention_heads','vocab_size'):
            if fin.get(key) is not None and base.get(key) is not None and fin[key] != base[key]:
                raise ValueError(f'Original/final {key} mismatch for {cp.uid}')


def _part_path(stage_root, analysis, i):
    return stage_root/analysis/f'part-{i:04d}.parquet'


def _configuration_signature(args):
    """Stable signature to prevent mixing incompatible shard configurations."""
    source = args.performance.expanduser().resolve() if args.performance else None
    source_state = None
    if source:
        stat = source.stat()
        source_state = (str(source),stat.st_size,stat.st_mtime_ns)
    root=Path(__file__).resolve().parent
    # Ensures cached shards are never combined across scientific code revisions.
    codehash=hashlib.sha256()
    for filename in ('analyses.py','metrics.py','checkpoints.py','performance.py',
                     'storage.py','parallel.py'):
        codehash.update(filename.encode())
        codehash.update((root/filename).read_bytes())
    payload = dict(code_revision=codehash.hexdigest(),
                   analysis=args.analysis, originals=str(args.originals),
                   trained=str(args.trained), performance=source_state,
                   splits_per_backbone=args.splits_per_backbone,
                   spectral_mode=args.spectral_mode,max_svd_elements=args.max_svd_elements,
                   topk=args.topk,subspace_rank=args.subspace_rank,
                   topk_fraction=args.topk_fraction,max_topk_elements=args.max_topk_elements,
                   sign_floor=args.sign_floor,min_fingerprint_components=args.min_fingerprint_components,
                   min_aligned_fraction=args.min_aligned_fraction,
                   factorial_chunk_elements=args.factorial_chunk_elements)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def _write_shard(data, path, signature):
    frame=pd.DataFrame(data)
    if not frame.empty:
        write_parquet(frame,path,metadata={'execution':'parallel_shard','final_result':'false', 'config_signature':signature})
    else:
        # A valid empty Parquet is a completion marker, NOT a standalone sidecar.
        _empty_shard(path,signature)


def _empty_shard(path,signature):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        tab=pa.table({'_empty':pa.array([],type=pa.int64())})
        tab=tab.replace_schema_metadata({b'config_signature':signature.encode()})
        pq.write_table(tab, temporary)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info('Wrote empty shard %s (valid 0-row Parquet)',path)


def _read_required(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required input Parquet missing: {path}. Complete its SLURM finalize job first.')
    return pd.read_parquet(path)


def _run_shard(args,bases,checkpoints,stage_root,scores):
    analysis=args.analysis
    n=shard_count(analysis,bases,checkpoints,args.splits_per_backbone)
    idx=args.shard
    if idx is None:
        raw=os.environ.get('SLURM_ARRAY_TASK_ID')
        if raw is None:raise ValueError('--shard or SLURM_ARRAY_TASK_ID required')
        idx=int(raw)
    if idx<0 or idx>=n:raise ValueError(f'Shard index {idx} outside [0,{n-1}] for analysis {analysis}')
    target=_part_path(stage_root,analysis,idx)
    signature=_configuration_signature(args)
    if target.exists() and not args.force:
        pf=pq.ParquetFile(target)  # validate rather than silently trusting corrupt files
        old=(pf.schema_arrow.metadata or {}).get(b'config_signature',b'').decode()
        if old!=signature:
            raise ValueError(f'Staged shard {target} belongs to another configuration. '
                             'Delete the old staging directory or restore the original options; '
                             'never combine scientifically inconsistent shards.')
        LOG.info('Already complete, skipping %s',target)
        return
    ordered=sorted(bases)
    group=[]
    if analysis in {'01','02','08'}:
        backbone=ordered[idx]
        group=[c for c in checkpoints if c.backbone==backbone]
        LOG.info('%s part %d/%d backbone=%s (%d checkpoints)',analysis,idx+1,n,backbone,len(group))
        _validate_cp_configs(group,bases)
        if analysis in ('01','02'):
            rows=[]; base_cache={}
            for k,cp in enumerate(group,1):
                LOG.info('A1 %s shard=%d checkpoint=%d/%d %s',analysis,idx,k,len(group),cp.uid)
                drift,spectra=describe_checkpoint(cp,bases[backbone],args,base_cache,
                    compute_drift=analysis=='01',compute_spectra=analysis=='02')
                rows.extend(drift if analysis=='01' else spectra)
            frame=pd.DataFrame(rows)
            if analysis=='01':frame=with_performance(frame,scores)
            _write_shard(frame,target,signature)
            return
        if analysis=='08':
            _write_shard(factorial(group,bases[backbone],args),target,signature)
            return
    elif analysis in PAIR_KINDS:
        back_idx,part=divmod(idx,args.splits_per_backbone)
        backbone=ordered[back_idx]
        group=[c for c in checkpoints if c.backbone==backbone]
        _validate_cp_configs(group,bases)
        kind=PAIR_KINDS[analysis]
        selected=[]
        for a,b in combinations(group,2):
            pairkind=('cross_language' if a.entity==b.entity else
                      'cross_entity' if a.language==b.language else 'cross_language_entity')
            if pairkind==kind:selected.append((a,b))
        selected=selected[part::args.splits_per_backbone]
        LOG.info('%s part=%d/%d backbone=%s pairs=%d',analysis,idx+1,n,backbone,len(selected))
        rows=[]; basis_cache={}
        for k,(a,b) in enumerate(selected,1):
            LOG.info('%s shard=%d pair=%d/%d %s <-> %s',analysis,idx,k,len(selected),a.uid,b.uid)
            rows.extend(compare_related(a,b,bases[backbone],args,basis_cache))
        frame=with_performance(pd.DataFrame(rows),scores)
        _write_shard(frame,target,signature)
        return
    elif analysis=='07':
        strata=sorted({(c.language,c.entity) for c in checkpoints})
        language,entity=strata[idx]
        group=[c for c in checkpoints if (c.language,c.entity)==(language,entity)]
        spectra=_read_required(args.output/NAMES['02'])
        drift=_read_required(args.output/NAMES['01'])
        spec_by={k:g for k,g in spectra.groupby('checkpoint',sort=False)}
        drift_by={k:g for k,g in drift.groupby('checkpoint',sort=False)}
        LOG.info('07 stratum %d/%d %s/%s checkpoints=%d',idx+1,n,language,entity,len(group))
        rows=[]
        for a,b in combinations(group,2):
            if a.backbone==b.backbone:continue
            if a.uid not in spec_by or b.uid not in spec_by or a.uid not in drift_by or b.uid not in drift_by:
                raise ValueError(f'Missing A1 fingerprints for pair {a.uid} vs {b.uid}')
            rows.append(compare_backbones(a,b,
                pd.concat((spec_by[a.uid],spec_by[b.uid]),ignore_index=True),
                pd.concat((drift_by[a.uid],drift_by[b.uid]),ignore_index=True),
                args.min_fingerprint_components))
        frame=ranks(pd.DataFrame(rows))
        _write_shard(with_performance(frame,scores),target,signature)
        return
    raise ValueError(analysis)


def _rank_lookup(paths,mode):
    """Recreate rankings only AFTER seeing all pair rows in the backbone."""
    small=[]
    needed=['checkpoint_a','checkpoint_b','level','comparison_kind','metric_family',
            'cosine','backbone','entity_a','language_a']
    for path in paths:
        file=pq.ParquetFile(path)
        if file.metadata.num_rows==0:continue
        have=set(file.schema_arrow.names)
        if not set(needed)<=have:raise ValueError(f'Shard missing encoder ranking columns: {path}')
        frame=pq.read_table(path,columns=needed).to_pandas()
        mask=(frame.level=='encoder')&(frame.comparison_kind=='delta')&(frame.metric_family=='vector')
        small.append(frame.loc[mask,needed])
    if not small:return {}
    ranked=rank_related_encoder_pairs(pd.concat(small,ignore_index=True),mode)
    if ranked[['checkpoint_a','checkpoint_b']].duplicated().any():
        raise ValueError('Duplicated encoder pair in shard schedule')
    return dict(zip(zip(ranked.checkpoint_a,ranked.checkpoint_b),ranked.rank_delta_cosine))


def _finalize(args,bases,checkpoints,stage_root):
    analysis=args.analysis
    n=shard_count(analysis,bases,checkpoints,args.splits_per_backbone)
    paths=[_part_path(stage_root,analysis,i) for i in range(n)]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(f'{len(missing)} of {n} shard Parquets missing for analysis {analysis}. Examples: {missing[:5]}')
    files=[pq.ParquetFile(p) for p in paths]
    signature=_configuration_signature(args)
    for path,pf in zip(paths,files):
        old=(pf.schema_arrow.metadata or {}).get(b'config_signature',b'').decode()
        if old!=signature:
            raise ValueError(f'Shard options mismatch at {path}: cannot finalize mixed-run results')
    present=[(p,pf) for p,pf in zip(paths,files) if pf.metadata.num_rows>0]
    if not present:
        LOG.warning('No observations for %s; no final result produced.',analysis)
        return
    # Arrow can unify per-shard columns without concatenating all millions of rows in RAM.
    schema=pa.unify_schemas([pf.schema_arrow.remove_metadata() for _,pf in present],promote_options='permissive')
    rank_map={}
    if analysis in ('03','04'):
        rank_map=_rank_lookup([p for p,_ in present],PAIR_KINDS[analysis])
        schema=schema.append(pa.field('rank_delta_cosine',pa.float64()))
    meta={**METADATA,'execution':'slurm_shards_then_atomic_finalize','analysis':NAMES[analysis].removesuffix('.parquet'),
          'partition_count':str(n),'config_signature':signature,
          'ranking_scope':'global_per_backbone' if rank_map else 'as_original'}
    schema=schema.with_metadata({str(k).encode():str(v).encode() for k,v in meta.items()})
    out=args.output/NAMES[analysis]
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_name('.'+out.name+'.'+uuid.uuid4().hex+'.partial')
    expected=sum(pf.metadata.num_rows for _,pf in present)
    count=0
    try:
        with pq.ParquetWriter(temporary,schema,compression='zstd',compression_level=5,
                              use_dictionary=True,write_statistics=True) as writer:
            for path,pf in present:
                LOG.info('Finalize %s rows=%d',path.name,pf.metadata.num_rows)
                for batch in pf.iter_batches(batch_size=32768):
                    # Stay in Arrow memory: don't materialize millions of
                    # object-dtype pandas rows or convert nested lists back.
                    incoming=pa.Table.from_batches([batch])
                    columns=[]
                    for field in schema:
                        if field.name=='rank_delta_cosine' and analysis in ('03','04'):
                            pair_a=incoming['checkpoint_a'].to_pylist()
                            pair_b=incoming['checkpoint_b'].to_pylist()
                            # The monolithic runner assigns ranks ONLY to encoder-level
                            # delta/vector rows. Pair IDs repeat across tensor, layer,
                            # head, overlap, subspace and final-weight rows; never leak
                            # the pair's rank into those scientifically different rows.
                            levels=incoming['level'].to_pylist()
                            kinds=incoming['comparison_kind'].to_pylist()
                            families=incoming['metric_family'].to_pylist()
                            rank_values=[rank_map.get((a,b),np.nan)
                                if (level=='encoder' and kind=='delta' and family=='vector')
                                else np.nan
                                for a,b,level,kind,family in
                                zip(pair_a,pair_b,levels,kinds,families)]
                            arr=pa.array(rank_values,type=pa.float64(),from_pandas=True)
                        elif field.name in incoming.column_names:
                            arr=incoming[field.name].cast(field.type,safe=True)
                        else:
                            arr=pa.nulls(incoming.num_rows,type=field.type)
                        columns.append(arr)
                    table=pa.Table.from_arrays(columns,schema=schema)
                    writer.write_table(table,row_group_size=32768)
                    count+=incoming.num_rows
        verify=pq.ParquetFile(temporary)
        if verify.metadata.num_rows!=expected or expected!=count:
            raise IOError(f'Final row mismatch: expected={expected}, wrote={count}')
        os.replace(temporary,out)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info('FINALIZED %s (%d rows, %d columns)',out,count,len(schema.names))
    # Cleanup only after final Parquet was written and validated successfully.
    for path in paths:path.unlink()
    try:(stage_root/analysis).rmdir()
    except OSError:pass
    try:stage_root.rmdir()
    except OSError:pass


def _run_derived(args):
    out=args.output;kind=args.analysis
    cross=_read_required(out/NAMES['07'])
    if kind=='11':
        write_parquet(model_distance_summary(cross),out/NAMES[kind]);return
    if kind=='09':
        metrics=[c for c in ('trace_distance_base','trace_distance_final','trace_distance_delta',
            'curve_l1_base','curve_l1_final','curve_l1_delta','quantile_w1_base','quantile_w1_final',
            'quantile_w1_delta','adaptation_profile_l1') if c in cross]
        if not metrics:raise ValueError('07 contains no comparable metrics')
        aggregated=cross.groupby(['backbone_a','backbone_b','entity'],as_index=False).agg(
            n_languages=('language','nunique'),**{m:(m,'mean') for m in metrics})
        for col in metrics:
            aggregated['rank_'+col]=aggregated.groupby('entity')[col].rank(method='min')
        aggregated['aggregation']='mean_across_languages_descriptive_not_independent'
        write_parquet(aggregated,out/NAMES[kind]);return
    if kind=='10':
        if not args.performance:
            raise ValueError('Analysis 10 requires --performance or PERFORMANCE=...')
        drift=_read_required(out/NAMES['01'])
        # Reapply scores if upstream jobs did not receive PERFORMANCE.
        scores=load_performance(args.performance)
        drift=with_performance(drift,scores) if 'f1_mean' not in drift else drift
        cross=with_performance(cross,scores) if 'abs_f1_gap' not in cross else cross
        write_parquet(summarize_f1(drift,cross),out/NAMES[kind]);return
    raise ValueError(kind)


def run(argv=None):
    args=parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.splits_per_backbone<1:raise ValueError('--splits-per-backbone must be >= 1')
    if args.dry_run or args.deep_dry_run:
        raise ValueError('Parallel --dry-run is not supported; first run the verified '
                         'monolithic `python -m weightspace --deep-dry-run` preflight.')
    args.originals=args.originals.expanduser().resolve()
    args.trained=args.trained.expanduser().resolve()
    args.output=args.output.expanduser().resolve()
    stage_root=(args.stage_dir or (args.output/'.weightspace_stage_v4')).expanduser().resolve()
    if args.analysis in PARTITIONED:
        bases,cps=discover(args.originals,args.trained)
        LOG.info('analysis=%s originals=%d checkpoints=%d',args.analysis,len(bases),len(cps))
        if args.finalize:
            _finalize(args,bases,cps,stage_root)
        else:
            scores=load_performance(args.performance) if args.performance else pd.DataFrame()
            _run_shard(args,bases,cps,stage_root,scores)
    else:
        if args.shard is not None or args.finalize:
            raise ValueError('Analyses 09/10/11 are derived single-job outputs; no shards needed')
        _run_derived(args)


def main():run()
if __name__=='__main__':main()
