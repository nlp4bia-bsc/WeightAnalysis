"""Scientific/regression tests for SLURM sharding and final Parquet assembly."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file
from weightspace.analyses import describe_checkpoint
from weightspace.checkpoints import discover
from types import SimpleNamespace


@pytest.fixture
def little_study(tmp_path):
    originals=tmp_path/'base';trained=tmp_path/'fine'
    for model in ('b1','b2'):
        p=originals/model;p.mkdir(parents=True)
        tensors={}
        for layer in range(2):
            for comp in ('attention.self.query','attention.self.key','attention.self.value',
                         'attention.output.dense','intermediate.dense','output.dense'):
                tensors[f'bert.encoder.layer.{layer}.{comp}.weight']=torch.eye(3,dtype=torch.float32)*(layer+.7)
        tensors['bert.embeddings.word_embeddings.weight']=torch.ones(8,3)
        save_file(tensors,p/'model.safetensors')
        (p/'config.json').write_text(json.dumps({'model_type':'bert','hidden_size':3,'num_hidden_layers':2}))
        for language in ('es','it'):
            for entity in ('disease','symptom'):
                d=trained/language/entity/model;d.mkdir(parents=True)
                bump=.07 if language=='es' else -.02
                if entity=='symptom':bump+=.015
                save_file({**{k:v+bump for k,v in tensors.items()},
                           'classifier.weight':torch.ones(2,3)},d/'model.safetensors')
                (d/'config.json').write_text(json.dumps({'model_type':'bert','hidden_size':3}))
    return originals,trained


def test_a1_drift_and_spectra_independent_match_original(little_study):
    bases,cps=discover(*little_study)
    settings=SimpleNamespace(min_aligned_fraction=.99, spectral_mode='exact',max_svd_elements=5000,
                             topk=3,device='cpu')
    cp=cps[0]
    both_d,both_s=describe_checkpoint(cp,bases[cp.backbone],settings,{})
    only_d,no_s=describe_checkpoint(cp,bases[cp.backbone],settings,{},compute_spectra=False)
    no_d,only_s=describe_checkpoint(cp,bases[cp.backbone],settings,{},compute_drift=False)
    assert len(both_d)==len(only_d)>0 and not no_s
    assert len(both_s)==len(only_s)>0 and not no_d
    pd.testing.assert_frame_equal(pd.DataFrame(both_d),pd.DataFrame(only_d))
    a=pd.DataFrame(both_s).sort_values(['tensor','source']).reset_index(drop=True)
    b=pd.DataFrame(only_s).sort_values(['tensor','source']).reset_index(drop=True)
    for col in ('matrix_fro','matrix_rows','matrix_cols','status'):
        pd.testing.assert_series_equal(a[col],b[col])


def test_parallel_shards_and_reassembly_equal_monolithic(little_study,tmp_path):
    pytest.importorskip('pyarrow')
    from weightspace.cli import run as original_run
    from weightspace.parallel import run as parallel_run,shard_count
    originals,trained=little_study
    old=tmp_path/'old';new=tmp_path/'new'
    common=['--originals',str(originals),'--trained',str(trained),
            '--spectral-mode','exact','--max-svd-elements','5000','--subspace-rank','1',
            '--factorial-chunk-elements','7','--topk-fraction','.1']
    original_run(common+['--output',str(old)])
    bases,cps=discover(originals,trained)
    for analysis in ('01','02','03','04','05','08'):
        n=shard_count(analysis,bases,cps,2)
        for shard in range(n):
            parallel_run(common+['--output',str(new),'--analysis',analysis,
                                 '--splits-per-backbone','2','--shard',str(shard)])
        parallel_run(common+['--output',str(new),'--analysis',analysis,
                             '--splits-per-backbone','2','--finalize'])
    for analysis in ('07',):
        for shard in range(shard_count(analysis,bases,cps,2)):
            parallel_run(common+['--output',str(new),'--analysis',analysis,
                                 '--splits-per-backbone','2','--shard',str(shard)])
        parallel_run(common+['--output',str(new),'--analysis',analysis,
                             '--splits-per-backbone','2','--finalize'])
    for analysis in ('09','11'):
        parallel_run(common+['--output',str(new),'--analysis',analysis])
    assert not (new/'.weightspace_stage_v4').exists()
    for name in sorted(p.name for p in old.glob('*.parquet')):
        x=pd.read_parquet(old/name)
        y=pd.read_parquet(new/name)
        assert set(x)==set(y),name
        assert len(x)==len(y),name
        # Row ordering may differ by shard; compare stable, scientifically relevant
        # tensor/pair identifiers and numerical summary columns after normalization.
        candidates=[c for c in ('checkpoint','checkpoint_a','checkpoint_b','backbone',
                    'backbone_a','backbone_b','language','language_a','language_b',
                    'entity','entity_a','entity_b','level','layer','component','tensor',
                    'source','comparison_kind','metric_family','metric','feature') if c in x]
        assert candidates
        numerical=[c for c in ('cosine','relative_l2','delta_l2','matrix_fro',
                   'rank_delta_cosine','trace_distance_delta','rank_trace_distance_delta',
                   'ss_total','mean_distance') if c in x]
        def canonical(frame):
            f=frame[candidates+numerical].copy()
            f=f.sort_values(candidates,na_position='first',kind='stable').reset_index(drop=True)
            return f
        pd.testing.assert_frame_equal(canonical(x),canonical(y),check_dtype=False,rtol=1e-8,atol=1e-8)


def test_related_ranks_only_encoder_delta_vector(little_study,tmp_path):
    """Final rank must stay NaN on tensor/layer/head/final/subspace rows."""
    pytest.importorskip('pyarrow')
    from weightspace.parallel import run as parallel_run, shard_count, NAMES
    originals,trained=little_study
    output=tmp_path/'rank_only'
    common=['--originals',str(originals),'--trained',str(trained),
            '--output',str(output),'--spectral-mode','skip','--subspace-rank','0',
            '--topk-fraction','0','--splits-per-backbone','2']
    bases,cps=discover(originals,trained)
    for analysis in ('03','04'):
        for shard in range(shard_count(analysis,bases,cps,2)):
            parallel_run(common+['--analysis',analysis,'--shard',str(shard)])
        parallel_run(common+['--analysis',analysis,'--finalize'])
        df=pd.read_parquet(output/NAMES[analysis])
        targeted=(df.level=='encoder')&(df.comparison_kind=='delta')&(df.metric_family=='vector')
        assert targeted.any() and (~targeted).any()
        assert df.loc[targeted,'rank_delta_cosine'].notna().all()
        assert df.loc[~targeted,'rank_delta_cosine'].isna().all()
