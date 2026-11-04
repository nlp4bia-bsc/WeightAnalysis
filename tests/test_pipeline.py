import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file
from weightspace.checkpoints import discover,WeightReader,audit_alignment,normalize,config_metadata
from weightspace.analyses import describe_checkpoint,compare_related,compare_backbones,factorial
from weightspace.cli import run
from weightspace.storage import load_performance,with_performance


@pytest.fixture
def experiment(tmp_path):
    originals=tmp_path/'models';trained=tmp_path/'weights'
    for backbone in ('b1','b2'):
        initial={}
        for layer in range(3):
            for name in ('attention.self.query','attention.self.key','attention.self.value',
                         'attention.output.dense','intermediate.dense','output.dense'):
                # Sized 4x4; self-contained and invertible.
                initial[f'bert.encoder.layer.{layer}.{name}.weight']=torch.diag(torch.tensor([4.,3.,2.,1.]))*.1+layer*.02
        initial['bert.embeddings.word_embeddings.weight']=torch.randn(20,4)
        base=originals/backbone;base.mkdir(parents=True)
        save_file(initial,base/'model.safetensors')
        (base/'config.json').write_text(json.dumps({'model_type':'bert','hidden_size':4,'num_hidden_layers':3}))
        for language in ('es','it'):
            for entity in ('disease','symptom'):
                target=trained/language/entity/backbone;target.mkdir(parents=True)
                bump=.03 if language=='es' else .07
                if entity=='symptom':bump=-bump
                modified={k:v+torch.ones_like(v)*bump for k,v in initial.items()}
                modified['classifier.weight']=torch.ones((2,4))
                save_file(modified,target/'model.safetensors')
                (target/'config.json').write_text(json.dumps({'model_type':'bert','num_labels':2,'hidden_size':4}))
    return originals,trained


def test_discovery(experiment):
    bases,cps=discover(*experiment)
    assert len(cps)==8 and len(bases)==2
    with WeightReader(cps[0].path) as r,WeightReader(bases[cps[0].backbone]) as b:
        keys,status=audit_alignment(r,b)
        assert len(keys)==19 and status['element_coverage']==1
        assert 'classifier.weight' not in keys
    assert normalize('model.bert.encoder.layer.1.attention.self.query.weight')=='bert.encoder.layer.1.attention.self.query.weight'


def test_discovery_ignores_hidden_admin_tree_and_enforces_layout(experiment):
    """Regression for Picasso fscratch .overwrite_v2 backups."""
    originals, trained = experiment
    # Administrative trees may contain arbitrary nested directories, even
    # directories named like real languages and models, and must be ignored.
    bad = trained / '.overwrite_v2' / 'run_PRnB2nXZ_20261024T174438Z'
    (bad / '.aziz_grouped_v2_done').mkdir(parents=True)
    (bad / 'cz').mkdir()
    (bad / 'sv').mkdir()
    (trained / 'en' / '.some_backup' / 'disease' / 'b1').mkdir(parents=True)
    (trained / 'it' / 'disease' / '.some_backup').mkdir()
    (originals / '.old_originals' / 'b1').mkdir(parents=True)
    # This directory is not a known language and must not be traversed.
    (trained / 'old_runs' / 'en' / 'disease').mkdir(parents=True)
    bases, checkpoints = discover(originals, trained)
    assert len(checkpoints) == 8
    assert len(bases) == 2
    assert {c.language for c in checkpoints} == {'es', 'it'}


def test_discovery_rejects_missing_weights_in_valid_model_dir(experiment):
    originals, trained = experiment
    (trained / 'es' / 'procedure' / 'broken-model').mkdir(parents=True)
    with pytest.raises(ValueError, match='no recognized weights'):
        discover(originals, trained)


def test_discovery_rejects_unknown_visible_entity(experiment):
    originals, trained = experiment
    (trained / 'es' / 'procedures' / 'b1').mkdir(parents=True)
    with pytest.raises(ValueError, match='Unexpected entity directory'):
        discover(originals, trained)


def test_direct_and_factorial(experiment):
    from types import SimpleNamespace
    s=SimpleNamespace(min_aligned_fraction=.99,spectral_mode='exact',max_svd_elements=100000,topk=3,
       subspace_rank=2,topk_fraction=.25,max_topk_elements=1000,sign_floor=1e-8,
       device='cpu',factorial_chunk_elements=13)
    bases,cps=discover(*experiment)
    ds,sp=describe_checkpoint(cps[0],bases[cps[0].backbone],s)
    assert any(x.get('level')=='encoder' and x['delta_l2']>0 for x in ds)
    assert any(x.get('source')=='delta' and x.get('status')=='exact' for x in sp)
    a,b=cps[:2]
    comp=compare_related(a,b,bases[a.backbone],s)
    assert any(x.get('metric_family')=='vector' for x in comp)
    f=factorial([x for x in cps if x.backbone=='b1'],bases['b1'],s)
    assert len(f)>0
    for row in f:
        if row['ss_total']>1e-12:
            assert row['fraction_language']+row['fraction_entity']+row['fraction_interaction']==pytest.approx(1,abs=1e-8)
    df=pd.DataFrame(sp)
    drift=pd.DataFrame(ds)
    assert np.isfinite(compare_backbones(cps[0],next(x for x in cps if x.backbone=='b2' and x.language==cps[0].language and x.entity==cps[0].entity),df,drift)['adaptation_profile_l1'])==False # missing second profile is correctly null


def test_full_orchestration_without_parquet_engine(experiment,tmp_path,monkeypatch):
    # This environment lacks Arrow. Validate all 8 experiment families and
    # schema-ready pandas tables. Actual Parquet round-trip has a separate test.
    import weightspace.cli as cli
    import sys
    saved={}
    def fake_writer(rows,path,metadata=None):
        df=pd.DataFrame(rows)
        if not df.empty:saved[path.name]=df.copy()
        return len(df)
    import types
    monkeypatch.setitem(sys.modules,'pyarrow',types.ModuleType('pyarrow'))
    monkeypatch.setattr(cli,'write_parquet',fake_writer)
    run(['--originals',str(experiment[0]),'--trained',str(experiment[1]),'--output',str(tmp_path/'out'),
         '--spectral-mode','exact','--subspace-rank','1','--factorial-chunk-elements','11'])
    for x in ('01_pretraining_drift.parquet','02_spectral_fingerprints.parquet','03_cross_language.parquet',
              '04_cross_entity.parquet','05_cross_language_entity.parquet','07_cross_backbone.parquet',
              '08_language_entity_decomposition.parquet','09_cross_backbone_by_entity.parquet','11_model_centrality.parquet'):
        assert x in saved,x
    assert '06_cross_fold.parquet' not in saved
    assert 'element_coverage' in saved['01_pretraining_drift.parquet']
    assert 'rank_curve_l1_delta' in saved['07_cross_backbone.parquet']


def test_real_parquet_roundtrip_if_arrow(experiment,tmp_path):
    pyarrow=pytest.importorskip('pyarrow')
    import weightspace.storage as io
    path=tmp_path/'01_pretraining_drift.parquet'
    df=pd.DataFrame([{'backbone':'test','value':1.,'curve':[0.,.5,1.]}])
    io.write_parquet(df,path)
    got=pd.read_parquet(path)
    assert len(got)==1 and got.iloc[0].backbone=='test'
    assert not list(tmp_path.glob('*.partial'))


def test_sharded_safetensors(experiment,tmp_path):
    # Sanity-test strict HF shard-map validation and reader access.
    source=experiment[0]/'b1'
    target=tmp_path/'sharded';target.mkdir()
    from safetensors.torch import save_file
    tensors={'bert.encoder.layer.0.attention.self.query.weight':torch.eye(3),
             'bert.encoder.layer.0.attention.self.key.weight':torch.eye(3)*2}
    save_file({list(tensors)[0]:list(tensors.values())[0]},target/'model-00001-of-00002.safetensors')
    save_file({list(tensors)[1]:list(tensors.values())[1]},target/'model-00002-of-00002.safetensors')
    (target/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{
        key:f'model-{i+1:05d}-of-00002.safetensors' for i,key in enumerate(tensors)}}))
    with WeightReader(target) as r:
        assert len(r.keys)==2
        assert torch.equal(r.get(list(tensors)[1]),tensors[list(tensors)[1]])
    (target/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'fake.weight':'missing.safetensors'}}))
    with pytest.raises(FileNotFoundError):
        with WeightReader(target):pass


def test_missing_and_bad_checkpoint(experiment,tmp_path):
    originals,trained=experiment
    (trained/'es'/'disease'/'b1'/'model.safetensors').unlink()
    with pytest.raises(ValueError,match='no recognized weights'):
        discover(originals,trained)


def test_performance_join(tmp_path):
    from weightspace.storage import with_performance
    from weightspace.performance import summarize_f1
    scores=pd.DataFrame([dict(backbone='a',language='es',entity='disease',f1_mean=.7,f1_std=.05),
                         dict(backbone='b',language='es',entity='disease',f1_mean=.8,f1_std=.03)])
    pair=pd.DataFrame([dict(backbone_a='a',backbone_b='b',language='es',entity='disease',curve_l1_delta=.5)])
    joined=with_performance(pair,scores)
    assert joined.iloc[0].abs_f1_gap==pytest.approx(.1)
    assert joined.iloc[0].performance_scope=='mean_sd_four_folds_vs_weights_fold1_only'
    result=summarize_f1(pd.DataFrame(),joined)
    assert len(result)==1 and result.iloc[0].status=='insufficient_variation_or_n'


def test_config_dimension_mismatch_fails(experiment):
    from weightspace.cli import run
    original=experiment[0]/'b1'/'config.json'
    original.write_text(json.dumps({'model_type':'bert','hidden_size':99,'num_hidden_layers':3}))
    with pytest.raises(ValueError,match='hidden_size'):
        run(['--originals',str(experiment[0]),'--trained',str(experiment[1]),'--output','/tmp/unused','--dry-run'])


def test_factorial_known_effects(tmp_path):
    """Create data with language-only and entity-only components with orthogonal coordinates."""
    from types import SimpleNamespace
    from weightspace.analyses import factorial
    root=tmp_path/'d';root.mkdir()
    base=root/'base';base.mkdir()
    k='bert.encoder.layer.0.attention.self.query.weight'
    initial={k:torch.zeros(2,2)}
    save_file(initial,base/'model.safetensors')
    cps=[]
    from weightspace.checkpoints import Checkpoint
    for l,lv in [('es',1.),('it',-1.)]:
        for e,ev in [('disease',2.),('symptom',-2.)]:
            target=root/l/e/'m';target.mkdir(parents=True)
            w=torch.tensor([[lv,ev],[0.,0.]])
            save_file({k:w},target/'model.safetensors')
            cps.append(Checkpoint('m',l,e,target))
    s=SimpleNamespace(min_aligned_fraction=.99,factorial_chunk_elements=1)
    r=factorial(cps,base,s)
    encoder=next(q for q in r if q['level']=='encoder')
    assert encoder['ss_interaction']==pytest.approx(0)
    assert encoder['ss_language']==pytest.approx(4)
    assert encoder['ss_entity']==pytest.approx(16)
    assert encoder['ss_total']==pytest.approx(20)


def test_full_physical_parquet_pipeline_if_arrow(experiment,tmp_path):
    pytest.importorskip('pyarrow')
    from weightspace.cli import run
    destination=tmp_path/'results'
    run(['--originals',str(experiment[0]),'--trained',str(experiment[1]),
         '--output',str(destination),'--spectral-mode','exact','--subspace-rank','1',
         '--factorial-chunk-elements','11'])
    expected={'01_pretraining_drift.parquet','02_spectral_fingerprints.parquet',
              '03_cross_language.parquet','04_cross_entity.parquet','05_cross_language_entity.parquet',
              '07_cross_backbone.parquet','08_language_entity_decomposition.parquet',
              '09_cross_backbone_by_entity.parquet','11_model_centrality.parquet'}
    assert {p.name for p in destination.iterdir()}==expected
    for p in destination.iterdir():
        df=pd.read_parquet(p)
        assert not df.empty
    assert 'trained_path' in pd.read_parquet(destination/'01_pretraining_drift.parquet')


def test_cross_backbone_matched_fingerprints(experiment):
    from types import SimpleNamespace
    bases,cps=discover(*experiment)
    s=SimpleNamespace(min_aligned_fraction=.99,spectral_mode='exact',max_svd_elements=100000,
                      topk=3,subspace_rank=0,topk_fraction=0.,max_topk_elements=1000,
                      sign_floor=1e-8,device='cpu',factorial_chunk_elements=10)
    dr=[];sp=[]
    chosen=[c for c in cps if c.language=='es' and c.entity=='disease']
    assert len(chosen)==2
    for c in chosen:
        d,z=describe_checkpoint(c,bases[c.backbone],s)
        dr.extend(d);sp.extend(z)
    comp=compare_backbones(chosen[0],chosen[1],pd.DataFrame(sp),pd.DataFrame(dr))
    assert comp['n_components_curve_l1_final']>=4
    assert comp['curve_l1_final']==pytest.approx(0.,abs=1e-8)
    assert comp['curve_l1_delta']==pytest.approx(0.,abs=1e-8)
    assert comp['quantile_w1_delta']==pytest.approx(0.,abs=1e-8)
    assert comp['trace_similarity_final']==pytest.approx(1.,abs=1e-12)
    assert comp['trace_distance_final']==pytest.approx(0.,abs=1e-12)


def test_model_centrality_complete_and_missing():
    from weightspace.analyses import model_distance_summary
    vals=[('A','B',1.),('A','C',3.),('B','C',2.)]
    rows=[dict(language='es',entity='disease',backbone_a=a,backbone_b=b,curve_l1_delta=dist)
          for a,b,dist in vals]
    summ=model_distance_summary(pd.DataFrame(rows))
    item=summ[(summ.backbone=='B')&(summ.metric=='curve_l1_delta')].iloc[0]
    assert item.n_neighbors==2
    assert item.nearest_backbone=='A'
    assert item.mean_distance==pytest.approx(1.5)
    rows[0]['curve_l1_delta']=float('nan')
    partial=model_distance_summary(pd.DataFrame(rows))
    assert not partial[(partial.backbone=='B')&(partial.metric=='curve_l1_delta')].iloc[0].complete_comparison
    assert pd.isna(partial[(partial.backbone=='B')&(partial.metric=='curve_l1_delta')].iloc[0].rank_most_central)
