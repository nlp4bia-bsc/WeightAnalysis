"""Picasso submission regression: no --dependency and no empty job IDs."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def script_environment(tmp_path):
    root=Path(__file__).resolve().parents[1]
    project=tmp_path/'WeighSpace_Analysis'
    project.mkdir()
    for name in ('submit_weightspace_parallel.sh','advance_weightspace_parallel.sh','run_weightspace_parallel.sh','retry_weightspace_parallel.sh','slurm_jobid.sh','weightspace_parallel.py'):
        shutil.copyfile(root/name,project/name)
    (project/'weightspace').mkdir()
    for name in ('__init__.py','checkpoints.py','slurm_guard.py'):
        shutil.copyfile(root/'weightspace'/name, project/'weightspace'/name)
    originals=tmp_path/'models'
    trained=tmp_path/'weights'
    for model in ('model_a','model_b'):
        b=originals/model
        b.mkdir(parents=True)
        save_file({'bert.embeddings.word_embeddings.weight':torch.ones(4,2)},b/'model.safetensors')
        (b/'config.json').write_text('{"model_type":"bert"}')
        for lang in ('es','sv'):
            d=trained/lang/'disease'/model
            d.mkdir(parents=True)
            save_file({'bert.embeddings.word_embeddings.weight':torch.ones(4,2)},d/'model.safetensors')
            (d/'config.json').write_text('{"model_type":"bert"}')
    bindir=tmp_path/'fake_bin';bindir.mkdir()
    fake=bindir/'sbatch'
    fake.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_SBATCH_LOG"
if [[ "$*" == *--dependency* ]]; then echo 'Job dependency problem' >&2; exit 19; fi
count=$(wc -l < "$FAKE_SBATCH_LOG")
if [[ -n "${FAKE_FAIL_ON:-}" && "$count" == "$FAKE_FAIL_ON" ]]; then
  echo 'Simulated Slurm submission failure' >&2
  exit 9
fi
case "${FAKE_OUTPUT_MODE:-normal}" in
  normal) printf '90%04d\\n' "$count" ;;
  crlf) printf '90%04d\\r\\n' "$count" ;;
  whitespace) printf '  90%04d  \\n' "$count" ;;
  notices) printf 'sbatch.orig: (lua): cluster notice\\n90%04d\\r\\n' "$count" ;;
  cluster) printf '90%04d;picasso\\r\\n' "$count" ;;
  traditional) printf 'Submitted batch job 90%04d\\n' "$count" ;;
  garbled) printf 'sbatch accepted (unparseable) ID=90%04d\\n' "$count" ;;
  *) exit 13 ;;
esac
''')
    fake.chmod(0o755)
    output=tmp_path/'output'; output.mkdir()
    log=tmp_path/'calls.txt'
    env=os.environ.copy()
    env.update(PATH=str(bindir)+os.pathsep+env['PATH'], PYTHONPATH=str(project),
               WEIGHTSPACE_PROJECT_DIR=str(project),ORIGINALS=str(originals),
               TRAINED=str(trained), OUTPUT=str(output),SPLITS_PER_BACKBONE='2',
               FAKE_SBATCH_LOG=str(log))
    env.pop('WEIGHTSPACE_RUN_DIR',None)
    env.pop('STAGE_DIR',None)
    env.pop('PERFORMANCE',None)
    return dict(project=project,originals=originals,trained=trained,
                output=output,log=log,env=env)


def run_shell(ctx, script, *, env=None):
    return subprocess.run(['bash',str(ctx['project']/script)],
                          cwd=ctx['project'],env=env or ctx['env'],
                          text=True,capture_output=True)


def _setup(ctx):
    result=run_shell(ctx,'submit_weightspace_parallel.sh')
    assert result.returncode==0, result.stderr
    run_dirs=list((ctx['output']/'.weightspace_runs').iterdir())
    assert len(run_dirs)==1
    run_dir=run_dirs[0]
    env={**ctx['env'],'WEIGHTSPACE_RUN_DIR':str(run_dir)}
    return run_dir,env


def test_independent_submission_without_dependency(script_environment):
    ctx=script_environment
    run_dir, env=_setup(ctx)
    calls=ctx['log'].read_text().splitlines()
    assert len(calls)==6
    assert all('--dependency' not in c for c in calls)
    assert sum('--constraint=dgx' in c for c in calls)==4
    assert sum('--constraint=cpu' in c for c in calls)==2
    assert all('--array=0-' in c for c in calls)
    assert '--array=0-3' in calls[2]
    assert all('%2' not in c and '%3' not in c for c in calls)
    assert len(list((run_dir/'submitted').glob('shard-*')))==6
    # Resume must not submit already-created arrays twice.
    resume=run_shell(ctx,'submit_weightspace_parallel.sh',env=env)
    assert resume.returncode==0, resume.stderr
    assert len(ctx['log'].read_text().splitlines())==6


def test_dispatch_phase_gating_and_idempotency(script_environment):
    ctx=script_environment
    run_dir, env=_setup(ctx)
    stage=ctx['output']/('.weightspace_stage_v4_'+run_dir.name)
    def mark(analysis,n):
        d=stage/analysis;d.mkdir(parents=True,exist_ok=True)
        for i in range(n):
            (d/f'part-{i:04d}.parquet').write_text('atomic committed fixture')
    def advance():
        r=run_shell(ctx,'advance_weightspace_parallel.sh',env=env)
        assert r.returncode==0, r.stderr
    advance()
    assert len(ctx['log'].read_text().splitlines())==6
    mark('01',2);advance()
    assert (run_dir/'submitted/final-01').exists()
    assert not (run_dir/'submitted/shard-07').exists()
    mark('02',2);advance()
    assert (run_dir/'submitted/final-02').exists()
    # Marking only one finalizer must not release 07.
    (run_dir/'completed/final-01').write_text('ok')
    advance()
    assert not (run_dir/'submitted/shard-07').exists()
    (run_dir/'completed/final-02').write_text('ok')
    advance()
    assert (run_dir/'submitted/shard-07').exists()
    mark('07',2);advance()
    assert (run_dir/'submitted/final-07').exists()
    assert not (run_dir/'submitted/derived-09').exists()
    (run_dir/'completed/final-07').write_text('ok')
    advance()
    assert (run_dir/'submitted/derived-09').exists()
    assert (run_dir/'submitted/derived-11').exists()
    assert not (run_dir/'submitted/derived-10').exists()
    previous=ctx['log'].read_text()
    advance()
    assert ctx['log'].read_text()==previous
    assert all('--dependency' not in s for s in previous.splitlines())


def test_submission_failure_is_not_silently_ignored(script_environment):
    ctx=script_environment
    env={**ctx['env'],'FAKE_FAIL_ON':'2'}
    r=run_shell(ctx,'submit_weightspace_parallel.sh',env=env)
    assert r.returncode != 0
    assert 'FATAL' in r.stderr
    run_dir=next((ctx['output']/'.weightspace_runs').iterdir())
    assert (run_dir/'submitted/shard-01').exists()
    assert not (run_dir/'submitted/shard-02').exists()
    env.pop('FAKE_FAIL_ON')
    env['WEIGHTSPACE_RUN_DIR']=str(run_dir)
    r=run_shell(ctx,'submit_weightspace_parallel.sh',env=env)
    assert r.returncode==0,r.stderr
    assert len(list((run_dir/'submitted').glob('shard-*')))==6


def test_freeze_rejects_later_modified_checkpoints(script_environment):
    ctx=script_environment
    run_dir, env=_setup(ctx)
    before=subprocess.run(['python','-m','weightspace.slurm_guard','check',
                           '--run-dir',str(run_dir)],cwd=ctx['project'],
                          env=env,capture_output=True,text=True)
    assert before.returncode==0, before.stderr
    sample=next(ctx['trained'].glob('es/disease/*/config.json'))
    sample.write_text('{"model_type":"bert","hidden_size":4096}')
    after=subprocess.run(['python','-m','weightspace.slurm_guard','check',
                          '--run-dir',str(run_dir)],cwd=ctx['project'],env=env,
                         capture_output=True,text=True)
    assert after.returncode != 0
    assert 'inventory changed' in after.stderr


def test_same_scientific_options_for_shard_and_finalizer():
    content=(Path(__file__).resolve().parents[1]/'run_weightspace_parallel.sh').read_text()
    assert content.index('ARGS=(--originals') < content.index('case "$TASK_MODE" in')
    assert content.count('--max-svd-elements 3000000')==1
    assert content.count('--spectral-mode auto')==1
    assert 'WEIGHTSPACE_RUN_DIR' in content


def test_retry_only_missing_shard_indices(script_environment):
    ctx=script_environment
    run_dir,env=_setup(ctx)
    stage=ctx['output']/('.weightspace_stage_v4_'+run_dir.name)
    d=stage/'01';d.mkdir(parents=True)
    (d/'part-0000.parquet').write_text('finished')
    result=run_shell(ctx,'retry_weightspace_parallel.sh',env=env)
    assert result.returncode==0, result.stderr
    calls=ctx['log'].read_text().splitlines()[6:]
    cpu_01=[c for c in calls if '--job-name=ws01 ' in c]
    assert len(cpu_01)==1
    assert '--array=1 ' in cpu_01[0]
    assert all('--dependency' not in c for c in calls)
    assert len(list((run_dir/'submitted').glob('retry-shard-*'))) == 6


@pytest.mark.parametrize('mode', ['normal','crlf','whitespace','notices','cluster','traditional'])
def test_site_wrapped_sbatch_output_parses_for_all_six_arrays(script_environment, mode):
    ctx=script_environment
    env={**ctx['env'],'FAKE_OUTPUT_MODE':mode}
    result=run_shell(ctx,'submit_weightspace_parallel.sh',env=env)
    assert result.returncode==0,(mode,result.stdout,result.stderr)
    run_dir=next((ctx['output']/'.weightspace_runs').iterdir())
    markers=sorted((run_dir/'submitted').glob('shard-*'))
    assert len(markers)==6
    assert [p.read_text().strip() for p in markers]==[f'90{i:04d}' for i in range(1,7)]


def test_site_wrapped_sbatch_output_parses_in_dispatcher(script_environment):
    ctx=script_environment
    run_dir,env=_setup(ctx)
    env['FAKE_OUTPUT_MODE']='notices'
    stage=ctx['output']/('.weightspace_stage_v4_'+run_dir.name)
    d=stage/'01';d.mkdir(parents=True)
    for i in range(2):
        (d/f'part-{i:04d}.parquet').write_text('fixture')
    r=run_shell(ctx,'advance_weightspace_parallel.sh',env=env)
    assert r.returncode==0,r.stderr
    assert (run_dir/'submitted/final-01').read_text().strip()=='900007'


def test_unknown_successful_sbatch_output_is_quarantined_not_resent(script_environment):
    ctx=script_environment
    env={**ctx['env'],'FAKE_OUTPUT_MODE':'garbled'}
    r=run_shell(ctx,'submit_weightspace_parallel.sh',env=env)
    assert r.returncode!=0
    run_dir=next((ctx['output']/'.weightspace_runs').iterdir())
    assert (run_dir/'submitted/unparsed-shard-01').is_file()
    assert not (run_dir/'submitted/shard-01').exists()
    first=ctx['log'].read_text()
    env['WEIGHTSPACE_RUN_DIR']=str(run_dir)
    env['FAKE_OUTPUT_MODE']='normal'
    second=run_shell(ctx,'submit_weightspace_parallel.sh',env=env)
    assert second.returncode!=0
    assert ctx['log'].read_text()==first


def test_submitted_scripts_and_fixture_have_all_companion_files():
    """Catch packaging/test-version drift before running any sbatch simulation."""
    root = Path(__file__).resolve().parents[1]
    needed = ('submit_weightspace_parallel.sh', 'advance_weightspace_parallel.sh',
              'retry_weightspace_parallel.sh', 'slurm_jobid.sh')
    for name in needed:
        assert (root / name).is_file(), f'Missing install file: {name}'
    fixture_source = Path(__file__).read_text()
    assert "'slurm_jobid.sh'" in fixture_source, 'Outdated orchestration tests (v4.3 fixture)'

