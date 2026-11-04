"""Opt-in differential tests running REAL author scorer code from a LOCAL checkout.

To enable:
  git clone --depth=1 https://github.com/OPTML-Group/LLM-Biometrics.git /tmp/LLM-Biometrics
  OFFICIAL_LLM_BIOMETRICS=/tmp/LLM-Biometrics python -m pytest -q tests/test_external_oracles.py

Runs scorer modules ONLY; never downloads checkpoints, trains or does inference.
Inspect the upstream repository before running third-party source code.
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from weightspace.metrics import (
    paper_trace_similarity, paper_subspace_layer_alignment,
    paper_subspace_hierarchical_alignment,
)

REPO=os.getenv('OFFICIAL_LLM_BIOMETRICS','')


def _reference_file(name):
    if not REPO:
        pytest.skip('Set OFFICIAL_LLM_BIOMETRICS to an audited local author checkout')
    p=Path(REPO)/'llm_biometrics'/name
    if not p.is_file():
        pytest.fail(f'Official scorer not present at {p}')
    return p


def _load_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


# Author code direct oracle: Chen, Shang & Liu, COLM 2026, Eq. 5.
# https://github.com/OPTML-Group/LLM-Biometrics/blob/main/llm_biometrics/trace.py
@pytest.mark.parametrize('seed',[1,2,3,4,5,6])
def test_trace_against_authors_actual_scorer(seed):
    official=_load_file('official_trace_scoring',_reference_file('trace.py'))
    rng=np.random.default_rng(seed)
    a={name:rng.uniform(.1,4.,size=n) for name,n in [('q',5),('k',3),('v',6)]}
    b={name:rng.uniform(.1,4.,size=n) for name,n in [('q',4),('k',7),('v',6)]}
    expected, components=official.trace_similarity(a,b)
    result=paper_trace_similarity(a,b)
    assert result['trace_similarity']==pytest.approx(expected,abs=1e-12)
    for name,score in components.items():
        assert result['trace_per_component'][name]==pytest.approx(score,abs=1e-12)


# Author code direct oracle: Chen, Shang & Liu, COLM 2026, Eq. 8 and Alg A2.
# https://github.com/OPTML-Group/LLM-Biometrics/blob/main/llm_biometrics/subspace.py
@pytest.mark.parametrize('seed',[8,13,21])
def test_subspace_against_authors_actual_scorer(seed):
    _reference_file('subspace.py')
    # Official subspace.py uses a relative import of its defaults module.
    package_name='_official_llmbiometrics_validation'
    namespace=types.ModuleType(package_name)
    namespace.__path__=[str(Path(REPO)/'llm_biometrics')]
    sys.modules[package_name]=namespace
    _load_file(package_name+'.defaults',_reference_file('defaults.py'))
    official=_load_file(package_name+'.subspace',_reference_file('subspace.py'))
    rng=np.random.default_rng(seed)
    layer_scores={}
    for comp in ('q','ffn_up'):
        layer_scores[comp]=[]
        for _ in range(5):
            qa,_=np.linalg.qr(rng.standard_normal((20,4)))
            qb,_=np.linalg.qr(rng.standard_normal((20,4)))
            ref=official.layer_alignment(qa,qb,J=2)
            got=paper_subspace_layer_alignment(torch.from_numpy(qa),torch.from_numpy(qb),j=2)
            assert got==pytest.approx(ref,abs=1e-12)
            layer_scores[comp].append(got)
    ours=paper_subspace_hierarchical_alignment(layer_scores,n_bottom=3)
    expect=np.mean([np.mean(np.sort(s)[:3]) for s in layer_scores.values()])
    assert ours['subspace_similarity']==pytest.approx(expect,abs=1e-12)
