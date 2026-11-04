"""Single ZSTD Parquet per populated experiment, with atomic completion."""
from __future__ import annotations
from pathlib import Path
import logging
import math
import os
import uuid
import numpy as np
import pandas as pd
LOG=logging.getLogger(__name__)

METADATA={
 'schema_version':'3.0',
 'study':'MultiClinAI checkpoint-only fold-1 weight-space study',
 'comparability':'Direct deltas only within a shared original backbone; heterogeneous encoders compared via intrinsic profiles',
 'performance':'Optional F1 mean±SD over four folds; NOT attributable to checkpoint fold 1',
 'spectra':'Truncated SVD does not report exact effective rank, nuclear norm or energy CDF; TRACE is Frobenius layer-profile correlation',
 'refs':'Ilharco ICLR 2023; Radiya-Dixit AISTATS 2020; Roy/Vetterli EUSIPCO 2007; Yadav NeurIPS 2023; Li Findings ACL 2025; Chen et al. COLM 2026',
}

# Parquet ZSTD compression/dictionary encoding: Apache Arrow/Parquet standard.
# No manifest or sidecar is created; method metadata embedded in Parquet schema.
def write_parquet(rows, path:Path,metadata:dict|None=None):
    if rows is None:return 0
    if isinstance(rows,pd.DataFrame): df=rows.copy()
    else:df=pd.DataFrame(rows)
    if df.empty:
        LOG.info('No observations for %s; omitted',path.name);return 0
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('pyarrow not installed; install requirements (do NOT run entire HPC job before preflight)') from exc
    df=df.reset_index(drop=True)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col]=df[col].replace([np.inf,-np.inf],np.nan)
        elif df[col].dtype=='object':
            # Keep list-valued columns for spectral CDF/quantile arrays.
            if df[col].dropna().map(lambda v:isinstance(v,(list,tuple,np.ndarray))).any():
                df[col]=df[col].map(lambda v:list(v) if isinstance(v,np.ndarray) else v)
    # Transform NaN to Arrow null instead of special float where possible.
    table=pa.Table.from_pandas(df,preserve_index=False)
    md={**METADATA,**(metadata or {}), 'analysis':path.stem}
    table=table.replace_schema_metadata({k.encode():str(v).encode() for k,v in md.items()})
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        pq.write_table(table,temporary,compression='zstd',compression_level=5,
            use_dictionary=True,write_statistics=True,row_group_size=32_768)
        # Verify it is readable BEFORE replacing a previous complete result.
        check=pq.ParquetFile(temporary)
        if check.metadata.num_rows!=len(df): raise IOError('Parquet row count mismatch')
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info('Wrote %s (%d rows, %d cols)',path,len(df),len(df.columns))
    return len(df)


def load_performance(path:Path|None):
    if path is None:return pd.DataFrame()
    df=pd.read_parquet(path) if path.suffix.lower()=='.parquet' else pd.read_csv(path)
    required=['backbone','language','entity','f1_mean','f1_std']
    if not set(required)<=set(df.columns): raise ValueError(f'Performance table must contain {required}')
    df=df[required].copy()
    for col in ['backbone','language','entity']:df[col]=df[col].astype(str)
    df['language']=df.language.str.lower();df['entity']=df.entity.str.lower()
    for col in ['f1_mean','f1_std']:df[col]=pd.to_numeric(df[col],errors='raise')
    if df[required[:3]].duplicated().any():raise ValueError('Duplicate F1 keys')
    if (df.f1_std<0).any():raise ValueError('Negative F1 SD')
    if not np.isfinite(df[['f1_mean','f1_std']].to_numpy()).all(): raise ValueError('Nonfinite F1')
    return df


def with_performance(df:pd.DataFrame,score:pd.DataFrame)->pd.DataFrame:
    if df.empty or score.empty:return df
    keys=['backbone','language','entity']
    if set(keys)<=set(df.columns):
        result=df.merge(score,on=keys,how='left',validate='m:1')
    elif {'backbone','language_a','entity_a','language_b','entity_b'}<=set(df.columns):
        result=df
        for letter in ('a','b'):
            changed=score.rename(columns={'language':f'language_{letter}','entity':f'entity_{letter}',
                                          'f1_mean':f'f1_mean_{letter}','f1_std':f'f1_std_{letter}'})
            result=result.merge(changed,on=['backbone',f'language_{letter}',f'entity_{letter}'],how='left',validate='m:1')
        result['abs_f1_gap']=(result.f1_mean_a-result.f1_mean_b).abs()
    elif {'backbone_a','backbone_b','language','entity'}<=set(df.columns):
        result=df
        for letter in ('a','b'):
            changed=score.rename(columns={'backbone':f'backbone_{letter}',
                                          'f1_mean':f'f1_mean_{letter}','f1_std':f'f1_std_{letter}'})
            result=result.merge(changed,on=[f'backbone_{letter}','language','entity'],how='left',validate='m:1')
        result['abs_f1_gap']=(result.f1_mean_a-result.f1_mean_b).abs()
    else:return df
    result['performance_scope']='mean_sd_four_folds_vs_weights_fold1_only'
    return result
