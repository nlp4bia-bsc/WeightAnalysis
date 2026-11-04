"""Optional, strictly descriptive post-hoc F1 associations (no causal claims)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Spearman (1904) rank correlation and Pearson's product-moment correlation
# are classical statistics; we apply them descriptively to checkpoint metrics
# and *four-fold-aggregated* NER scores, without treating a saved fold as 4 folds.
# No p-values are reported for non-independent dyads or small groups.
def summarize_f1(drift:pd.DataFrame, cross_backbone:pd.DataFrame)->pd.DataFrame:
    rows=[]
    df=drift[(drift.level=='encoder') & drift.f1_mean.notna()].copy() if 'f1_mean' in drift else pd.DataFrame()
    if not df.empty:
        for group_cols in [['entity'],['backbone'],['backbone','entity'],['language','entity']]:
            for group,sub in df.groupby(group_cols,dropna=False):
                vals=group if isinstance(group,tuple) else (group,)
                identity=dict(zip(group_cols,vals))
                for metric in ('relative_l2','delta_rms','cosine_final_base'):
                    if metric not in sub:continue
                    rows.append(correlation_row(identity,'configuration',metric,'f1_mean',sub[metric],sub.f1_mean))
    # Dependencies among pairs make these rankings/correlations descriptive ONLY.
    df=cross_backbone.copy()
    if not df.empty and 'abs_f1_gap' in df:
        for (language,entity),sub in df.groupby(['language','entity']):
            for metric in ('curve_l1_delta','quantile_w1_delta','adaptation_profile_l1'):
                if metric in sub:
                    rows.append(correlation_row(dict(language=language,entity=entity),'nonindependent_pairs',
                                                 metric,'abs_f1_gap',sub[metric],sub.abs_f1_gap))
    return pd.DataFrame(rows)


def correlation_row(identity,unit,metric,outcome,x,y):
    data=pd.DataFrame({'x':x,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
    n=len(data)
    if n<3 or data.x.nunique()<2 or data.y.nunique()<2:
        sr=pr=float('nan');status='insufficient_variation_or_n'
    else:
        sr=float(spearmanr(data.x,data.y).statistic)
        pr=float(pearsonr(data.x,data.y).statistic)
        status='descriptive_only'
    return dict(**identity,analysis_unit=unit,feature=metric,outcome=outcome,n=n,
                spearman_rho=sr,pearson_r=pr,status=status,
                note='Fold1 checkpoint weights vs 4-fold mean F1; associations not causal; dyads not independent')
