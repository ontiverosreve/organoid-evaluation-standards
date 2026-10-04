"""Audit #2 Exp F: feedback-schedule comparison within HCC (paper Fig. 6).

Paper claim: only the Stimulus-feedback condition shows a significant T1->T2
increase; No-feedback does not. The release has no 'silent' tags and no mapping
doc for the feedback pkls' 'control' column, so we reconstruct STIM-like vs NF
from the main CSV's HCC tags:
  STIM-like (closed-loop stimulus): tags with 'random_3' or 'reseed'
  NF (no feedback): tags with 'no_feedback'
This is an approximation; the silent condition is absent from the release.
"""
import sys, types
import pandas as pd
mod = types.ModuleType('pandas.core.indexes.numeric')
mod.Int64Index = pd.Index; mod.UInt64Index = pd.Index
mod.Float64Index = pd.Index; mod.NumericIndex = pd.Index
sys.modules['pandas.core.indexes.numeric'] = mod

import numpy as np
from scipy import stats
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RES = os.path.join(HERE, '..', 'results')

df = pd.read_csv(os.path.join(DATA, 'sentience_corr.csv'),
                 usecols=['chip_id', 'group', 'tag', 'date', 'session_num',
                          'elapse_seconds', 'hit_count', 'long_rally', 'ace'])
df['el_r'] = df['elapse_seconds'].round(1)
df = df.drop_duplicates(subset=['chip_id', 'date', 'session_num', 'el_r', 'hit_count'],
                        keep='first').copy()
df = df[df['group'] == 2].copy()  # HCC only
df['cond'] = 'other'
stim = df['tag'].str.contains('random_3|reseed', case=False, regex=True)
nf = df['tag'].str.contains('no_feedback', case=False)
df.loc[stim, 'cond'] = 'STIM-like'
df.loc[nf & ~stim, 'cond'] = 'NF'
print('HCC tag -> cond:')
print(df.groupby(['tag', 'cond']).size().to_string())

df['id'] = df['chip_id'].astype(str) + '|' + df['date'].astype(str) + '|' + df['session_num'].astype(str)
d = df[df['elapse_seconds'] < 300].copy(); d['half'] = 0
e = df[(df['elapse_seconds'] >= 300) & (df['elapse_seconds'] < 1200)].copy(); e['half'] = 1
w = pd.concat([d, e])
g = w.groupby(['id', 'cond', 'half']).agg(hit_count=('hit_count', 'mean'),
                                          n=('hit_count', 'size')).reset_index()
cnt = g.groupby('id')['half'].nunique()
g = g[g['id'].isin(cnt[cnt == 2].index)]

rows = []
for cond in ['STIM-like', 'NF']:
    s = g[g['cond'] == cond]
    p = s.pivot(index='id', columns='half')
    t1 = p[('hit_count', 0)].values; t2 = p[('hit_count', 1)].values
    t, pv = stats.ttest_rel(t2, t1)
    rows.append(dict(cond=cond, n_sessions=len(t1), t1_mean=t1.mean(),
                     t2_mean=t2.mean(), diff=t2.mean() - t1.mean(), t_stat=t, p_value=pv))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, 'experiment_F_feedback.csv'), index=False)
print('\n=== Exp F: STIM-like vs NF within HCC ===')
print(out.to_string())
