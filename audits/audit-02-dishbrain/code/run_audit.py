"""Audit #2: Kagan et al. 2022 Neuron (DishBrain Pong) — honest-controls audit.

Target claim: in-vitro neuronal cultures (MCC=mouse, HCC=human cortical cells on
HD-MEA) show learning in closed-loop Pong — average rally length increases from
T1 (first 5 min) to T2 (last 15 min) — while controls (CTL media-only, RST rest,
IS in-silico random paddle) do not.

Data: OSF 5u6qv, Data/in_vitro_cells_sentience_corr.csv (md5 753a94e3fbf6958dd517f7053993e97a).
NOTE: these are 2D neural-culture monolayers, NOT 3D organoids.

Data-quality finding: the released CSV contains ~15% near-duplicate rally rows
(same chip/date/session/rounded-elapse/hit_count, elapse differing by ~ms) and a
precomputed 'half' column inconsistent with a 300 s split. We deduplicate and
recompute half = elapse_seconds > 300 ourselves (paper: T1 = 0-5 min, T2 = 6-20).
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
os.makedirs(RES, exist_ok=True)

GROUPS = {0: 'MCC', 1: 'CTL', 2: 'HCC', 3: 'RST', 4: 'IS'}

def load_clean():
    df = pd.read_csv(os.path.join(DATA, 'sentience_corr.csv'),
                     usecols=['chip_id', 'group', 'tag', 'date', 'session_num',
                              'elapse_seconds', 'hit_count', 'long_rally', 'ace'])
    n0 = len(df)
    df['el_r'] = df['elapse_seconds'].round(1)
    df = df.drop_duplicates(subset=['chip_id', 'date', 'session_num', 'el_r', 'hit_count'],
                            keep='first').copy()
    n1 = len(df)
    print(f'dedup: {n0} -> {n1} rows ({n0-n1} near-dupes removed)')
    df['half'] = (df['elapse_seconds'] > 300).astype(int)  # paper: T1=0-5min, T2=6-20min
    df['id'] = (df['chip_id'].astype(str) + '|' + df['date'].astype(str) + '|' +
                df['session_num'].astype(str))
    return df

def session_half_means(df, t1_lo=0, t1_hi=300, t2_lo=300, t2_hi=1200, label=''):
    """One row per (session, half): mean of rally metrics in each window."""
    d = df[(df['elapse_seconds'] >= t1_lo) & (df['elapse_seconds'] < t1_hi)].copy()
    d['half'] = 0
    e = df[(df['elapse_seconds'] >= t2_lo) & (df['elapse_seconds'] < t2_hi)].copy()
    e['half'] = 1
    w = pd.concat([d, e])
    g = w.groupby(['id', 'group', 'half']).agg(
        hit_count=('hit_count', 'mean'), ace=('ace', 'mean'),
        long_rally=('long_rally', 'mean'), n_rallies=('hit_count', 'size')).reset_index()
    # drop false starts: sessions missing either half (paper's notebook does this)
    cnt = g.groupby('id')['half'].nunique()
    g = g[g['id'].isin(cnt[cnt == 2].index)].copy()
    return g

def paired_tests(g):
    """Paired t-test T2 vs T1 within each group, per metric."""
    rows = []
    for grp, name in GROUPS.items():
        s = g[g['group'] == grp]
        if len(s) == 0:
            continue
        p = s.pivot(index='id', columns='half')
        for metric in ['hit_count', 'ace', 'long_rally']:
            t1 = p[(metric, 0)].values
            t2 = p[(metric, 1)].values
            t, pv = stats.ttest_rel(t2, t1)
            d = t2 - t1
            rows.append(dict(group=name, metric=metric, n_sessions=len(t1),
                             t1_mean=t1.mean(), t2_mean=t2.mean(),
                             diff_mean=d.mean(), diff_median=np.median(d),
                             t_stat=t, p_value=pv))
    return pd.DataFrame(rows)

def main():
    df = load_clean()

    # ---- session inventory ----
    inv = df.groupby('group').agg(n_rallies=('hit_count', 'size'),
                                  n_sessions=('id', 'nunique'),
                                  n_chips=('chip_id', 'nunique'),
                                  max_elapse=('elapse_seconds', 'max'),
                                  mean_elapse=('elapse_seconds', 'mean')).reset_index()
    inv['group_name'] = inv['group'].map(GROUPS)
    inv.to_csv(os.path.join(RES, 'experiment_inventory.csv'), index=False)
    print(inv.to_string())

    # ---- Exp A: replicate paper's T1 (0-5) vs T2 (6-20) ----
    gA = session_half_means(df)
    resA = paired_tests(gA)
    resA['experiment'] = 'A_paper_windows'
    resA.to_csv(os.path.join(RES, 'experiment_A_replication.csv'), index=False)
    print('\n=== Exp A: paper windows (0-5 vs 6-20 min) ===')
    print(resA[['group', 'metric', 'n_sessions', 't1_mean', 't2_mean', 't_stat', 'p_value']].to_string())

    # ---- Exp B: equal windows ----
    gB1 = session_half_means(df, t2_lo=900, t2_hi=1200)   # first 5 vs last 5
    resB1 = paired_tests(gB1); resB1['experiment'] = 'B_first5_vs_last5'
    gB2 = session_half_means(df, t2_lo=300, t2_hi=600)    # first 5 vs minutes 6-10
    resB2 = paired_tests(gB2); resB2['experiment'] = 'B_first5_vs_min6to10'
    resB = pd.concat([resB1, resB2], ignore_index=True)
    resB.to_csv(os.path.join(RES, 'experiment_B_equal_windows.csv'), index=False)
    print('\n=== Exp B: equal windows ===')
    print(resB[['experiment', 'group', 'metric', 'n_sessions', 't1_mean', 't2_mean', 't_stat', 'p_value']].to_string())

    # ---- Exp C: minute-by-minute curves + attrition ----
    df['minute'] = (df['elapse_seconds'] / 60).astype(int).clip(0, 20)
    sm = df.groupby(['id', 'group', 'minute']).agg(hit=('hit_count', 'mean'),
                                                    n=('hit_count', 'size')).reset_index()
    mc = sm.groupby(['group', 'minute']).agg(mean_hit=('hit', 'mean'),
                                             sem_hit=('hit', lambda x: stats.sem(x, nan_policy='omit')),
                                             mean_n=('n', 'mean'),
                                             n_sessions=('hit', 'size')).reset_index()
    mc['group_name'] = mc['group'].map(GROUPS)
    mc.to_csv(os.path.join(RES, 'experiment_C_minute_curves.csv'), index=False)
    # per-group linear trend of mean_hit over minutes 0..19
    trows = []
    for grp, name in GROUPS.items():
        m = mc[(mc['group'] == grp) & (mc['minute'] < 20)]
        slope, icept, r, p, se = stats.linregress(m['minute'], m['mean_hit'])
        trows.append(dict(group=name, slope_per_min=slope, r_value=r, p_value=p,
                          n_minutes=len(m)))
    pd.DataFrame(trows).to_csv(os.path.join(RES, 'experiment_C_trends.csv'), index=False)
    print('\n=== Exp C: per-group linear trends ===')
    print(pd.DataFrame(trows).to_string())

    # ---- Exp E: baselines + per-chip consistency ----
    piv = gA.pivot(index='id', columns='half')
    erows = []
    for grp, name in GROUPS.items():
        s = gA[gA['group'] == grp]
        if len(s) == 0:
            continue
        p = s.pivot(index='id', columns='half')
        t1 = p[('hit_count', 0)]
        erows.append(dict(group=name, n=len(t1), t1_mean=t1.mean(), t1_sd=t1.std()))
    pd.DataFrame(erows).to_csv(os.path.join(RES, 'experiment_E_baselines.csv'), index=False)
    print('\n=== Exp E: T1 baselines ===')
    print(pd.DataFrame(erows).to_string())

    # per-chip consistency: chips with >=4 sessions, fraction with T2>T1
    gA['chip'] = gA['id'].str.split('|').str[0]
    crows = []
    for (grp, chip), s in gA.groupby(['group', 'chip']):
        p = s.pivot(index='id', columns='half')
        if len(p) < 4:
            continue
        d = p[('hit_count', 1)] - p[('hit_count', 0)]
        npos = (d > 0).sum()
        # sign test
        pv = stats.binomtest(npos, len(d), 0.5).pvalue
        crows.append(dict(group=GROUPS[grp], chip=chip, n_sessions=len(d),
                          frac_improved=npos / len(d), sign_p=pv,
                          mean_diff=d.mean()))
    pd.DataFrame(crows).to_csv(os.path.join(RES, 'experiment_E_chip_consistency.csv'), index=False)
    print('\n=== Exp E: per-chip consistency (>=4 sessions) ===')
    print(pd.DataFrame(crows).to_string())

    print('\nDone. Results in', RES)

if __name__ == '__main__':
    main()
