"""Audit #2 part 2: final-design subset, permutation nulls, plots."""
import sys, types
import pandas as pd
mod = types.ModuleType('pandas.core.indexes.numeric')
mod.Int64Index = pd.Index; mod.UInt64Index = pd.Index
mod.Float64Index = pd.Index; mod.NumericIndex = pd.Index
sys.modules['pandas.core.indexes.numeric'] = mod

import numpy as np
from scipy import stats
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RES = os.path.join(HERE, '..', 'results')
PLOTS = os.path.join(HERE, '..', 'plots')
os.makedirs(RES, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)

GROUPS = {0: 'MCC', 1: 'CTL', 2: 'HCC', 3: 'RST', 4: 'IS'}
# final DishBrain design tags (highest-density sensory info, cf. paper Fig. 4)
MAIN_TAGS = {
    0: ['ngn2_test_rate_code_random_3_reseed', 'prim_test_rate_code_random_3'],
    1: ['ctl_test_reseed', 'ctl_test_rate_code_3'],
    2: ['GFP_low_test_rate_code_random_3'],
    3: ['GFP_low_test_rate_code_random_3', 'ngn2_test_rate_code_random_3_reseed',
        'prim_test_rate_code_random_3'],
    4: ['in-silico_ratecode_spikes_reseed'],
}

def load_clean():
    df = pd.read_csv(os.path.join(DATA, 'sentience_corr.csv'),
                     usecols=['chip_id', 'group', 'tag', 'date', 'session_num',
                              'elapse_seconds', 'hit_count', 'long_rally', 'ace'])
    df['el_r'] = df['elapse_seconds'].round(1)
    df = df.drop_duplicates(subset=['chip_id', 'date', 'session_num', 'el_r', 'hit_count'],
                            keep='first').copy()
    df['id'] = (df['chip_id'].astype(str) + '|' + df['date'].astype(str) + '|' +
                df['session_num'].astype(str))
    return df

def session_half_means(df, t1_hi=300, t2_lo=300, t2_hi=1200):
    d = df[df['elapse_seconds'] < t1_hi].copy(); d['half'] = 0
    e = df[(df['elapse_seconds'] >= t2_lo) & (df['elapse_seconds'] < t2_hi)].copy(); e['half'] = 1
    w = pd.concat([d, e])
    g = w.groupby(['id', 'group', 'half']).agg(
        hit_count=('hit_count', 'mean'), ace=('ace', 'mean'),
        long_rally=('long_rally', 'mean')).reset_index()
    cnt = g.groupby('id')['half'].nunique()
    return g[g['id'].isin(cnt[cnt == 2].index)].copy()

def paired_t_by_group(g, metric='hit_count'):
    out = {}
    for grp, name in GROUPS.items():
        s = g[g['group'] == grp]
        if len(s) == 0:
            continue
        p = s.pivot(index='id', columns='half')
        t1 = p[(metric, 0)].values; t2 = p[(metric, 1)].values
        t, pv = stats.ttest_rel(t2, t1)
        out[name] = dict(t=t, p=pv, n=len(t1), dmean=(t2 - t1).mean())
    return out

def main():
    df = load_clean()
    rng = np.random.default_rng(7)

    # ---- Exp A2: final-design subset ----
    keep = []
    for grp, tags in MAIN_TAGS.items():
        keep.append(df[(df['group'] == grp) & (df['tag'].isin(tags))])
    dsub = pd.concat(keep)
    gA = session_half_means(dsub)
    rows = []
    for metric in ['hit_count', 'ace', 'long_rally']:
        for name, r in paired_t_by_group(gA, metric).items():
            rows.append(dict(experiment='A2_final_design', metric=metric, group=name, **r))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'experiment_A2_final_design.csv'), index=False)
    print('=== Exp A2: final-design tags only ===')
    print(pd.DataFrame(rows).to_string())

    # ---- Exp D1: within-session time-shuffle permutation null ----
    # For each session, permute half labels across its rallies (fixed n per half),
    # recompute paired t-stat per group. 2000 reps.
    g_full = session_half_means(df)
    rallies = df[df['elapse_seconds'] < 1200].copy()
    rallies['half_obs'] = (rallies['elapse_seconds'] >= 300).astype(int)
    sess_ids = g_full['id'].unique()
    # session -> group map, and rally indices per session
    sess_group = g_full.drop_duplicates('id').set_index('id')['group'].to_dict()
    by_sess = {sid: rallies[rallies['id'] == sid] for sid in sess_ids}
    # observed t-stats
    obs = {name: r['t'] for name, r in paired_t_by_group(g_full, 'hit_count').items()}
    NREP = 2000
    null_t = {name: [] for name in obs}
    for rep in range(NREP):
        diffs = {name: [] for name in obs}
        for sid, s in by_sess.items():
            h = s['half_obs'].values.copy()
            rng.shuffle(h)
            m0 = s.loc[h == 0, 'hit_count'].mean()
            m1 = s.loc[h == 1, 'hit_count'].mean()
            diffs[GROUPS[sess_group[sid]]].append(m1 - m0)
        for name, dlist in diffs.items():
            d = np.array(dlist)
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            null_t[name].append(t)
        if (rep + 1) % 500 == 0:
            print(f'  perm rep {rep+1}/{NREP}')
    drows = []
    for name in obs:
        nt = np.array(null_t[name])
        p_emp = (np.abs(nt) >= abs(obs[name])).mean()
        drows.append(dict(group=name, t_observed=obs[name], p_permutation=p_emp,
                          null_sd=nt.std(), n_reps=NREP))
    pd.DataFrame(drows).to_csv(os.path.join(RES, 'experiment_D1_timeshuffle.csv'), index=False)
    print('\n=== Exp D1: within-session time-shuffle null (hit_count) ===')
    print(pd.DataFrame(drows).to_string())

    # ---- Exp D2: group-label shuffle null ----
    # Shuffle group labels across sessions; statistic = min(t_MCC, t_HCC).
    ids = np.array(list(sess_group.keys()))
    grps = np.array([sess_group[i] for i in ids])
    # per-session diffs
    piv = g_full.pivot(index='id', columns='half')['hit_count']
    sdiff = (piv[1] - piv[0]).to_dict()
    def stat_of(gr):
        ts = {}
        for grp, name in GROUPS.items():
            dd = np.array([sdiff[i] for i, gg in zip(ids, gr) if gg == grp])
            if len(dd) < 3:
                ts[name] = -np.inf
            else:
                ts[name] = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
        return min(ts['MCC'], ts['HCC'])
    obs_stat = stat_of(grps)
    null_stats = []
    for rep in range(NREP):
        gp = grps.copy(); rng.shuffle(gp)
        null_stats.append(stat_of(gp))
    null_stats = np.array(null_stats)
    p_d2 = (null_stats >= obs_stat).mean()
    pd.DataFrame([dict(statistic='min(t_MCC,t_HCC)', observed=obs_stat,
                       p_group_shuffle=p_d2, n_reps=NREP)]).to_csv(
        os.path.join(RES, 'experiment_D2_groupshuffle.csv'), index=False)
    print(f'\n=== Exp D2: group-shuffle null: stat={obs_stat:.3f}, p={p_d2:.4f} ===')

    # ---- Plots ----
    mc = pd.read_csv(os.path.join(RES, 'experiment_C_minute_curves.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for grp, name in GROUPS.items():
        m = mc[(mc['group'] == grp) & (mc['minute'] < 20)].sort_values('minute')
        ax.errorbar(m['minute'], m['mean_hit'], yerr=m['sem_hit'], label=name, capsize=2)
    ax.axvline(5, color='k', ls='--', lw=1)
    ax.set_xlabel('session minute'); ax.set_ylabel('mean rally length (hits)')
    ax.set_title('Learning curves: mean rally length per minute')
    ax.legend()
    ax = axes[1]
    for grp, name in GROUPS.items():
        m = mc[(mc['group'] == grp) & (mc['minute'] < 20)].sort_values('minute')
        ax.plot(m['minute'], m['mean_n'], label=name)
    ax.axvline(5, color='k', ls='--', lw=1)
    ax.set_xlabel('session minute'); ax.set_ylabel('mean rallies played per minute')
    ax.set_title('Attrition check: rally rate over session')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'learning_curves.png'), dpi=120)
    plt.close(fig)

    # T1 vs T2 bar plot (Exp A)
    a = pd.read_csv(os.path.join(RES, 'experiment_A_replication.csv'))
    a = a[a['metric'] == 'hit_count']
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(a)); w = 0.35
    ax.bar(x - w/2, a['t1_mean'], w, label='T1 (0-5 min)')
    ax.bar(x + w/2, a['t2_mean'], w, label='T2 (6-20 min)')
    ax.set_xticks(x); ax.set_xticklabels(a['group'])
    ax.set_ylabel('mean rally length'); ax.set_title('T1 vs T2 mean rally length by group (full release, n=720 sessions)')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 't1_vs_t2.png'), dpi=120)
    plt.close(fig)
    print('\nPlots saved.')

if __name__ == '__main__':
    main()
