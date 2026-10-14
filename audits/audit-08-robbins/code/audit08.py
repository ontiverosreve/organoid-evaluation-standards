"""Audit #8 — FULL audit of Robbins et al. closed-loop organoid learning deposit.
Zenodo 17684862 (CC-BY-4.0), aggregated_experiment_data.pkl + BrainDance code.
Analyses: A metric audit of `top_decile`; B organoid x day stratified controls;
C retention (chronological split, carryover, overnight reset); D per-organoid distribution.
Env: numpy 2.5.4 / scipy 1.18.1 / pandas 3.0.6 venv (pickle needs numpy>=2). Seed 20261014.
Outputs: audit08_results.json, audit08_per_organoid.csv. All numbers computed here.
"""
import pickle, json, math
import numpy as np, pandas as pd
from collections import defaultdict
from scipy import stats

rng = np.random.default_rng(20261014)
with open('aggregated_experiment_data.pkl', 'rb') as f:
    data = pickle.load(f)

# ---------------------------------------------------------------- run table
rows = []
for eid, e in data.items():
    md = e['metadata']
    for r in (e.get('cartpole') or {}).get('runs') or []:
        rew = np.asarray((r.get('reward_log') or {}).get('reward', []), dtype=float)
        rows.append(dict(exp=eid, organoid=md['organoid'], chip=md['chip'],
                         batch=md['batch'], paradigm=md['paradigm'],
                         etype=r['exp_type'], start=str(r['start_time']),
                         day=str(r['start_time'])[:10],
                         td_stored=r.get('top_decile'), rew=rew))
R = pd.DataFrame(rows)
R['dt'] = pd.to_datetime(R['start'])
R = R.sort_values('dt').reset_index(drop=True)

def rho_ord(y):
    """Spearman-like statistic with ordinal ranks (ties by position).
    Used consistently for observed AND permutation-null values wherever an
    episode-order permutation p-value is reported (exact randomization test)."""
    n = len(y)
    if n < 3:
        return np.nan
    ry = np.argsort(np.argsort(y)).astype(float) + 1.0
    x = np.arange(n, dtype=float) + 1.0
    return float(np.corrcoef(x, ry)[0, 1])

STAT_KEYS = ['n', 'rho', 'rho_ord', 'late_early', 'entry', 'exit', 'median',
             'mean', 'sd', 'iqr', 'p90', 'tdmean']

def run_stats(rew):
    n = len(rew)
    if n < 8:
        return {k: np.nan for k in STAT_KEYS} | {'n': float(n)}
    k = max(1, n // 4)
    srt = np.sort(rew)
    kk = max(1, math.ceil(0.1 * n))
    return dict(
        n=n,
        rho=float(stats.spearmanr(np.arange(n), rew).statistic),
        rho_ord=rho_ord(rew),
        late_early=float(np.mean(rew[-k:]) - np.mean(rew[:k])),
        entry=float(np.mean(rew[:k])), exit=float(np.mean(rew[-k:])),
        median=float(np.median(rew)), mean=float(np.mean(rew)),
        sd=float(np.std(rew, ddof=1)) if n > 1 else 0.0,
        iqr=float(np.percentile(rew, 75) - np.percentile(rew, 25)),
        p90=float(np.percentile(rew, 90)),                      # = stored top_decile (verified)
        tdmean=float(np.mean(srt[-kk:])),                       # documented formula: mean of top 10%
    )

S = [run_stats(r) for r in R['rew']]
S = pd.DataFrame(S)
R = pd.concat([R.drop(columns=['rew']), S], axis=1)
REW = list(rows[i]['rew'] for i in range(len(rows)))  # keep arrays aligned with sorted R? NO -> rebuild
# R was sorted by dt; rebuild reward arrays in R's row order via start+exp+basename identity:
key2rew = {}
for eid, e in data.items():
    for r in (e.get('cartpole') or {}).get('runs') or []:
        key2rew[(eid, str(r['start_time']), r['exp_type'])] = np.asarray(
            (r.get('reward_log') or {}).get('reward', []), dtype=float)
# duplicates possible on the key; fall back to positional match by (exp, start) multimap
from collections import deque
pool = defaultdict(deque)
for eid, e in data.items():
    for r in (e.get('cartpole') or {}).get('runs') or []:
        pool[(eid, str(r['start_time']))].append(
            np.asarray((r.get('reward_log') or {}).get('reward', []), dtype=float))
REW = [pool[(row.exp, row.start)].popleft() for row in R.itertuples()]
# exact identity check: each REW array must reproduce its row's stored top_decile (= p90)
assert all(abs(np.percentile(REW[i], 90) - R['td_stored'].iloc[i]) < 1e-9 for i in range(len(R)))

OUT = {}
def rec(section, **kw):
    OUT.setdefault(section, {}).update(kw)
    print(f"[{section}] " + " | ".join(f"{k}={v}" for k, v in kw.items()))

print(f"runs total={len(R)}  with stats (n>=8 episodes)={int(R['rho'].notna().sum())}")
rec('counts', total_runs=int(len(R)),
    runs_with_stats=int(R['rho'].notna().sum()),
    rl_runs=int((R.etype == 'rl').sum()),
    organoids=int(R.organoid.nunique()), chips=int(R.chip.nunique()))

# ================================================================ A. METRIC AUDIT
# A1 reconstruction (all 426 runs, any length)
recs = []
for eid, e in data.items():
    for r in (e.get('cartpole') or {}).get('runs') or []:
        rew = np.asarray((r.get('reward_log') or {}).get('reward', []), dtype=float)
        if rew.size and r.get('top_decile') is not None:
            recs.append((abs(np.percentile(rew, 90) - r['top_decile']),
                         abs(np.mean(np.sort(rew)[-max(1, math.ceil(.1 * len(rew))):]) - r['top_decile'])))
recs = np.array(recs)
rec('A1_reconstruction', runs_checked=int(len(recs)),
    max_abs_diff_p90_linear=float(recs[:, 0].max()),
    max_abs_diff_top_decile_mean=float(recs[:, 1].max()),
    mean_abs_diff_top_decile_mean=float(recs[:, 1].mean()),
    frac_runs_top_decile_mean_diff_gt_0p05=float(np.mean(recs[:, 1] > 0.05)))

# A2 mechanical sensitivity of the metric: association with n and with spread
A2 = {}
for cond, sub in [('all', R.dropna(subset=['median'])), ('rl', R[(R.etype == 'rl')].dropna(subset=['median']))]:
    for metric in ['p90', 'tdmean', 'median']:
        for pred in ['n', 'sd', 'iqr']:
            A2[f'{cond}:{metric}~{pred}'] = float(stats.spearmanr(sub[metric], sub[pred]).statistic)
rec('A2_sensitivity', **{k: round(v, 3) for k, v in A2.items()})

# A3 location vs scale trends across chronological RL run order, per organoid
def per_org_metric_trends(metric, min_runs=5):
    rhos = {}
    sub = R[(R.etype == 'rl')].dropna(subset=[metric])
    for org, g in sub.groupby('organoid'):
        if len(g) >= min_runs:
            rhos[org] = float(stats.spearmanr(np.arange(len(g)), g[metric]).statistic)
    return rhos

def median_rho_null(metric, min_runs=5, nperm=10000):
    sub = R[(R.etype == 'rl')].dropna(subset=[metric])
    groups = [g[metric].to_numpy(float) for _, g in sub.groupby('organoid') if len(g) >= min_runs]
    def med(groups):
        return float(np.median([stats.spearmanr(np.arange(len(v)), v).statistic for v in groups]))
    obs = med(groups)
    cnt = 0
    for _ in range(nperm):
        if med([rng.permutation(v) for v in groups]) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (nperm + 1)

A3 = {}
for metric in ['p90', 'tdmean', 'median', 'mean', 'sd', 'iqr']:
    obs, p = median_rho_null(metric)
    A3[metric] = dict(median_per_org_rho=obs, shuffle_p=p,
                      per_org={k: round(v, 3) for k, v in per_org_metric_trends(metric).items()})
    rec('A3_trend_' + metric, median_rho=round(obs, 3), p=round(p, 4))

# A4 held-out split-half replication of the within-run trend (interleaved halves)
def half_stats(rew):
    if len(rew) < 16:
        return None
    out = {}
    for name, idx in [('odd', np.arange(1, len(rew), 2)), ('even', np.arange(0, len(rew), 2))]:
        h = rew[idx]
        k = max(1, len(h) // 4)
        out[name] = (rho_ord(h), float(np.mean(h[-k:]) - np.mean(h[:k])))
    return out

HALF = {}
for i, rew in enumerate(REW):
    hs = half_stats(rew)
    if hs:
        HALF[i] = hs
A4 = {}
for cond in ['rl', 'random', 'none']:
    idxs = [i for i in HALF if R['etype'].iloc[i] == cond]
    if len(idxs) < 5:
        continue
    ro = np.array([HALF[i]['odd'][0] for i in idxs]); re_ = np.array([HALF[i]['even'][0] for i in idxs])
    leo = np.array([HALF[i]['odd'][1] for i in idxs]); lee = np.array([HALF[i]['even'][1] for i in idxs])
    A4[cond] = dict(runs=int(len(idxs)), median_rho_odd=float(np.median(ro)),
                    median_rho_even=float(np.median(re_)),
                    median_le_odd=float(np.median(leo)), median_le_even=float(np.median(lee)),
                    frac_both_rho_pos=float(np.mean((ro > 0) & (re_ > 0))),
                    corr_rho_halves=float(stats.spearmanr(ro, re_).statistic))
    rec('A4_half_' + cond, runs=len(idxs), med_rho_odd=round(A4[cond]['median_rho_odd'], 3),
        med_rho_even=round(A4[cond]['median_rho_even'], 3),
        frac_both_pos=round(A4[cond]['frac_both_rho_pos'], 2),
        corr_halves=round(A4[cond]['corr_rho_halves'], 3))
# A4 null: episode-order permutation, RL runs, stat = median rho_ord averaged over halves
rl_half_idx = [i for i in HALF if R['etype'].iloc[i] == 'rl']
obs_a4 = float(np.mean([np.median([HALF[i][h][0] for i in rl_half_idx]) for h in ['odd', 'even']]))
cnt = 0; NPERM_A4 = 1000
for _ in range(NPERM_A4):
    meds = []
    for h, off in [('odd', 1), ('even', 0)]:
        vals = []
        for i in rl_half_idx:
            pr = rng.permutation(REW[i])
            vals.append(rho_ord(pr[np.arange(off, len(pr), 2)]))
        meds.append(np.median(vals))
    if np.mean(meds) >= obs_a4:
        cnt += 1
rec('A4_null_RL', obs_mean_of_half_median_rhos=round(obs_a4, 4), perm_p=round((cnt + 1) / (NPERM_A4 + 1), 4))

# A5 across-run chronology recomputed under alternative metrics (see A3 for p90/median/mean/tdmean)
rec('A5_chronology_metrics',
    p90_median_rho=round(A3['p90']['median_per_org_rho'], 3), p90_p=round(A3['p90']['shuffle_p'], 4),
    tdmean_median_rho=round(A3['tdmean']['median_per_org_rho'], 3), tdmean_p=round(A3['tdmean']['shuffle_p'], 4),
    median_median_rho=round(A3['median']['median_per_org_rho'], 3), median_p=round(A3['median']['shuffle_p'], 4),
    mean_median_rho=round(A3['mean']['median_per_org_rho'], 3), mean_p=round(A3['mean']['shuffle_p'], 4))

# ================================================================ B. STRATIFIED CONTROLS (organoid x day)
R['stratum'] = R['organoid'] + '|' + R['day']
V = R.dropna(subset=['rho']).copy()
# B1: RL trend within strata (strata with >=2 RL runs), episode-order permutation null
strata_rl = [g for _, g in V[V.etype == 'rl'].groupby('stratum') if len(g) >= 2]
def stratum_stat(frame_list, col):
    return float(np.median([np.median(g[col]) for g in frame_list]))

def fast_stats(rew):
    """Only the two permutation-loop statistics: late-early and rho_ord (no scipy)."""
    n = len(rew)
    k = max(1, n // 4)
    return float(np.mean(rew[-k:]) - np.mean(rew[:k])), rho_ord(rew)

obs_b1_le = stratum_stat(strata_rl, 'late_early')
obs_b1_rho = float(np.median([np.median(g['rho_ord']) for g in strata_rl]))
rl_idx_by_stratum = [[i for i in g.index] for g in strata_rl]
pos = {idx: j for j, idx in enumerate(R.index)}
cnt_le = cnt_rho = 0; NPERM_B1 = 1000
for _ in range(NPERM_B1):
    le_s, rho_s = [], []
    for idxs in rl_idx_by_stratum:
        les, rhos = [], []
        for idx in idxs:
            le, ro = fast_stats(rng.permutation(REW[pos[idx]]))
            les.append(le); rhos.append(ro)
        le_s.append(np.median(les)); rho_s.append(np.median(rhos))
    if np.median(le_s) >= obs_b1_le: cnt_le += 1
    if np.median(rho_s) >= obs_b1_rho: cnt_rho += 1
rec('B1_stratified_RL_trend', strata=int(len(strata_rl)),
    rl_runs_covered=int(sum(len(x) for x in rl_idx_by_stratum)),
    median_stratum_median_late_early=round(obs_b1_le, 3), perm_p_le=round((cnt_le + 1) / (NPERM_B1 + 1), 4),
    median_stratum_median_rho=round(obs_b1_rho, 4), perm_p_rho=round((cnt_rho + 1) / (NPERM_B1 + 1), 4))

# B2: RL-vs-control contrast within mixed strata, label permutation
def mixed_strata(frame):
    out = []
    for _, g in frame.groupby('stratum'):
        rl = g[g.etype == 'rl']; ct = g[g.etype.isin(['random', 'none'])]
        if len(rl) >= 1 and len(ct) >= 1:
            out.append((rl, ct))
    return out

def contrast_test(pairs, col, nperm=10000):
    diffs = np.array([rl[col].mean() - ct[col].mean() for rl, ct in pairs])
    obs = float(np.median(diffs))
    allstats = [np.concatenate([rl[col].to_numpy(float), ct[col].to_numpy(float)]) for rl, ct in pairs]
    ns = [len(rl) for rl, _ in pairs]
    cnt = 0
    for _ in range(nperm):
        d = []
        for vals, n1 in zip(allstats, ns):
            pv = rng.permutation(vals)
            d.append(pv[:n1].mean() - pv[n1:].mean())
        if np.median(d) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (nperm + 1), float(np.mean(diffs > 0)), diffs

pairs_all = mixed_strata(V)
for col in ['rho', 'late_early']:
    obs, p, fpos, _ = contrast_test(pairs_all, col)
    rec('B2_contrast_all_' + col, strata=int(len(pairs_all)), median_diff=round(obs, 4),
        label_perm_p=round(p, 4), frac_strata_RL_gt_control=round(fpos, 2))
pairs_cyc = mixed_strata(V[V.paradigm == 'cycled'])
for col in ['rho', 'late_early']:
    obs, p, fpos, _ = contrast_test(pairs_cyc, col)
    rec('B2_contrast_cycled_' + col, strata=int(len(pairs_cyc)), median_diff=round(obs, 4),
        label_perm_p=round(p, 4), frac_strata_RL_gt_control=round(fpos, 2))
# control-type split (random-only strata, none-only strata)
for ctype in ['random', 'none']:
    prs = []
    for _, g in V.groupby('stratum'):
        rl = g[g.etype == 'rl']; ct = g[g.etype == ctype]
        if len(rl) >= 1 and len(ct) >= 1:
            prs.append((rl, ct))
    if len(prs) >= 4:
        obs, p, fpos, _ = contrast_test(prs, 'rho')
        rec(f'B2_contrast_vs_{ctype}_rho', strata=int(len(prs)), median_diff=round(obs, 4),
            label_perm_p=round(p, 4), frac_strata_RL_gt_control=round(fpos, 2))

# ================================================================ C. RETENTION
RL = V[V.etype == 'rl'].copy()
# C1 chronological split-half per organoid (metrics: p90, median, mean)
C1 = {}
for metric in ['p90', 'median', 'mean']:
    diffs, orgs = [], []
    for org, g in RL.groupby('organoid'):
        if len(g) >= 6:
            h = len(g) // 2
            diffs.append(float(g[metric].iloc[h:].mean() - g[metric].iloc[:h].mean()))
            orgs.append(org)
    diffs = np.array(diffs)
    C1[metric] = dict(organoids=int(len(diffs)), median_diff=float(np.median(diffs)),
                      frac_positive=float(np.mean(diffs > 0)),
                      wilcoxon_p=float(stats.wilcoxon(diffs).pvalue) if np.any(diffs != 0) else float('nan'),
                      per_org={o: round(float(d), 2) for o, d in zip(orgs, diffs)})
    rec('C1_splithalf_' + metric, organoids=len(diffs), median_diff=round(float(np.median(diffs)), 3),
        frac_pos=round(float(np.mean(diffs > 0)), 2), wilcoxon_p=round(C1[metric]['wilcoxon_p'], 4))
# C2 carryover: entry (first-quartile) level vs chronological order; exit->next-entry gap
entry_trends = per_org_metric_trends('entry', min_runs=5)
obs_e, p_e = median_rho_null('entry')
rec('C2_entry_trend', median_rho=round(obs_e, 3), shuffle_p=round(p_e, 4),
    per_org={k: round(v, 2) for k, v in entry_trends.items()})
exit_trends = per_org_metric_trends('exit', min_runs=5)
obs_x, p_x = median_rho_null('exit')
rec('C2_exit_trend', median_rho=round(obs_x, 3), shuffle_p=round(p_x, 4))
gaps, gains = [], []
gap_rows = []
for org, g in RL.groupby('organoid'):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        gap = float(g['entry'].iloc[i + 1] - g['exit'].iloc[i])
        gain = float(g['exit'].iloc[i] - g['entry'].iloc[i])
        same_day = g['day'].iloc[i + 1] == g['day'].iloc[i]
        gaps.append(gap); gains.append(gain)
        gap_rows.append((org, same_day, gap, gain))
gaps = np.array(gaps); gains = np.array(gains)
GR = pd.DataFrame(gap_rows, columns=['organoid', 'same_day', 'gap', 'gain'])
rec('C2_carryover_gap', consecutive_pairs=int(len(gaps)), median_gap=round(float(np.median(gaps)), 3),
    median_within_run_gain=round(float(np.median(gains)), 3),
    retention_ratio_median_gap_over_gain=round(float(np.median(gaps) / np.median(gains)), 3) if np.median(gains) != 0 else None,
    wilcoxon_gap_vs_0_p=float(stats.wilcoxon(gaps).pvalue))
for sd, lab in [(True, 'same_day'), (False, 'overnight_or_later_day')]:
    sub = GR[GR.same_day == sd]
    if len(sub) >= 5:
        rec('C2_gap_' + lab, pairs=int(len(sub)), median_gap=round(float(sub.gap.median()), 3),
            median_gain=round(float(sub.gain.median()), 3),
            wilcoxon_p=float(stats.wilcoxon(sub.gap).pvalue))
# organoid-level median gap (cluster-safe summary)
org_gap = GR.groupby('organoid')['gap'].median()
rec('C2_gap_per_organoid', organoids=int(len(org_gap)), median_of_medians=round(float(org_gap.median()), 3),
    frac_orgs_positive=float(np.mean(org_gap > 0)))

# ================================================================ D. PER-ORGANOID DISTRIBUTION
D = []
for org, g in R.groupby('organoid'):
    rl = g[g.etype == 'rl'].dropna(subset=['rho'])
    ct = g[g.etype.isin(['random', 'none'])].dropna(subset=['rho'])
    row = dict(organoid=org, chip=g['chip'].iloc[0], paradigms=','.join(sorted(g.paradigm.unique())),
               n_rl=int(len(rl)), n_ctrl=int(len(ct)), days=int(g.day.nunique()),
               median_rl_rho=float(rl.rho.median()) if len(rl) else np.nan,
               median_rl_le=float(rl.late_early.median()) if len(rl) else np.nan,
               frac_rl_rho_pos=float(np.mean(rl.rho > 0)) if len(rl) else np.nan,
               median_ctrl_rho=float(ct.rho.median()) if len(ct) else np.nan,
               median_ctrl_le=float(ct.late_early.median()) if len(ct) else np.nan)
    # per-organoid episode-shuffle permutation p for median late-early & median rho_ord
    if len(rl) >= 3:
        idxs = list(rl.index); obs_le = float(rl.late_early.median()); obs_r = float(rl.rho_ord.median())
        cle = cr = 0; NP = 2000
        for _ in range(NP):
            les, rhos = [], []
            for idx in idxs:
                le, ro = fast_stats(rng.permutation(REW[pos[idx]]))
                les.append(le); rhos.append(ro)
            if np.median(les) >= obs_le: cle += 1
            if np.median(rhos) >= obs_r: cr += 1
        row['perm_p_le'] = (cle + 1) / (NP + 1); row['perm_p_rho'] = (cr + 1) / (NP + 1)
    D.append(row)
D = pd.DataFrame(D).sort_values('n_rl', ascending=False)
D.to_csv('audit08_per_organoid.csv', index=False)
elig = D[D.n_rl >= 3]
rec('D_distribution', organoids_total=int(len(D)), organoids_with_ge3_rl=int(len(elig)),
    n_median_rl_rho_pos=int((elig.median_rl_rho > 0).sum()),
    n_individually_sig_le_p05=int((elig.perm_p_le < 0.05).sum()),
    n_individually_sig_rho_p05=int((elig.perm_p_rho < 0.05).sum()),
    n_with_controls=int((elig.n_ctrl > 0).sum()),
    n_rl_gt_ctrl_rho=int((elig[elig.n_ctrl > 0].median_rl_rho > elig[elig.n_ctrl > 0].median_ctrl_rho).sum()))
print(D.to_string(index=False))

OUT['A3'] = A3; OUT['A4'] = A4; OUT['C1'] = C1; OUT['D_table'] = D.where(pd.notna(D), None).to_dict('records')
with open('audit08_results.json', 'w') as f:
    json.dump(OUT, f, indent=1, default=str)
print("WROTE audit08_results.json + audit08_per_organoid.csv")
