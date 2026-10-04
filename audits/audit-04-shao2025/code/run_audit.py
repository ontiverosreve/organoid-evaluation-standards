"""Audit #4 (Shao et al. 2025): honest-controls analyses.

A. Pattern decoding within culture+day (replication of the paper's claim),
   stratified 5-fold CV + the paper's chronological block scheme.
B. Artifact controls: same decoding in artifact-only / no-blank / early-blank windows.
C. Session-identity nulls: culture-ID decoding, day-ID decoding, cross-day
   transfer, permutation nulls.
D. Training claim: paired day3-day1 accuracy across cultures + excitability check.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from scipy import stats

OUT = os.path.expanduser('~/workspace/reservoir-eeg/work/audit_shao2025')
CACHE = os.path.join(OUT, 'cache')
RES = os.path.join(OUT, 'results')
os.makedirs(RES, exist_ok=True)

WINDOWS = ['paper', 'noblank', 'artifact', 'early', 'wide', 'vwide']

def load(exp, cult, day):
    p = os.path.join(CACHE, f'{exp}_c{cult}_d{day}.npz')
    if not os.path.exists(p): return None
    return dict(np.load(p, allow_pickle=True))

def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000))

def cv_acc(X, y, n_splits=5, seed=0):
    X = np.asarray(X, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        m = clf().fit(X[tr], y[tr])
        accs.append(m.score(X[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))

def block_acc(X, y, patid, seed=0):
    """Paper's scheme: per pattern, 5 chronological blocks of 80 trials;
    train on first 60 of each block, test on last 20."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y); patid = np.asarray(patid)
    accs = []
    for b in range(5):
        tr_idx, te_idx = [], []
        for p in np.unique(patid):
            idx = np.where(patid == p)[0]
            # chronological order = file/trial order as built
            blk = idx[b*80:(b+1)*80]
            tr_idx += blk[:60].tolist(); te_idx += blk[60:].tolist()
        tr_idx = np.array(tr_idx); te_idx = np.array(te_idx)
        if len(te_idx) == 0: continue
        m = clf().fit(X[tr_idx], y[tr_idx])
        accs.append(m.score(X[te_idx], y[te_idx]))
    return float(np.mean(accs)) if accs else np.nan, float(np.std(accs)) if accs else np.nan

def fam_binary(fams):
    return np.array([0 if f == 'L' else 1 for f in fams])  # L vs non-L(X)

def main():
    rows_pat, rows_cult, rows_day, rows_xfer, rows_perm = [], [], [], [], []
    train_rows = []
    elec_cache = {}  # (exp,cult,day) -> (E_paper, units)

    def electrode_agg(d):
        key = None
        X = np.asarray(d['X_paper'], dtype=np.float64)
        units = np.asarray(d['units'])
        E = np.zeros((X.shape[0], 60))
        for j, eid in enumerate(units[:, 0]):
            if 0 <= eid < 60:
                E[:, eid] += X[:, j]
        return E

    # ---------- A & B: pattern decoding per culture-day, all windows ----------
    for exp in ['two_pattern', 'six_pattern']:
        for cult in range(1, 11):
            for day in range(1, 4):
                d = load(exp, cult, day)
                if d is None: continue
                npat = len(np.unique(d['patid']))
                if npat < 2: continue
                ypat = np.asarray(d['patid'])
                yfam = np.asarray(d['fam'])
                for w in WINDOWS:
                    X = d['X_' + w]
                    a, s = cv_acc(X, ypat)
                    rows_pat.append(dict(experiment=exp, culture=cult, day=day,
                        n_patterns=npat, window=w, scheme='strat5fold',
                        accuracy=round(a, 4), sd=round(s, 4),
                        chance=round(1.0/npat, 4),
                        n_trials=X.shape[0], n_units=X.shape[1]))
                    ab, sb = block_acc(X, ypat, ypat)
                    rows_pat.append(dict(experiment=exp, culture=cult, day=day,
                        n_patterns=npat, window=w, scheme='block60_20',
                        accuracy=round(ab, 4), sd=round(sb, 4),
                        chance=round(1.0/npat, 4),
                        n_trials=X.shape[0], n_units=X.shape[1]))
                # binary L-vs-X for two_pattern
                if exp == 'two_pattern':
                    yb = fam_binary(yfam)
                    if len(np.unique(yb)) == 2:
                        a, s = cv_acc(d['X_paper'], yb)
                        rows_pat.append(dict(experiment=exp, culture=cult, day=day,
                            n_patterns=2, window='paper', scheme='binary_L_vs_X',
                            accuracy=round(a, 4), sd=round(s, 4), chance=0.5,
                            n_trials=len(yb), n_units=d['X_paper'].shape[1]))
                # mean firing rate (excitability): spikes per trial per unit in paper window
                rate = float(np.asarray(d['X_paper']).sum() / d['X_paper'].shape[0] / d['X_paper'].shape[1])
                # day-1..3 accuracy on paper window for training claim
                a, s = cv_acc(d['X_paper'], ypat)
                train_rows.append(dict(experiment=exp, culture=cult, day=day,
                    n_patterns=npat, acc_paper=round(a,4), mean_rate=round(rate,4),
                    n_trials=d['X_paper'].shape[0], n_units=d['X_paper'].shape[1]))
                elec_cache[(exp, cult, day)] = electrode_agg(d)
                print(f'A {exp} c{cult} d{day}: npat={npat} acc(paper)={a:.3f}', flush=True)

    pat_df = pd.DataFrame(rows_pat)
    pat_df.to_csv(os.path.join(RES, 'decoding_pattern.csv'), index=False)

    # ---------- C1: culture-ID decoding (pooled days, electrode-aggregated) ----------
    for exp in ['two_pattern', 'six_pattern']:
        Xs, ys = [], []
        for cult in range(1, 11):
            for day in range(1, 4):
                e = elec_cache.get((exp, cult, day))
                if e is None: continue
                Xs.append(e); ys += [cult] * e.shape[0]
        X = np.vstack(Xs); y = np.array(ys)
        a, s = cv_acc(X, y)
        rows_cult.append(dict(experiment=exp, scheme='strat5fold', window='paper_elec',
            accuracy=round(a,4), sd=round(s,4), chance=0.1, n_trials=X.shape[0]))
        print(f'C1 {exp}: culture-ID acc={a:.4f} (chance 0.10)', flush=True)
    pd.DataFrame(rows_cult).to_csv(os.path.join(RES, 'decoding_culture_id.csv'), index=False)

    # ---------- C2: day-ID decoding within culture ----------
    for exp in ['two_pattern', 'six_pattern']:
        for cult in range(1, 11):
            Xs, ys = [], []
            for day in range(1, 4):
                e = elec_cache.get((exp, cult, day))
                if e is None: continue
                Xs.append(e); ys += [day] * e.shape[0]
            if len(set(ys)) < 2: continue
            X = np.vstack(Xs); y = np.array(ys)
            a, s = cv_acc(X, y)
            rows_day.append(dict(experiment=exp, culture=cult,
                accuracy=round(a,4), sd=round(s,4), chance=round(1/len(set(ys)),4),
                n_days=len(set(ys)), n_trials=X.shape[0]))
    day_df = pd.DataFrame(rows_day)
    day_df.to_csv(os.path.join(RES, 'decoding_day_id.csv'), index=False)
    print('C2 day-ID decoding done', flush=True)

    # ---------- C3: cross-day transfer (train day1 -> test day3, and reverse) ----------
    for exp in ['two_pattern', 'six_pattern']:
        for cult in range(1, 11):
            d1, d3 = load(exp, cult, 1), load(exp, cult, 3)
            if d1 is None or d3 is None: continue
            u1 = {tuple(u): i for i, u in enumerate(np.asarray(d1['units']))}
            u3 = {tuple(u): i for i, u in enumerate(np.asarray(d3['units']))}
            shared = sorted(set(u1) & set(u3))
            if len(shared) < 5: continue
            i1 = [u1[u] for u in shared]; i3 = [u3[u] for u in shared]
            X1 = np.asarray(d1['X_paper'], dtype=np.float64)[:, i1]
            X3 = np.asarray(d3['X_paper'], dtype=np.float64)[:, i3]
            y1, y3 = np.asarray(d1['patid']), np.asarray(d3['patid'])
            # align label spaces: patterns present on both days
            common_pats = sorted(set(y1) & set(y3))
            if len(common_pats) < 2: continue
            m12 = np.isin(y1, common_pats); m3 = np.isin(y3, common_pats)
            # remap to 0..k-1
            remap = {p: i for i, p in enumerate(common_pats)}
            yy1 = np.array([remap[p] for p in y1[m12]]); yy3 = np.array([remap[p] for p in y3[m3]])
            m = clf().fit(X1[m12], yy1)
            a13 = m.score(X3[m3], yy3)
            m = clf().fit(X3[m3], yy3)
            a31 = m.score(X1[m12], yy1)
            rows_xfer.append(dict(experiment=exp, culture=cult, n_shared_units=len(shared),
                n_patterns=len(common_pats), train_d1_test_d3=round(a13,4),
                train_d3_test_d1=round(a31,4), chance=round(1/len(common_pats),4)))
            print(f'C3 {exp} c{cult}: d1->d3={a13:.3f} d3->d1={a31:.3f} (chance {1/len(common_pats):.3f})', flush=True)
    pd.DataFrame(rows_xfer).to_csv(os.path.join(RES, 'cross_day_transfer.csv'), index=False)

    # ---------- C4: permutation nulls for main decoding ----------
    # 20 shuffles x single stratified 80/20 split (fast, honest empirical chance)
    from sklearn.model_selection import train_test_split
    rng = np.random.RandomState(0)
    for exp in ['two_pattern', 'six_pattern']:
        for cult in range(1, 11):
            for day in range(1, 4):
                d = load(exp, cult, day)
                if d is None: continue
                X = np.asarray(d['X_paper'], dtype=np.float64); y = np.asarray(d['patid'])
                if len(np.unique(y)) < 2: continue
                nulls = []
                for r in range(20):
                    yr = rng.permutation(y)
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, yr, test_size=0.2, stratify=yr, random_state=r)
                    m = clf().fit(Xtr, ytr)
                    nulls.append(m.score(Xte, yte))
                rows_perm.append(dict(experiment=exp, culture=cult, day=day,
                    null_mean=round(float(np.mean(nulls)),4),
                    null_max=round(float(np.max(nulls)),4),
                    chance=round(1/len(np.unique(y)),4)))
                print(f'C4 {exp} c{cult} d{day}: null_mean={np.mean(nulls):.3f}', flush=True)
    pd.DataFrame(rows_perm).to_csv(os.path.join(RES, 'permutation_nulls.csv'), index=False)
    print('C4 permutation nulls done', flush=True)

    # ---------- D: training claim ----------
    tr = pd.DataFrame(train_rows)
    tr.to_csv(os.path.join(RES, 'training_claim.csv'), index=False)
    for exp in ['two_pattern', 'six_pattern']:
        sub = tr[tr.experiment == exp].pivot(index='culture', columns='day', values='acc_paper')
        sub = sub.dropna()
        if 1 in sub.columns and 3 in sub.columns and len(sub) >= 3:
            d1, d3 = sub[1].values, sub[3].values
            t, p = stats.ttest_rel(d3, d1)
            print(f'D {exp}: n={len(sub)} day1={d1.mean():.4f} day3={d3.mean():.4f} '
                  f'delta={np.mean(d3-d1):+.4f} t={t:.3f} p={p:.4g}', flush=True)
            rsub = tr[tr.experiment == exp].pivot(index='culture', columns='day', values='mean_rate').dropna()
            if 1 in rsub.columns and 3 in rsub.columns:
                rd = rsub[3].values - rsub[1].values
                ad = d3 - d1
                if len(sub) >= 4 and np.std(rd) > 0 and np.std(ad) > 0:
                    rho, pr = stats.pearsonr(rd, ad)
                    print(f'   excitability: rate day1={rsub[1].mean():.4f} day3={rsub[3].mean():.4f} '
                          f'corr(delta_rate, delta_acc)={rho:.3f} p={pr:.4g}', flush=True)
    print('ALL DONE')

if __name__ == '__main__':
    main()
