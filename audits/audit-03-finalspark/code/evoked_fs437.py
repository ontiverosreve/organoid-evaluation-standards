#!/usr/bin/env python
"""Workstream 3 of Audit #3 (honest-controls audit of FinalSpark FS437 whole-life MEA).

CONFIDENTIAL: raw data stays in data/fs437_export/; only aggregates leave this script.

Task A: per-protocol evoked ratios (P1/P2/P3) + stim-shuffled honest nulls (20 reps);
        protocol decoding P1 vs P2 vs P3 from [0,50] ms 32-electrode spike counts
        (logistic regression + canonical 150-node leaky reservoir), 5-fold CV,
        artifact-blanked [10,50] ms variant, 20-rep label-shuffle nulls.
Task B: cross-day transfer (standard v0.3 section 3.7): P2/P3 decoder trained on
        June 7-8 -> tested on June 9 (P2 anchor class), and P1/P2 trained on
        June 9 -> tested on June 7-8 (P2 anchor); day-ID decoder on baseline rate
        vectors as the session-identity confound check.
Task C: dead-culture negative control (June 11 stims): evoked ratio + live-trained
        protocol decoder applied to dead-culture stims.

Canonical pipeline: code/audit_lib.py (150 nodes, spectral radius 0.9, sparsity 0.1,
alphas [0.005, 0.01, 0.02], seeds [7, 11, 22], quality-weighted ridge lam 1e-3,
20-rep nulls). The reservoir is adapted to per-stim trials with a vectorized
implementation of the identical recurrence; ridge readout extended to 3 classes.

Outputs: results/evoked_fs437.csv, results/transfer_fs437.csv,
         results/dead_culture_fs437.csv; run log appended to results/run_log.txt.
"""
import os, sys, time, csv
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as AL

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
HDF = os.path.join(WORK, 'data', 'fs437_export', 'fs437_package.hdf5')
RES = os.path.join(WORK, 'results')
MS = 1_000_000  # ns per ms
RNG = np.random.default_rng(20261007)
NREP = AL.N_SHUFFLE  # 20

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------- load
log("loading events + stimulations")
ev = pd.read_hdf(HDF, 'fs437_wholelife_events', columns=['electrode', 'time_of_event'])
ev = ev.reset_index(drop=True)
st = pd.read_hdf(HDF, 'fs437_wholelife_stimulations',
                 columns=['electrode', 'time_of_stim', 'a1', 'a2', 'd1', 'd2',
                          'nb_stim_pulse', 'pulse_train_period', 'stim_shape'])
st = st.reset_index(drop=True)
ev_ns = ev['time_of_event'].to_numpy(dtype=np.int64)          # int64 ns
ev_el = ev['electrode'].to_numpy(dtype=np.int64)
st_ns = pd.to_datetime(st['time_of_stim']).to_numpy().astype('datetime64[ns]').astype(np.int64)
st_el = st['electrode'].to_numpy(dtype=np.int64)
st_day = pd.to_datetime(st['time_of_stim']).dt.strftime('%m-%d').to_numpy()
st['pname'] = ('a=' + st['a1'].astype(int).astype(str)
               + ',nb=' + st['nb_stim_pulse'].astype(int).astype(str)
               + ',per=' + st['pulse_train_period'].astype(int).astype(str))
P1 = 'a=3,nb=2,per=2000'    # June-9 dense day (360,765)
P2 = 'a=1,nb=2,per=10000'   # 06-08 (7,582) + 06-09 (95,876)
P3 = 'a=5,nb=2,per=10000'   # 06-07 (33,579) + 06-08 (35,965)
log(f"events={len(ev):,} stims={len(st):,}")

ev_by_el = {e: np.sort(ev_ns[ev_el == e]) for e in range(32)}
log("per-electrode event arrays built")

def counts_in_window(stim_times_ns, electrode, lo_ns, hi_ns):
    """Vectorized: for each stim time, #events on `electrode` in [t+lo, t+hi]."""
    E = ev_by_el[electrode]
    return (np.searchsorted(E, stim_times_ns + hi_ns, side='right')
            - np.searchsorted(E, stim_times_ns + lo_ns, side='left'))

# analysis stims restricted to the recording span (pre-recording stims have no events)
REC0, REC1 = ev_ns.min(), ev_ns.max()
in_rec = (st_ns >= REC0) & (st_ns <= REC1)

POST = (0, 50 * MS)            # [0,50] ms
BASE = (-500 * MS, -50 * MS)   # [-500,-50] ms

# ================================================================ Task A1
log("=== Task A1: evoked ratios + stim-shuffled nulls ===")

def evoked_ratio_idx(idx, times_ns):
    """mean post rate / mean baseline rate over stim subset (same electrode)."""
    idx = np.asarray(idx)
    post, base = [], []
    for e in range(32):
        m = idx[st_el[idx] == e]
        if len(m) == 0:
            continue
        t = times_ns[m]
        post.append(counts_in_window(t, e, *POST))
        base.append(counts_in_window(t, e, *BASE))
    pc = np.concatenate(post) / 0.05
    bc = np.concatenate(base) / 0.45
    return float(pc.mean() / bc.mean()), len(pc)

# (day, electrode) groups over in-recording stims: fixed across null reps,
# so precompute once. The null permutes ISIs within each group.
_all_idx = np.flatnonzero(in_rec)
_day_sub = st_day[_all_idx]
_el_sub = st_el[_all_idx]
_GROUPS = []
for _d in np.unique(_day_sub):
    _dm = (_day_sub == _d)
    for _e in range(32):
        _m = _all_idx[_dm & (_el_sub == _e)]
        if len(_m):
            _GROUPS.append(_m)
log(f"{len(_GROUPS)} (day,electrode) groups precomputed")

def shuffled_times_all(rep):
    """Permute ISIs within each (day, electrode) group over ALL in-recording
    stims. Preserves the exact per-electrode per-day ISI multiset; breaks
    stim<->event alignment. Protocol labels stay attached (only times move)."""
    r = np.random.default_rng(10_000 + rep)
    new_t = np.empty_like(st_ns)
    for m in _GROUPS:
        t = np.sort(st_ns[m])
        isi = np.diff(t)
        r.shuffle(isi)
        nt = np.empty_like(t)
        nt[0] = t[0]
        if len(t) > 1:
            nt[1:] = t[0] + np.cumsum(isi)
        # map shuffled times (built in sorted order) back to original order
        new_t[m] = nt[np.searchsorted(t, st_ns[m])]
    return new_t

a1_rows = []
proto_idx = {p: np.flatnonzero((st['pname'] == p).to_numpy() & in_rec) for p in (P1, P2, P3)}
obs_ratios = {p: evoked_ratio_idx(proto_idx[p], st_ns) for p in (P1, P2, P3)}
null_ratios = {p: [] for p in (P1, P2, P3)}
for rep in range(NREP):
    nt = shuffled_times_all(rep)
    for p in (P1, P2, P3):
        r, _ = evoked_ratio_idx(proto_idx[p], nt)
        null_ratios[p].append(r)
    if rep % 5 == 0:
        log(f"  null rep {rep}/{NREP} done")
for pname, plabel in [(P1, 'P1'), (P2, 'P2'), (P3, 'P3')]:
    obs, n = obs_ratios[pname]
    nulls = np.array(null_ratios[pname])
    p = (1 + np.sum(nulls >= obs)) / (1 + NREP)
    a1_rows.append(dict(measurement='evoked_ratio', variant='post[0,50]ms_vs_base[-500,-50]ms',
                        protocol=plabel, n=n, observed=round(obs, 4), obs_std='',
                        null_mean=round(float(nulls.mean()), 4),
                        null_p95=round(float(np.percentile(nulls, 95)), 4),
                        null_min=round(float(nulls.min()), 4), null_max=round(float(nulls.max()), 4),
                        p_value=round(float(p), 4), chance=1.0,
                        notes='honest null: ISIs permuted within (day,electrode); protocol labels kept'))
    log(f"  {plabel}: n={n:,} obs={obs:.3f} null_mean={nulls.mean():.3f} "
        f"null_p95={np.percentile(nulls,95):.3f} p={p:.4f}")

# ================================================================ shared helpers
def trial_counts(stim_idx, lo_ms, hi_ms, n_bins):
    """(n_trials, n_bins, 32) int16 event counts in [lo,hi] ms post-stim."""
    idx = np.asarray(stim_idx)
    n = len(idx)
    X = np.zeros((n, n_bins, 32), dtype=np.int16)
    edges = np.linspace(lo_ms * MS, hi_ms * MS, n_bins + 1).astype(np.int64)
    for e in range(32):
        rows = np.flatnonzero(st_el[idx] == e)
        if len(rows) == 0:
            continue
        t = st_ns[idx[rows]]
        E = ev_by_el[e]
        pos = np.searchsorted(E, t[:, None] + edges[None, :], side='left')
        X[rows[:, None], np.arange(n_bins)[None, :], e] = np.diff(pos, axis=1)
    return X

def baseline_count_matrix(stim_idx):
    """(n_trials, 32) baseline-window counts per electrode per stim."""
    idx = np.asarray(stim_idx)
    n = len(idx)
    X = np.zeros((n, 32), dtype=np.int16)
    for e in range(32):
        rows = np.flatnonzero(st_el[idx] == e)
        if len(rows) == 0:
            continue
        X[rows, e] = counts_in_window(st_ns[idx[rows]], e, *BASE)
    return X

def res_states_fixed(X, alpha, seed):
    n, T, C = X.shape
    W_res, W_in = AL.build_reservoir(seed, C)
    WrT, WiT = W_res.T, W_in.T
    s = np.zeros((n, W_res.shape[0]))
    acc = np.zeros_like(s)
    for ti in range(1, T):
        u = X[:, ti - 1, :]
        s = (1.0 - alpha) * s + alpha * np.tanh(s @ WrT + u @ WiT)
        acc += s
    return acc / T  # mean over T rows incl. initial zero row, as in audit_lib

def ridge_fit_mc(Xtr, ytr, wtr, n_classes, lam=AL.LAM):
    Y = np.eye(n_classes)[ytr]
    Ws = np.sqrt(wtr)[:, None]
    A = (Xtr * Ws).T @ (Xtr * Ws) + lam * np.eye(Xtr.shape[1])
    B = (Xtr * Ws).T @ (Y * Ws)
    return np.linalg.solve(A, B)

def quality_weights(X):
    """audit_lib.block_quality_weight per trial. X: (n, T, C)."""
    return np.array([AL.block_quality_weight(X[i].astype(float)) for i in range(len(X))])

def cv_lr(X, y, n_reps_null=0, seed0=0):
    """5-fold CV logistic regression; optional 20-rep label-shuffle null."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed0)
    accs = []
    for tr, te in skf.split(X, y):
        m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        accs.append(float((m.predict(X[te]) == y[te]).mean()))
    out = dict(acc=np.mean(accs), std=np.std(accs))
    if n_reps_null:
        nulls = []
        for rep in range(n_reps_null):
            yl = np.random.default_rng(9000 + rep).permutation(y)
            a = []
            for tr, te in skf.split(X, yl):
                m = LogisticRegression(max_iter=1000).fit(X[tr], yl[tr])
                a.append(float((m.predict(X[te]) == yl[te]).mean()))
            nulls.append(np.mean(a))
        nulls = np.array(nulls)
        out['null'] = nulls
    return out

def cv_reservoir(X, y, n_classes, n_reps_null=0):
    """Sweep canonical alphas x seeds; 5-fold CV ridge readout per combo."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    w = quality_weights(X)
    res = {}
    for alpha in AL.ALPHAS:
        accs, seeds_acc = [], []
        states_by_seed = {}
        for seed in AL.SEEDS:
            S = res_states_fixed(X.astype(np.float32), alpha, seed)
            states_by_seed[seed] = S
            fa = []
            for tr, te in skf.split(S, y):
                Wo = ridge_fit_mc(S[tr], y[tr], w[tr], n_classes)
                fa.append(float((np.argmax(S[te] @ Wo, axis=1) == y[te]).mean()))
            seeds_acc.append(np.mean(fa)); accs.append(np.mean(fa))
        d = dict(acc=np.mean(accs), std=np.std(accs), seeds_acc=seeds_acc)
        if n_reps_null:
            S = states_by_seed[AL.SEEDS[0]]  # fixed (alpha, seed=7) null per convention
            nulls = []
            for rep in range(n_reps_null):
                yl = np.random.default_rng(9000 + rep).permutation(y)
                a = []
                for tr, te in skf.split(S, yl):
                    Wo = ridge_fit_mc(S[tr], yl[tr], w[tr], n_classes)
                    a.append(float((np.argmax(S[te] @ Wo, axis=1) == yl[te]).mean()))
                nulls.append(np.mean(a))
            d['null'] = np.array(nulls)
        res[alpha] = d
    return res

def transfer_eval(Xtr, ytr, Xte, yte, kind, n_classes, n_reps_null=0):
    """Train on (Xtr,ytr), evaluate on (Xte,yte). kind in {'lr','reservoir'}.
    Returns dict; reservoir sweeps alphas x seeds."""
    if kind == 'lr':
        # LR works on the 32-count summary (same features as A2's decode_lr)
        Xtr2 = Xtr.sum(axis=1).astype(np.float64)
        Xte2 = Xte.sum(axis=1).astype(np.float64)
        m = LogisticRegression(max_iter=1000).fit(Xtr2, ytr)
        pred = m.predict(Xte2)
        acc = float((pred == yte).mean())
        out = dict(acc=acc, std=0.0, pred0=float((pred == 0).mean()))
        if n_reps_null:
            nulls, nullp0 = [], []
            for rep in range(n_reps_null):
                yl = np.random.default_rng(9000 + rep).permutation(ytr)
                mm = LogisticRegression(max_iter=1000).fit(Xtr2, yl)
                pr = mm.predict(Xte2)
                nulls.append(float((pr == yte).mean()))
                nullp0.append(float((pr == 0).mean()))
            out['null'] = np.array(nulls); out['null_pred0'] = np.array(nullp0)
        return {'lr': out}
    wtr = quality_weights(Xtr)
    res = {}
    for alpha in AL.ALPHAS:
        accs, p0s = [], []
        for seed in AL.SEEDS:
            Str = res_states_fixed(Xtr.astype(np.float32), alpha, seed)
            Ste = res_states_fixed(Xte.astype(np.float32), alpha, seed)
            Wo = ridge_fit_mc(Str, ytr, wtr, n_classes)
            pr = np.argmax(Ste @ Wo, axis=1)
            accs.append(float((pr == yte).mean())); p0s.append(float((pr == 0).mean()))
        # near-zero-state fraction on test (degenerate argmax(0)=class 0 check)
        zfrac = float((np.abs(Ste).max(axis=1) < 1e-6).mean())
        d = dict(acc=np.mean(accs), std=np.std(accs),
                 pred0=np.mean(p0s), test_zero_state_frac=zfrac)
        if n_reps_null:
            Str = res_states_fixed(Xtr.astype(np.float32), alpha, AL.SEEDS[0])
            Ste = res_states_fixed(Xte.astype(np.float32), alpha, AL.SEEDS[0])
            nulls, nullp0 = [], []
            for rep in range(n_reps_null):
                yl = np.random.default_rng(9000 + rep).permutation(ytr)
                Wo = ridge_fit_mc(Str, yl, wtr, n_classes)
                pr = np.argmax(Ste @ Wo, axis=1)
                nulls.append(float((pr == yte).mean())); nullp0.append(float((pr == 0).mean()))
            d['null'] = np.array(nulls); d['null_pred0'] = np.array(nullp0)
        res[alpha] = d
    return res

# ================================================================ Task A2
log("=== Task A2: protocol decoding P1/P2/P3 ===")
N_PER = 6000
sub_idx, sub_y = [], []
for k, p in enumerate([P1, P2, P3]):
    take = RNG.choice(proto_idx[p], size=min(N_PER, len(proto_idx[p])), replace=False)
    sub_idx.append(take); sub_y.append(np.full(len(take), k))
sub_idx = np.concatenate(sub_idx); sub_y = np.concatenate(sub_y)
perm = RNG.permutation(len(sub_idx)); sub_idx, sub_y = sub_idx[perm], sub_y[perm]
log(f"decoding subsample: {len(sub_idx):,} trials")

X_trials = trial_counts(sub_idx, 0, 50, 50)          # (n, 50, 32), bins 0..50 ms
X_lr_full = X_trials.sum(axis=1).astype(np.float64)  # 32 counts in [0,50] ms
X_lr_blank = X_trials[:, 10:, :].sum(axis=1).astype(np.float64)  # [10,50] ms
X_res_blank = X_trials.copy(); X_res_blank[:, :10, :] = 0       # zero [0,10) ms bins

a2_rows = []
def emit_a2(measurement, variant, X, y, kind, n_classes=3):
    if kind == 'lr':
        r = cv_lr(X, y, n_reps_null=NREP)
        nulls = r['null']
        p = (1 + np.sum(nulls >= r['acc'])) / (1 + NREP)
        a2_rows.append(dict(measurement=measurement, variant=variant, protocol='P1/P2/P3',
                            n=len(y), observed=round(r['acc'], 4), obs_std=round(r['std'], 4),
                            null_mean=round(float(nulls.mean()), 4),
                            null_p95=round(float(np.percentile(nulls, 95)), 4),
                            null_min=round(float(nulls.min()), 4), null_max=round(float(nulls.max()), 4),
                            p_value=round(float(p), 4), chance=round(1/3, 4),
                            notes='5-fold CV; 20-rep label-shuffle null'))
        log(f"  {measurement} {variant}: acc={r['acc']:.3f}+/-{r['std']:.3f} "
            f"null_mean={nulls.mean():.3f} p95={np.percentile(nulls,95):.3f} p={p:.4f}")
    else:
        r = cv_reservoir(X, y, n_classes, n_reps_null=NREP)
        for alpha in AL.ALPHAS:
            d = r[alpha]; nulls = d['null']
            p = (1 + np.sum(nulls >= d['acc'])) / (1 + NREP)
            a2_rows.append(dict(measurement=f"{measurement}_alpha{alpha}", variant=variant,
                                protocol='P1/P2/P3', n=len(y),
                                observed=round(d['acc'], 4), obs_std=round(d['std'], 4),
                                null_mean=round(float(nulls.mean()), 4),
                                null_p95=round(float(np.percentile(nulls, 95)), 4),
                                null_min=round(float(nulls.min()), 4),
                                null_max=round(float(nulls.max()), 4),
                                p_value=round(float(p), 4), chance=round(1/3, 4),
                                notes=f"5-fold CV; seeds {AL.SEEDS} accs="
                                      f"{[round(a,3) for a in d['seeds_acc']]}; 20-rep label-shuffle null at seed 7"))
            log(f"  {measurement} {variant} alpha={alpha}: acc={d['acc']:.3f}+/-{d['std']:.3f} "
                f"null_mean={nulls.mean():.3f} p={p:.4f}")

emit_a2('decode_lr', '[0,50]ms', X_lr_full, sub_y, 'lr')
emit_a2('decode_lr', '[10,50]ms_blanked', X_lr_blank, sub_y, 'lr')
emit_a2('decode_reservoir', '[0,50]ms', X_trials, sub_y, 'reservoir')
emit_a2('decode_reservoir', '[10,50]ms_blanked', X_res_blank, sub_y, 'reservoir')
del X_trials, X_res_blank

# ================================================================ Task B
log("=== Task B: cross-day transfer ===")
b_rows = []
def emit_transfer(measurement, train_period, test_period, classes, Xtr, ytr, Xte, yte,
                  per_class_note=''):
    n_classes = len(np.unique(np.concatenate([ytr, yte])))
    for kind in ['lr', 'reservoir']:
        r = transfer_eval(Xtr, ytr, Xte, yte, kind, n_classes, n_reps_null=NREP)
        items = [('lr', r['lr'])] if kind == 'lr' else [(f'reservoir_alpha{a}', r[a]) for a in AL.ALPHAS]
        for name, d in items:
            nulls = d['null']
            p = (1 + np.sum(nulls >= d['acc'])) / (1 + NREP)
            note = (per_class_note
                    + f"frac predicted class-0: obs={d.get('pred0', float('nan')):.3f}, "
                    f"null_mean={d.get('null_pred0', np.array([np.nan])).mean():.3f}; "
                    + (f"test near-zero-state frac={d['test_zero_state_frac']:.3f}; " if 'test_zero_state_frac' in d else '')
                    + '20-rep train-label-shuffle null')
            b_rows.append(dict(measurement=f"{measurement}_{name}", train_period=train_period,
                               test_period=test_period, classes=classes,
                               n_train=len(ytr), n_test=len(yte),
                               observed=round(d['acc'], 4), obs_std=round(d['std'], 4),
                               null_mean=round(float(nulls.mean()), 4),
                               null_p95=round(float(np.percentile(nulls, 95)), 4),
                               p_value=round(float(p), 4), chance=round(1/n_classes, 4),
                               notes=note))
            log(f"  {measurement}_{name}: {train_period}->{test_period} acc={d['acc']:.3f} "
                f"null_mean={nulls.mean():.3f} p={p:.4f}")

# --- B1 forward: train P2/P3 on June 7-8 -> test June 9 (P2 anchor class present) ---
idx_tr_b1 = np.concatenate([
    RNG.choice(np.flatnonzero((st['pname']==P2).to_numpy() & (st_day=='06-08') & in_rec), 5000, replace=False),
    RNG.choice(np.flatnonzero((st['pname']==P3).to_numpy() & np.isin(st_day,['06-07','06-08']) & in_rec), 5000, replace=False)])
y_tr_b1 = np.array([0]*5000 + [1]*5000)
idx_te_b1 = RNG.choice(np.flatnonzero((st['pname']==P2).to_numpy() & (st_day=='06-09') & in_rec), 10000, replace=False)
y_te_b1 = np.zeros(10000, dtype=int)  # all P2 -> observed = P2 recall
# in-distribution reference: held-out June-8 P2 (not in train)
pool_b1 = np.flatnonzero((st['pname']==P2).to_numpy() & (st_day=='06-08') & in_rec)
idx_ref_b1 = np.setdiff1d(pool_b1, idx_tr_b1[:5000]); y_ref_b1 = np.zeros(len(idx_ref_b1), dtype=int)
Xtr_b1 = trial_counts(idx_tr_b1, 0, 50, 50); Xte_b1 = trial_counts(idx_te_b1, 0, 50, 50)
Xref_b1 = trial_counts(idx_ref_b1, 0, 50, 50)
emit_transfer('transfer_forward', 'June7-8', 'June9', 'P2/P3', Xtr_b1, y_tr_b1, Xte_b1, y_te_b1,
              per_class_note='test set = June-9 P2 only: observed = P2 recall. ')
emit_transfer('transfer_forward_indist', 'June7-8', 'June8-heldout', 'P2/P3', Xtr_b1, y_tr_b1,
              Xref_b1, y_ref_b1, per_class_note='in-distribution reference: held-out June-8 P2 recall. ')

# --- B2 reverse: train P1/P2 on June 9 -> test June 7-8 (P2 anchor) ---
idx_tr_b2 = np.concatenate([
    RNG.choice(np.flatnonzero((st['pname']==P1).to_numpy() & (st_day=='06-09') & in_rec), 10000, replace=False),
    RNG.choice(np.flatnonzero((st['pname']==P2).to_numpy() & (st_day=='06-09') & in_rec), 10000, replace=False)])
y_tr_b2 = np.array([0]*10000 + [1]*10000)
pool_b2 = np.flatnonzero((st['pname']==P2).to_numpy() & np.isin(st_day,['06-07','06-08']) & in_rec)
idx_te_b2 = RNG.choice(pool_b2, 5000, replace=False); y_te_b2 = np.ones(5000, dtype=int)
pool_b2b = np.flatnonzero((st['pname']==P2).to_numpy() & (st_day=='06-09') & in_rec)
idx_ref_b2 = RNG.choice(np.setdiff1d(pool_b2b, idx_tr_b2[10000:]), 10000, replace=False)
y_ref_b2 = np.ones(10000, dtype=int)
Xtr_b2 = trial_counts(idx_tr_b2, 0, 50, 50); Xte_b2 = trial_counts(idx_te_b2, 0, 50, 50)
Xref_b2 = trial_counts(idx_ref_b2, 0, 50, 50)
emit_transfer('transfer_reverse', 'June9', 'June7-8', 'P1/P2', Xtr_b2, y_tr_b2, Xte_b2, y_te_b2,
              per_class_note='test set = June7-8 P2 only: observed = P2 recall. ')
emit_transfer('transfer_reverse_indist', 'June9', 'June9-heldout', 'P1/P2', Xtr_b2, y_tr_b2,
              Xref_b2, y_ref_b2, per_class_note='in-distribution reference: held-out June-9 P2 recall. ')

# --- B2b diagnostics: is the reverse-LR "perfect transfer" a degenerate always-P2 predictor?
# and does the forward decoder recognize P3 in-distribution?
log("=== Task B diagnostics ===")
Xtr_b2c = Xtr_b2.sum(axis=1).astype(np.float64)
m_b2 = LogisticRegression(max_iter=1000).fit(Xtr_b2c, y_tr_b2)
# held-out June-9 P1 (class 0) recall
pool_p1 = np.flatnonzero((st['pname']==P1).to_numpy() & (st_day=='06-09') & in_rec)
idx_p1h = RNG.choice(np.setdiff1d(pool_p1, idx_tr_b2[:10000]), 5000, replace=False)
X_p1h = trial_counts(idx_p1h, 0, 50, 50).sum(axis=1).astype(np.float64)
p1_rec = float((m_b2.predict(X_p1h) == 0).mean())
# 5-fold CV on the B2 train set (learnability)
r_cv = cv_lr(Xtr_b2c, y_tr_b2)
b_rows.append(dict(measurement='transfer_reverse_diag_lr', train_period='June9',
                   test_period='June9-heldout', classes='P1/P2',
                   n_train=len(y_tr_b2), n_test=len(idx_p1h),
                   observed=round(p1_rec, 4), obs_std=0.0,
                   null_mean='', null_p95='', p_value='',
                   chance=0.5,
                   notes=f'P1 recall on held-out June-9 P1 (degeneracy check for the 1.000 P2 recall); '
                         f'train 5-fold CV acc={r_cv["acc"]:.3f}+/-{r_cv["std"]:.3f}'))
log(f"  reverse diag: P1 recall={p1_rec:.3f}, train CV={r_cv['acc']:.3f}")
# forward decoder: P3 recall in-distribution
Xtr_b1c = Xtr_b1.sum(axis=1).astype(np.float64)
m_b1 = LogisticRegression(max_iter=1000).fit(Xtr_b1c, y_tr_b1)
pool_p3 = np.flatnonzero((st['pname']==P3).to_numpy() & np.isin(st_day,['06-07','06-08']) & in_rec)
idx_p3h = RNG.choice(np.setdiff1d(pool_p3, idx_tr_b1[5000:]), 5000, replace=False)
X_p3h = trial_counts(idx_p3h, 0, 50, 50).sum(axis=1).astype(np.float64)
p3_rec = float((m_b1.predict(X_p3h) == 1).mean())
r_cv1 = cv_lr(Xtr_b1c, y_tr_b1)
b_rows.append(dict(measurement='transfer_forward_diag_lr', train_period='June7-8',
                   test_period='June7-8-heldout', classes='P2/P3',
                   n_train=len(y_tr_b1), n_test=len(idx_p3h),
                   observed=round(p3_rec, 4), obs_std=0.0,
                   null_mean='', null_p95='', p_value='',
                   chance=0.5,
                   notes=f'P3 recall on held-out June7-8 P3; train 5-fold CV acc={r_cv1["acc"]:.3f}+/-{r_cv1["std"]:.3f}'))
log(f"  forward diag: P3 recall={p3_rec:.3f}, train CV={r_cv1['acc']:.3f}")

# --- B3 day-ID confound check: decode day from baseline rate vectors ---
idx_b3, y_b3 = [], []
for k, d in enumerate(['06-07', '06-08', '06-09']):
    pool = np.flatnonzero((st_day == d) & in_rec)
    take = RNG.choice(pool, 6000, replace=False)
    idx_b3.append(take); y_b3.append(np.full(len(take), k))
idx_b3 = np.concatenate(idx_b3); y_b3 = np.concatenate(y_b3)
X_b3 = baseline_count_matrix(idx_b3).astype(np.float64)
r = cv_lr(X_b3, y_b3, n_reps_null=NREP)
nulls = r['null']; p = (1 + np.sum(nulls >= r['acc'])) / (1 + NREP)
b_rows.append(dict(measurement='dayid_baseline_lr', train_period='06-07/06-08/06-09 CV',
                   test_period='06-07/06-08/06-09 CV', classes='day(06-07/06-08/06-09)',
                   n_train=len(y_b3), n_test=len(y_b3),
                   observed=round(r['acc'], 4), obs_std=round(r['std'], 4),
                   null_mean=round(float(nulls.mean()), 4),
                   null_p95=round(float(np.percentile(nulls, 95)), 4),
                   p_value=round(float(p), 4), chance=round(1/3, 4),
                   notes='5-fold CV on baseline [-500,-50]ms 32-count vectors; 20-rep label-shuffle null'))
log(f"  dayid_baseline_lr: acc={r['acc']:.3f}+/-{r['std']:.3f} null_mean={nulls.mean():.3f} p={p:.4f}")
# reservoir variant of day-ID (needs trial series, not count matrix)
X_b3t = trial_counts(idx_b3, 0, 50, 50)
rr = cv_reservoir(X_b3t, y_b3, 3, n_reps_null=NREP)
for alpha in AL.ALPHAS:
    d = rr[alpha]; nulls = d['null']; p = (1 + np.sum(nulls >= d['acc'])) / (1 + NREP)
    b_rows.append(dict(measurement=f'dayid_baseline_reservoir_alpha{alpha}',
                       train_period='06-07/06-08/06-09 CV', test_period='06-07/06-08/06-09 CV',
                       classes='day(06-07/06-08/06-09)', n_train=len(y_b3), n_test=len(y_b3),
                       observed=round(d['acc'], 4), obs_std=round(d['std'], 4),
                       null_mean=round(float(nulls.mean()), 4),
                       null_p95=round(float(np.percentile(nulls, 95)), 4),
                       p_value=round(float(p), 4), chance=round(1/3, 4),
                       notes='5-fold CV on [0,50]ms trial series; 20-rep label-shuffle null at seed 7'))
    log(f"  dayid_baseline_reservoir alpha={alpha}: acc={d['acc']:.3f} null_mean={nulls.mean():.3f} p={p:.4f}")

# ================================================================ Task C
log("=== Task C: dead-culture negative control ===")
c_rows = []
j11 = np.flatnonzero(st_day == '06-11')
c_rows.append(dict(measurement='n_stims_june11', value=len(j11),
                   notes='stims delivered 2025-06-11 07:00-08:00, after last event 06:38:02'))
ev_j11_span = int(((ev_ns >= st_ns[j11].min()) & (ev_ns <= st_ns[j11].max())).sum())
c_rows.append(dict(measurement='n_events_during_june11_stim_span', value=ev_j11_span,
                   notes='events overlapping the 07:00-08:00 stim window'))
post_c, base_c = [], []
for e in range(32):
    m = j11[st_el[j11] == e]
    if len(m) == 0:
        continue
    t = st_ns[m]
    post_c.append(counts_in_window(t, e, *POST))
    base_c.append(counts_in_window(t, e, *BASE))
post_c = np.concatenate(post_c); base_c = np.concatenate(base_c)
n_post_ev = int(post_c.sum()); n_base_ev = int(base_c.sum())
ratio = float((post_c.mean()/0.05) / (base_c.mean()/0.45)) if base_c.sum() > 0 else float('nan')
c_rows.append(dict(measurement='dead_evoked_post_events_total', value=n_post_ev,
                   notes='total events in [0,50]ms post-stim, same electrode, over 1,050 stims'))
c_rows.append(dict(measurement='dead_evoked_baseline_events_total', value=n_base_ev,
                   notes='total events in [-500,-50]ms pre-stim, same electrode'))
c_rows.append(dict(measurement='dead_evoked_ratio', value=ratio,
                   notes='mean post rate / mean baseline rate; NaN = 0/0, no detectable response'))
c_rows.append(dict(measurement='dead_stims_with_ge1_post_event', value=int((post_c > 0).sum()),
                   notes='expect 0'))
log(f"  June-11: {len(j11)} stims, post events={n_post_ev}, baseline events={n_base_ev}, ratio={ratio}")

# live-trained protocol decoder applied to dead-culture stims
X_live = X_lr_full  # 18k x 32, [0,50]ms counts, P1/P2/P3 balanced
live_model = LogisticRegression(max_iter=1000).fit(X_live, sub_y)
X_dead = trial_counts(j11, 0, 50, 50).sum(axis=1).astype(np.float64)
proba_dead = live_model.predict_proba(X_dead)
pred_dead = proba_dead.argmax(axis=1)
zero_frac = float((X_dead.sum(axis=1) == 0).mean())
top_prob_dead = float(proba_dead.max(axis=1).mean())
# live reference: 5-fold CV held-out top-probability
skf = StratifiedKFold(5, shuffle=True, random_state=0)
tp_live = []
for tr, te in skf.split(X_live, sub_y):
    m = LogisticRegression(max_iter=1000).fit(X_live[tr], sub_y[tr])
    tp_live.append(m.predict_proba(X_live[te]).max(axis=1))
top_prob_live = float(np.concatenate(tp_live).mean())
c_rows.append(dict(measurement='dead_zero_feature_fraction', value=round(zero_frac, 4),
                   notes='fraction of June-11 stims with all-zero 32-count [0,50]ms features'))
for k, pl in enumerate(['P1', 'P2', 'P3']):
    c_rows.append(dict(measurement=f'dead_predicted_class_fraction_{pl}',
                       value=round(float((pred_dead == k).mean()), 4),
                       notes='live-trained P1/P2/P3 LR applied to dead-culture stims'))
c_rows.append(dict(measurement='dead_mean_top_class_probability', value=round(top_prob_dead, 4),
                   notes='chance-level confidence = 0.333'))
c_rows.append(dict(measurement='live_mean_top_class_probability', value=round(top_prob_live, 4),
                   notes='same model, live held-out stims (5-fold CV)'))
# P2-matching June-11 stims (492 use exactly P2's parameters): P2 recall vs chance
j11_p2 = j11[st['pname'].to_numpy()[j11] == P2]
if len(j11_p2):
    X_dead_p2 = trial_counts(j11_p2, 0, 50, 50).sum(axis=1).astype(np.float64)
    rec = float((live_model.predict(X_dead_p2) == 1).mean())
    c_rows.append(dict(measurement='dead_P2protocol_stim_P2_recall', value=round(rec, 4),
                       notes=f'{len(j11_p2)} June-11 stims use exactly P2 parameters; '
                             'fraction classified P2 by live-trained decoder (chance=0.333)'))
log(f"  dead decoder: zero_frac={zero_frac:.3f} top_prob_dead={top_prob_dead:.3f} "
    f"top_prob_live={top_prob_live:.3f}")

# ================================================================ write CSVs
def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

write_csv(os.path.join(RES, 'evoked_fs437.csv'), a1_rows + a2_rows)
write_csv(os.path.join(RES, 'transfer_fs437.csv'), b_rows)
write_csv(os.path.join(RES, 'dead_culture_fs437.csv'), c_rows)
log(f"CSVs written to {RES}/")

with open(os.path.join(RES, 'run_log.txt'), 'a') as f:
    f.write(f"\n[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC] Workstream 3 (evoked_fs437.py): "
            f"A1 evoked ratios P1/P2/P3 + 20-rep stim-shuffled nulls; "
            f"A2 P1/P2/P3 decoding (LR + canonical reservoir alphas {AL.ALPHAS}, seeds {AL.SEEDS}, "
            f"5-fold CV, [0,50]ms and [10,50]ms-blanked, 20-rep label-shuffle nulls); "
            f"B cross-day transfer P2/P3 June7-8->June9, P1/P2 June9->June7-8, day-ID baseline confound check; "
            f"C dead-culture (June 11, {len(j11)} stims) negative control. "
            f"Outputs: evoked_fs437.csv ({len(a1_rows)+len(a2_rows)} rows), "
            f"transfer_fs437.csv ({len(b_rows)} rows), dead_culture_fs437.csv ({len(c_rows)} rows).\n")
log("run log appended")
log(f"DONE in {time.time()-t0:.1f}s")
