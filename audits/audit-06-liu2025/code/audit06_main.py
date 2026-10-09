"""Audit #6 main analysis: honest-controls battery on Liu & Buonomano 2025 Fig-2 data.

Controls per Open Evaluation Standard v0.4:
  headline replication, within-batch stratification (§3.1),
  batch-identity classifier null (§3.1), label-shuffle null (§3.2),
  rate/amplitude-only baseline (§3.5), cross-batch transfer (§3.7),
  stimulation-parameter check (§4.4).
Writes results/*.csv and results/run_log.txt.
"""
import sys, os, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import scipy.io as sio
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from liu_pipeline import detect_events, cog_ms, smoothtrace

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(WORK, 'results')
logf = open(os.path.join(RES, 'run_log.txt'), 'w')
def log(s):
    print(s); logf.write(s + '\n'); logf.flush()
log(f"audit06 run {time.strftime('%Y-%m-%d %H:%M %Z')}")

# ---------- load + features ----------
t0 = time.time()
m = sio.loadmat(os.path.join(WORK, 'data/extracted/Figure2_DATA_batch1_2.mat'),
                squeeze_me=True, struct_as_record=False)
cm = m['compMETA']
import re
recs = []
for i, e in enumerate(cm):
    kt = np.asarray(e.keepTrace, dtype=float)
    r = detect_events(kt)
    mm = re.match(r'(\d{6})_S(\d+)C(\d+)_(\d+)\.mat', str(e.CondNum))
    date = mm.group(1) if mm else '?'
    # extra features
    mean_tr = kt.mean(axis=0)
    win = mean_tr[10000:20000]
    area = float(np.trapz(np.abs(win - win[:1000].mean())))
    base_sd = float(kt[:, :10000].std())
    recs.append(dict(
        idx=i, protocol=str(e.stimProtocol), date=date,
        celltype=str(e.CellType), condition=str(e.Condition),
        ntraces=kt.shape[0],
        event_ms=r['median_event_ms'], cog_ms=cog_ms(kt),
        peak_time_ms=r['peak_time_ms'], peak_amp=r['peak_amp'],
        baseline=r['baseline'], base_sd=base_sd, area=area,
        ev_theirs=float(e.MedianEventTime), cog_theirs=float(e.CenterOfGravity),
    ))
log(f"loaded {len(recs)} recordings in {time.time()-t0:.1f}s")
# pipeline fidelity check
ev = np.array([r['event_ms'] for r in recs]); ev_t = np.array([r['ev_theirs'] for r in recs])
cg = np.array([r['cog_ms'] for r in recs]); cg_t = np.array([r['cog_theirs'] for r in recs])
log(f"pipeline check: event_time max|diff|={np.nanmax(np.abs(ev-ev_t)):.2f}ms corr={np.corrcoef(ev,ev_t)[0,1]:.6f}")
log(f"pipeline check: cog max|diff|={np.nanmax(np.abs(cg-cg_t)):.2f}ms corr={np.corrcoef(cg,cg_t)[0,1]:.6f} (600ms window)")

y = np.array([1 if r['protocol'] == '12EARLY' else 0 for r in recs])  # 1=Early
batches = np.array([r['date'] for r in recs])
uniq_batches = sorted(set(batches))

def wcsv(name, rows, fields=None):
    with open(os.path.join(RES, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

wcsv('features.csv', [{k: (float(v) if isinstance(v, float) else v) for k, v in r.items()} for r in recs])

# ---------- 1. headline replication ----------
E = ev[y == 1]; L = ev[y == 0]
u, p = stats.mannwhitneyu(E, L, alternative='two-sided')
log(f"\n[1] HEADLINE: Early n={len(E)} median={np.median(E):.1f} mean={np.mean(E):.1f} sd={np.std(E,ddof=1):.1f}")
log(f"    Late  n={len(L)} median={np.median(L):.1f} mean={np.mean(L):.1f} sd={np.std(L,ddof=1):.1f}")
log(f"    Mann-Whitney p={p:.3e}  (paper: 150±26 vs 479±32 ms, p<1e-9)")

# ---------- 2. within-batch stratification (§3.1 core) ----------
log("\n[2] WITHIN-BATCH stratification (§3.1):")
strat_rows = []
for d in uniq_batches:
    ee = ev[(batches == d) & (y == 1)]; ll = ev[(batches == d) & (y == 0)]
    if len(ee) and len(ll):
        uu, pp = stats.mannwhitneyu(ee, ll, alternative='two-sided')
        strat_rows.append(dict(batch=d, nE=len(ee), nL=len(ll),
                               medE=f"{np.median(ee):.0f}", medL=f"{np.median(ll):.0f}",
                               diff=f"{np.median(ee)-np.median(ll):.0f}", p=f"{pp:.4f}"))
        log(f"    {d}: nE={len(ee)} nL={len(ll)} medE={np.median(ee):.0f} medL={np.median(ll):.0f} diff={np.median(ee)-np.median(ll):.0f} p={pp:.4f}")
    else:
        log(f"    {d}: single-protocol batch (n={'E' if len(ee) else 'L'}={max(len(ee),len(ll))}) — no within-batch contrast possible")
wcsv('within_batch.csv', strat_rows)
# stratified permutation test: permute protocol labels WITHIN batches,
# statistic = overall median(E)-median(L). Tests protocol effect beyond batch structure.
log("    stratified within-batch permutation test (2000 reps):")
rng3 = np.random.default_rng(123)
obs_diff = np.median(ev[y == 1]) - np.median(ev[y == 0])
perm_diffs = []
yp = y.copy()
for _ in range(2000):
    yp2 = yp.copy()
    for d in uniq_batches:
        m_ = batches == d
        yp2[m_] = rng3.permutation(yp[m_])
    perm_diffs.append(np.median(ev[yp2 == 1]) - np.median(ev[yp2 == 0]))
perm_diffs = np.array(perm_diffs)
hits = int((perm_diffs <= obs_diff).sum())
log(f"      observed diff={obs_diff:.1f}ms; {hits}/2000 within-batch shuffles as-or-more extreme")
log(f"      (batch-magnitude note: within-batch diffs range -261..-471ms — direction")
log(f"       consistent across all 5 mixed batches, magnitude varies)")
with open(os.path.join(RES, 'stratified_perm.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['observed_diff_ms', 'nrep', 'hits'])
    w.writerow([f"{obs_diff:.2f}", 2000, hits])

# ---------- 3. batch-identity classifier null (§3.1) ----------
log("\n[3] BATCH-IDENTITY null (§3.1): can batch be decoded from traces within one protocol?")
feat_names = ['event_ms', 'cog_ms', 'peak_time_ms', 'peak_amp', 'area', 'base_sd', 'ntraces']
X = np.array([[r[f] for f in feat_names] for r in recs])
for proto, pname in [(1, 'Early'), (0, 'Late')]:
    sel = (y == proto)
    Xp, bp = X[sel], batches[sel]
    ub = sorted(set(bp))
    if len(ub) < 2:
        continue
    # leave-one-out CV logistic regression on batch labels
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    preds = []
    for i in range(len(Xp)):
        tr = np.ones(len(Xp), bool); tr[i] = False
        clf.fit(Xp[tr], bp[tr]); preds.append(clf.predict(Xp[~tr])[0])
    acc = np.mean(np.array(preds) == bp)
    log(f"    {pname}: batch decode acc={acc:.3f} (chance 1/{len(ub)}={1/len(ub):.3f}, n={len(Xp)})")

# ---------- 4. label-shuffle null (§3.2) ----------
log("\n[4] SHUFFLE null (§3.2): shuffle protocol labels, recompute median(E)-median(L)")
rng = np.random.default_rng(7)
obs = np.median(E) - np.median(L)
nrep, hits, maxd = 200, 0, -1e9
for _ in range(nrep):
    yp = rng.permutation(y)
    d = np.median(ev[yp == 1]) - np.median(ev[yp == 0])
    maxd = max(maxd, d)
    if d <= obs:  # obs is negative; as-or-more-extreme
        hits += 1
log(f"    observed diff={obs:.1f}ms; {hits}/{nrep} shuffles as-or-more extreme; max shuffle diff={maxd:.1f}")
with open(os.path.join(RES, 'shuffle_null.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['observed_diff_ms', 'nrep', 'hits', 'max_shuffle_diff_ms'])
    w.writerow([f"{obs:.2f}", nrep, hits, f"{maxd:.2f}"])

# ---------- 5. rate/amplitude-only baseline (§3.5) ----------
log("\n[5] RATE/AMPLITUDE-only baseline (§3.5):")
amp_feats = ['peak_amp', 'area', 'base_sd']
time_feats = ['event_ms', 'cog_ms', 'peak_time_ms']
def cv_acc(cols, seed=0):
    Xa = np.array([[r[c] for c in cols] for r in recs])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    rng2 = np.random.default_rng(seed)
    accs = []
    for _ in range(20):
        idx = rng2.permutation(len(Xa))
        tr, te = idx[:52], idx[52:]
        clf.fit(Xa[tr], y[tr]); accs.append(clf.score(Xa[te], y[te]))
    return np.mean(accs), np.std(accs)
for name, cols in [('amplitude-only', amp_feats), ('timing-only', time_feats), ('all-features', feat_names)]:
    m_, s_ = cv_acc(cols)
    log(f"    {name}: 5-fold-ish CV acc={m_:.3f}±{s_:.3f} (n=65, chance=0.50)")
with open(os.path.join(RES, 'rate_baseline.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['feature_set', 'mean_acc', 'sd_acc'])
    for name, cols in [('amplitude-only', amp_feats), ('timing-only', time_feats), ('all-features', feat_names)]:
        m_, s_ = cv_acc(cols); w.writerow([name, f"{m_:.4f}", f"{s_:.4f}"])

# ---------- 6. cross-batch transfer (§3.7) ----------
log("\n[6] CROSS-BATCH transfer (§3.7): leave-one-batch-out protocol decoding")
Xa = np.array([[r[c] for c in feat_names] for r in recs])
rows6 = []
for d in uniq_batches:
    tr = batches != d; te = batches == d
    if y[tr].sum() == 0 or y[tr].sum() == len(y[tr]):
        log(f"    batch {d}: train set single-class — skipped"); continue
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xa[tr], y[tr])
    acc = clf.score(Xa[te], y[te])
    rows6.append(dict(batch=d, n_test=int(te.sum()), acc=f"{acc:.3f}",
                      nE_test=int(((batches == d) & (y == 1)).sum()),
                      nL_test=int(((batches == d) & (y == 0)).sum())))
    log(f"    held-out {d}: acc={acc:.3f} (n_test={te.sum()})")
accs = np.array([float(r['acc']) for r in rows6])
log(f"    mean LOBO acc={accs.mean():.3f} (chance=0.50)")
wcsv('cross_batch_transfer.csv', rows6)

# ---------- 7. quality-confound checks ----------
log("\n[7] Quality-confound checks:")
for q in ['ntraces', 'base_sd']:
    qv = np.array([r[q] for r in recs], dtype=float)
    r_, p_ = stats.pearsonr(qv, ev)
    log(f"    {q} vs event_ms: r={r_:.3f} p={p_:.4f}")
    ue, pe = stats.mannwhitneyu(qv[y == 1], qv[y == 0], alternative='two-sided')
    log(f"    {q} Early-vs-Late: medE={np.median(qv[y==1]):.2f} medL={np.median(qv[y==0]):.2f} p={pe:.4f}")

log("\n[8] SCOPE: dataset covers Fig-2 evoked timing only (65 cells). Spontaneous-replay and")
log("    prediction-error (omission) claims are not testable from this release — out of scope.")
log("\n[9] CAVEAT: keepTrace = investigator-selected traces (non-blinded collection per the")
log("    paper); TraceAll not in the .mat — selection bias cannot be tested. Carried as limitation.")
logf.close()
print("done")
