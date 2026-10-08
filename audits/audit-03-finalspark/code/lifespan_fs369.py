"""Workstream 2 (Audit #3): FS369 lifespan session-identity decoding (the Stage C
mirror) + medium-change natural experiment.

CONFIDENTIAL data: events table read in chunks; ONLY aggregates are written out
(CSVs in results/ + run log). No raw spike data leaves the audit directory.

Task A: 5-min blocks -> 32-dim firing-rate vectors -> logistic regression,
  5-fold stratified CV, decoding (a) early/mid/late epoch (chance 1/3),
  (b) calendar-day ID (chance 1/K). 20-rep label-shuffle nulls. Ablation:
  epoch decoding on the 8 persistent electrodes vs the 8 collapsing ones.

Task B: Proto A stims only. Per-stim evoked ratio = rate[0,50]ms /
  rate[-500,-50]ms. POST (02-28, after medium change) vs PRE (02-20+02-21);
  honest control 02-20 vs 02-21 (no medium change). Plus baseline-rate contrast.

Framing: Task A is a confound demonstration. Continuous 21-day recording with a
spatially structured die-off should decode "session" near-perfectly -- exactly
the Stage C pattern (the pipeline separates recording sessions, not the
experimental variable). Near-perfect accuracy here is the EXPECTED, honest
result, not a discovery.
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import mannwhitneyu

BASE = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark"
F = BASE + "/data/fs369/fs369_package.hdf5"
MS = 1_000_000
DAY = 86_400_000_000_000
BLOCK_NS = 300_000_000_000  # 5-minute blocks

t0 = time.time()
print("== lifespan_fs369.py start ==", flush=True)

# ---------------------------------------------------------------- Proto A stims
st = pd.read_hdf(F, key="fs369_wholelife_stimulations")
spec = dict(a1=1.5, a2=1.5, d1=300.0, d2=300.0, nb_stim_pulse=2.0,
            pulse_train_period=10000.0, stim_polarity=1.0, stim_shape=0.0)
mask = np.ones(len(st), dtype=bool)
for k, v in spec.items():
    mask &= (st[k].values == v)
proto = st.loc[mask].copy()
sns = proto["time_of_stim"].values.astype("datetime64[ns]").astype("int64")
sns = np.sort(sns)
print(f"Proto A stims (full spec incl. period=10000): {len(sns)}", flush=True)

def day_of(s):
    return int(pd.Timestamp(s, tz="UTC").value // DAY)

D020, D021, D028 = day_of("2025-02-20"), day_of("2025-02-21"), day_of("2025-02-28")
sdays = sns // DAY
grp = np.full(len(sns), "", dtype=object)
grp[sdays == D028] = "post"
grp[sdays == D020] = "pre20"
grp[sdays == D021] = "pre21"
keep_s = grp != ""
sns_k, grp_k = sns[keep_s], grp[keep_s]
print("Stim groups (Proto A): " +
      ", ".join(f"{g}={int((grp_k == g).sum())}" for g in ["pre20", "pre21", "post"]),
      flush=True)
print("Proto A stims on other days (excluded):", int(len(sns) - len(sns_k)), flush=True)

# ------------------------------------------------- single chunked events pass
store = pd.HDFStore(F, mode="r")
nrows = int(store.get_storer("fs369_wholelife_events").nrows)
print("events nrows:", nrows, flush=True)

T0 = pd.Timestamp("2025-02-19 00:00", tz="UTC").value  # fixed origin (UTC)
T_END = pd.Timestamp("2025-03-13 00:00", tz="UTC").value
NB = int(np.ceil((T_END - T0) / BLOCK_NS))
counts = np.zeros((NB, 32), dtype=np.int64)          # Task A block x electrode
pre_c = np.zeros(len(sns_k), dtype=np.int64)         # Task B per-stim window counts
post_c = np.zeros(len(sns_k), dtype=np.int64)

CH = 4_000_000
nch = (nrows + CH - 1) // CH
for ci in range(nch):
    s0, s1 = ci * CH, min((ci + 1) * CH, nrows)
    df = store.select("fs369_wholelife_events", start=s0, stop=s1,
                      columns=["electrode", "time_of_event"])
    t = df["time_of_event"].values.astype("datetime64[ns]").astype("int64")
    e = df["electrode"].values.astype(np.int64) - 64
    bi = (t - T0) // BLOCK_NS
    np.add.at(counts, (bi, e), 1)                     # Task A accumulation
    # Task B: stims whose [-500ms, +50ms] window overlaps this chunk
    cmin, cmax = t[0], t[-1]
    lo = np.searchsorted(sns_k, cmin - 50 * MS)
    hi = np.searchsorted(sns_k, cmax + 500 * MS)
    if hi > lo:
        cs = sns_k[lo:hi]
        pre_c[lo:hi] += (np.searchsorted(t, cs - 50 * MS)
                         - np.searchsorted(t, cs - 500 * MS))
        post_c[lo:hi] += (np.searchsorted(t, cs + 50 * MS)
                          - np.searchsorted(t, cs))
    del df, t, e, bi
    if (ci + 1) % 6 == 0 or ci == nch - 1:
        print(f"  chunk {ci + 1}/{nch} ({time.time() - t0:.0f}s)", flush=True)
store.close()

# ================================================================= Task A
tot = counts.sum(axis=1)
keep_b = tot > 0
X = counts[keep_b].astype(np.float64) / 300.0        # Hz per electrode
bidx = np.nonzero(keep_b)[0]
tstart = T0 + bidx * BLOCK_NS
days_b = (tstart // DAY).astype(np.int64)
D0226, D0305 = day_of("2025-02-26"), day_of("2025-03-05")
epoch = np.where(days_b < D0226, 0, np.where(days_b >= D0305, 2, 1))
uniq_days, dayid = np.unique(days_b, return_inverse=True)
n_blocks = X.shape[0]
print(f"Task A: {n_blocks} non-empty 5-min blocks, {len(uniq_days)} calendar days",
      flush=True)
print("epoch class counts (early/mid/late):", np.bincount(epoch).tolist(), flush=True)
print("first/last block UTC:",
      pd.to_datetime(int(tstart[0]), utc=True),
      pd.to_datetime(int(tstart[-1]), utc=True), flush=True)

def cv_acc(Xm, y, seed=0):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_score(clf, Xm, y, cv=cv)

def null_reps(Xm, y, n_reps=20, seed=1234):
    rng = np.random.default_rng(seed)
    return [float(np.mean(cv_acc(Xm, rng.permutation(y)))) for _ in range(n_reps)]

rows = []
def run_decode(name, Xm, y, elec_desc, chance, n_classes, class_counts):
    folds = cv_acc(Xm, y)
    acc_mean, acc_sd = float(folds.mean()), float(folds.std())
    nulls = null_reps(Xm, y)
    p_emp = (1 + sum(n >= acc_mean for n in nulls)) / (1 + len(nulls))
    rows.append(dict(
        analysis=name, n_blocks=int(Xm.shape[0]), n_features=int(Xm.shape[1]),
        electrode_set=elec_desc, accuracy_mean=round(acc_mean, 4),
        accuracy_sd=round(acc_sd, 4),
        fold_accuracies=",".join(f"{f:.4f}" for f in folds),
        chance=chance, n_classes=n_classes, class_counts=class_counts,
        null_mean=round(float(np.mean(nulls)), 4),
        null_sd=round(float(np.std(nulls)), 4),
        null_min=round(float(np.min(nulls)), 4),
        null_max=round(float(np.max(nulls)), 4),
        null_p_empirical=round(p_emp, 4)))
    print(f"{name}: acc={acc_mean:.4f} +/- {acc_sd:.4f} (chance {chance}) "
          f"null={np.mean(nulls):.4f} +/- {np.std(nulls):.4f} "
          f"[{np.min(nulls):.4f},{np.max(nulls):.4f}] p_emp={p_emp:.4f}", flush=True)

run_decode("epoch_all32", X, epoch, "64-95 (all 32)",
           chance="1/3", n_classes=3,
           class_counts=",".join(map(str, np.bincount(epoch).tolist())))
run_decode("dayid_all32", X, dayid, "64-95 (all 32)",
           chance=f"1/{len(uniq_days)}", n_classes=len(uniq_days),
           class_counts=f"n_days={len(uniq_days)}")

# Ablation: persistent-8 vs collapsing-8. Persistence = late-third / early-third
# mean rate per electrode, computed from these blocks (time-ordered thirds).
nb = X.shape[0]
n3 = nb // 3
early_m = X[:n3].mean(axis=0)
late_m = X[2 * n3:].mean(axis=0)
persist_ratio = late_m / (early_m + 1e-9)
cand = np.arange(8, 24)                      # electrodes 72-87
top8 = cand[np.argsort(persist_ratio[cand])[::-1][:8]]
top8_elec = sorted(int(c + 64) for c in top8)
print("persistence ranking (elec: late/early):",
      {int(c + 64): round(float(persist_ratio[c]), 3) for c in cand}, flush=True)
print("selected persistent-8:", top8_elec, flush=True)
run_decode("epoch_persistent8", X[:, top8], epoch,
           f"persistent-8: {top8_elec}", chance="1/3", n_classes=3,
           class_counts=",".join(map(str, np.bincount(epoch).tolist())))
run_decode("epoch_collapsing8", X[:, :8], epoch, "collapsing-8: 64-71",
           chance="1/3", n_classes=3,
           class_counts=",".join(map(str, np.bincount(epoch).tolist())))

dec_df = pd.DataFrame(rows)
dec_path = BASE + "/results/lifespan_decoding_fs369.csv"
dec_df.to_csv(dec_path, index=False)
print("wrote", dec_path, flush=True)

# ================================================================= Task B
baseline_rate = pre_c / 0.45                            # Hz, per stim
evoked_ratio = np.where(pre_c > 0, post_c * 9.0 / pre_c, np.nan)

def group_sel(g):
    return grp_k == g

post = group_sel("post"); pre20 = group_sel("pre20"); pre21 = group_sel("pre21")
pre_all = pre20 | pre21

def mwu(x, y):
    r = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return float(r.statistic), float(r.pvalue)

brows = []
def run_contrast(contrast, metric, a, b, note=""):
    xa, xb = a[~np.isnan(a)], b[~np.isnan(b)]
    U, p = mwu(xa, xb)
    m1, m2 = float(np.median(xa)), float(np.median(xb))
    brows.append(dict(contrast=contrast, metric=metric,
                      n_g1=int(len(xa)), n_g2=int(len(xb)),
                      median_g1=round(m1, 5), median_g2=round(m2, 5),
                      median_diff_g1_minus_g2=round(m1 - m2, 5),
                      U_statistic=round(U, 1), p_value=p, notes=note))
    print(f"{contrast} | {metric}: n1={len(xa)} n2={len(xb)} "
          f"med1={m1:.5f} med2={m2:.5f} diff={m1-m2:+.5f} U={U:.1f} p={p:.3g} {note}",
          flush=True)

zpre_post = int(((pre_c == 0) & post).sum())
zpre_pre = int(((pre_c == 0) & pre_all).sum())
zpre_20 = int(((pre_c == 0) & pre20).sum())
zpre_21 = int(((pre_c == 0) & pre21).sum())
run_contrast("post_vs_pre_change", "evoked_ratio",
             evoked_ratio[post], evoked_ratio[pre_all],
             note=f"zero-pre excluded: post {zpre_post}, pre {zpre_pre}")
run_contrast("control_20_vs_21", "evoked_ratio",
             evoked_ratio[pre20], evoked_ratio[pre21],
             note=f"zero-pre excluded: 02-20 {zpre_20}, 02-21 {zpre_21}")
run_contrast("post_vs_pre_change", "baseline_rate_hz",
             baseline_rate[post], baseline_rate[pre_all])
run_contrast("control_20_vs_21", "baseline_rate_hz",
             baseline_rate[pre20], baseline_rate[pre21])

mc_df = pd.DataFrame(brows)
mc_path = BASE + "/results/medium_change_fs369.csv"
mc_df.to_csv(mc_path, index=False)
print("wrote", mc_path, flush=True)

# ------------------------------------------------------------- run log append
log = f"""
[lifespan_fs369.py run {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.gmtime())}]
Proto A (a1=a2=1.5,d1=d2=300,2 pulses,period=10000,pol=1,shape=0): {len(sns)} stims total.
  groups: pre20(02-20)={int(pre20.sum())}, pre21(02-21)={int(pre21.sum())}, post(02-28)={int(post.sum())}
Task A: {n_blocks} blocks, epoch counts early/mid/late={np.bincount(epoch).tolist()}, days={len(uniq_days)}
  persistent-8 (late/early top-8 of 72-87): {top8_elec}
  ratios: { {int(c+64): round(float(persist_ratio[c]),3) for c in cand} }
Task A decoding (mean+/-sd, 5-fold strat CV; nulls = 20 label shuffles):
"""
for r in rows:
    log += (f"  {r['analysis']}: acc={r['accuracy_mean']} +/- {r['accuracy_sd']} "
            f"(chance {r['chance']}) null={r['null_mean']} +/- {r['null_sd']} "
            f"[{r['null_min']},{r['null_max']}] p_emp={r['null_p_empirical']}\n")
log += "Task B (Proto A only; MWU two-sided asymptotic):\n"
for b in brows:
    log += (f"  {b['contrast']} | {b['metric']}: n1={b['n_g1']} n2={b['n_g2']} "
            f"med1={b['median_g1']} med2={b['median_g2']} diff={b['median_diff_g1_minus_g2']:+f} "
            f"U={b['U_statistic']} p={b['p_value']:.3g} {b['notes']}\n")
log += f"elapsed {time.time()-t0:.0f}s\n"
with open(BASE + "/results/run_log.txt", "a") as f:
    f.write(log)
print("appended run_log.txt", flush=True)
print(f"== done in {time.time()-t0:.0f}s ==", flush=True)
