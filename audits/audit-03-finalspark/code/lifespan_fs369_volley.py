"""Task B follow-up: volley-level honest analysis.

Discovery: FinalSpark delivers stimulation as SIMULTANEOUS whole-array volleys
(32 electrodes, same timestamp). The 640 pre-change "stims" are 20 volleys
(10/day); the 58,144 post-change stims are 1,817 volleys. Per-stim array-wide
windows are IDENTICAL for all 32 stims in a volley -> 32x pseudoreplication.
The honest unit of analysis is the volley.

This script: one chunked events pass accumulating per-volley pre/post counts
for every Proto A volley, then volley-level contrasts + habituation check
(first 100 vs last 100 volleys on 02-28) + other-days descriptives.
Appends volley-level rows (unit='volley') to results/medium_change_fs369.csv.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark"
F = BASE + "/data/fs369/fs369_package.hdf5"
MS = 1_000_000
DAY = 86_400_000_000_000

t0 = time.time()
st = pd.read_hdf(F, key="fs369_wholelife_stimulations")
spec = dict(a1=1.5, a2=1.5, d1=300.0, d2=300.0, nb_stim_pulse=2.0,
            pulse_train_period=10000.0, stim_polarity=1.0, stim_shape=0.0)
mask = np.ones(len(st), dtype=bool)
for k, v in spec.items():
    mask &= (st[k].values == v)
vts = np.unique(st.loc[mask, "time_of_stim"].values.astype("datetime64[ns]").astype("int64"))
print(f"Proto A volleys (unique timestamps): {len(vts)}", flush=True)
vdays = vts // DAY


def day_of(s):
    return int(pd.Timestamp(s, tz="UTC").value // DAY)


D = {d: day_of(f"2025-{d}") for d in ["02-20", "02-21", "02-28", "03-04", "03-06", "03-10"]}
vgrp = np.full(len(vts), "other", dtype=object)
vgrp[vdays == D["02-28"]] = "post"
vgrp[vdays == D["02-20"]] = "pre20"
vgrp[vdays == D["02-21"]] = "pre21"
label_of = {D["02-28"]: "post", D["02-20"]: "pre20", D["02-21"]: "pre21"}
for name, d in D.items():
    lab = label_of.get(d, "other")
    print(f"  volleys on {name}: group={lab} n={(vgrp == lab).sum()} "
          f"(day {name} total: {(vdays == d).sum()})", flush=True)

pre_v = np.zeros(len(vts), dtype=np.int64)
post_v = np.zeros(len(vts), dtype=np.int64)
store = pd.HDFStore(F, mode="r")
nrows = int(store.get_storer("fs369_wholelife_events").nrows)
CH = 4_000_000
nch = (nrows + CH - 1) // CH
for ci in range(nch):
    s0, s1 = ci * CH, min((ci + 1) * CH, nrows)
    df = store.select("fs369_wholelife_events", start=s0, stop=s1,
                      columns=["time_of_event"])
    t = df["time_of_event"].values.astype("datetime64[ns]").astype("int64")
    cmin, cmax = t[0], t[-1]
    lo = np.searchsorted(vts, cmin - 50 * MS)
    hi = np.searchsorted(vts, cmax + 500 * MS)
    if hi > lo:
        cs = vts[lo:hi]
        pre_v[lo:hi] += (np.searchsorted(t, cs - 50 * MS)
                         - np.searchsorted(t, cs - 500 * MS))
        post_v[lo:hi] += (np.searchsorted(t, cs + 50 * MS)
                          - np.searchsorted(t, cs))
    del df, t
store.close()
print(f"volley accumulation done ({time.time()-t0:.0f}s)", flush=True)

ratio_v = np.where(pre_v > 0, post_v * 9.0 / pre_v, np.nan)
base_v = pre_v / 0.45
zpre = int((pre_v == 0).sum())
print(f"volleys with zero pre-window events: {zpre}", flush=True)

post = vgrp == "post"; pre20 = vgrp == "pre20"; pre21 = vgrp == "pre21"
pre_all = pre20 | pre21


def mwu(x, y):
    r = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return float(r.statistic), float(r.pvalue)


rows = []


def run_contrast(contrast, metric, a, b, note=""):
    xa, xb = a[~np.isnan(a)], b[~np.isnan(b)]
    U, p = mwu(xa, xb)
    m1, m2 = float(np.median(xa)), float(np.median(xb))
    meanr = ""
    rows.append(dict(unit="volley", contrast=contrast, metric=metric,
                     n_g1=int(len(xa)), n_g2=int(len(xb)),
                     median_g1=round(m1, 4), median_g2=round(m2, 4),
                     median_diff_g1_minus_g2=round(m1 - m2, 4),
                     U_statistic=round(U, 1), p_value=p, notes=note))
    print(f"{contrast} | {metric}: n1={len(xa)} n2={len(xb)} med1={m1:.4f} "
          f"med2={m2:.4f} diff={m1-m2:+.4f} U={U:.1f} p={p:.3g} {note}", flush=True)


run_contrast("post_vs_pre_change", "evoked_ratio", ratio_v[post], ratio_v[pre_all],
             note=f"honest n: {int(post.sum())} vs {int(pre_all.sum())} volleys; zero-pre excluded: {zpre}")
run_contrast("control_20_vs_21", "evoked_ratio", ratio_v[pre20], ratio_v[pre21])
run_contrast("post_vs_pre_change", "baseline_rate_hz", base_v[post], base_v[pre_all])
run_contrast("control_20_vs_21", "baseline_rate_hz", base_v[pre20], base_v[pre21])

# habituation check: first 100 vs last 100 volleys of the 02-28 session
pv = vts[post]
o = np.argsort(pv)
r_first = ratio_v[post][o[:100]]
r_last = ratio_v[post][o[-100:]]
run_contrast("habituation_0228_first100_vs_last100", "evoked_ratio", r_first, r_last)

# mean-based (report-style) ratios per day-group, for comparison with explore §4
print("\nmean-based ratios sum(post)/sum(pre)*9 per group:", flush=True)
for name, m in [("pre20", pre20), ("pre21", pre21), ("pre_all", pre_all),
                ("post", post), ("other_days", vgrp == "other")]:
    if m.sum():
        mr = post_v[m].sum() * 9.0 / max(pre_v[m].sum(), 1)
        print(f"  {name}: n_volleys={int(m.sum())} mean_ratio={mr:.3f} "
              f"mean_baseline_hz={pre_v[m].mean()/0.45:.2f}", flush=True)

# rewrite CSV with unit column
mc_path = BASE + "/results/medium_change_fs369.csv"
old = pd.read_csv(mc_path)
old.insert(0, "unit", "per_stim")
new = pd.DataFrame(rows)
both = pd.concat([old, new], ignore_index=True)
both.to_csv(mc_path, index=False)
print("rewrote", mc_path, flush=True)

log = f"""
[lifespan_fs369_volley.py run {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.gmtime())}]
VOLLEY-LEVEL correction: Proto A stims are simultaneous whole-array volleys
(32 electrodes, identical timestamps). Pre-change "n=640 stims" = 20 volleys
(10 on 02-20, 10 on 02-21); post-change = 1,817 volleys. Per-stim array-wide
windows are identical within a volley -> per-stim MWU had 32x pseudoreplication.
Honest unit = volley. No window contamination: Proto A volleys are >=1s from any
other volley (02-20/02-21: ~10s apart; 02-28: ~1.01s apart).
Volley-level contrasts (MWU two-sided asymptotic):
"""
for r in rows:
    log += (f"  {r['contrast']} | {r['metric']}: n1={r['n_g1']} n2={r['n_g2']} "
            f"med1={r['median_g1']} med2={r['median_g2']} diff={r['median_diff_g1_minus_g2']:+f} "
            f"U={r['U_statistic']} p={r['p_value']:.3g} {r['notes']}\n")
with open(BASE + "/results/run_log.txt", "a") as f:
    f.write(log)
print("appended run_log.txt", flush=True)
print(f"== done in {time.time()-t0:.0f}s ==", flush=True)
