"""Stage 5 (redo): stim-response with stim-indexed accumulation (unbiased n)."""
import pandas as pd, numpy as np, time, json
F = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/data/fs369/fs369_package.hdf5"
OUT = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/results/fs369_stim_response.json"

st = pd.read_hdf(F, key="fs369_wholelife_stimulations")
stim_ns = st["time_of_stim"].values.astype("datetime64[ns]").astype("int64")
e_all = st["electrode"].values.astype(np.int64)
specs = {
    "P1_1.5uA_pos": dict(a1=1.5,a2=1.5,d1=300.0,d2=300.0,nb_stim_pulse=2.0,stim_polarity=1.0,stim_shape=0.0),
    "P2_2.5uA_neg": dict(a1=2.5,a2=2.5,d1=300.0,d2=300.0,nb_stim_pulse=2.0,stim_polarity=0.0,stim_shape=0.0),
    "P3_0.8uA_100us_neg": dict(a1=0.8,a2=0.8,d1=100.0,d2=100.0,nb_stim_pulse=2.0,stim_polarity=0.0,stim_shape=0.0),
}
MS = 1_000_000
# per-electrode stim arrays for the 3 protocols
stims_by_e = {}   # ee -> list of (name, tau)
for name, spec in specs.items():
    m = np.ones(len(st), dtype=bool)
    for k, v in spec.items():
        m &= (st[k].values == v)
    for ee, tau in zip(e_all[m], stim_ns[m]):
        stims_by_e.setdefault(int(ee), []).append((name, int(tau)))
acc = {}  # (ee, idx) -> [pre, post]; use dict of arrays per electrode
for ee, lst in stims_by_e.items():
    acc[ee] = {"names": [n for n, _ in lst], "tau": np.array([t for _, t in lst]),
               "pre": np.zeros(len(lst), dtype=np.int64), "post": np.zeros(len(lst), dtype=np.int64)}

store = pd.HDFStore(F, mode="r")
nrows = int(store.get_storer("fs369_wholelife_events").nrows)
CH = 4_000_000
n_chunks = (nrows + CH - 1) // CH
t0 = time.time()
for ci in range(n_chunks):
    s0, s1 = ci * CH, min((ci+1)*CH, nrows)
    df = store.select("fs369_wholelife_events", start=s0, stop=s1, columns=["electrode","time_of_event"])
    t = df["time_of_event"].values.astype("datetime64[ns]").astype("int64")
    e = df["electrode"].values.astype(np.int64)
    for ee in np.unique(e):
        A = acc.get(int(ee))
        if A is None: continue
        te = np.sort(t[e == ee])
        cmin, cmax = te[0], te[-1]
        tau = A["tau"]
        rel_idx = np.where((tau - 500*MS <= cmax) & (tau + 50*MS >= cmin))[0]
        if not len(rel_idx): continue
        tr = tau[rel_idx]
        pre = np.searchsorted(te, tr - 50*MS) - np.searchsorted(te, tr - 500*MS)
        post = np.searchsorted(te, tr + 50*MS) - np.searchsorted(te, tr)
        A["pre"][rel_idx] += pre
        A["post"][rel_idx] += post
    del df, t, e
    if (ci+1) % 8 == 0: print(f"chunk {ci+1}/{n_chunks}", flush=True)
store.close()

res = {}
for name in specs:
    pre_tot = post_tot = n = 0
    n_with_response = 0
    for ee, A in acc.items():
        m = np.array(A["names"]) == name
        pre_tot += int(A["pre"][m].sum()); post_tot += int(A["post"][m].sum()); n += int(m.sum())
        n_with_response += int(((A["post"][m] > A["pre"][m] * (0.05/0.45))).sum())
    pre_hz = pre_tot/(n*0.45); post_hz = post_tot/(n*0.05)
    res[name] = {"n_stims": n, "pre_events": pre_tot, "post_events": post_tot,
                 "baseline_Hz": round(pre_hz,3), "evoked_Hz": round(post_hz,3),
                 "ratio": round(post_hz/pre_hz,3) if pre_hz else None,
                 "frac_stims_post_gt_baselinex": round(n_with_response/n,4)}
    print(name, res[name])
json.dump(res, open(OUT,"w"), indent=1)
print("saved", OUT, f"({time.time()-t0:.0f}s)")
