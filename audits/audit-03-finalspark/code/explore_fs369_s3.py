"""Stage 3: chunked scan of the 94M-row events table + stim artifact/response analysis."""
import pandas as pd, numpy as np, time, collections, json
F = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/data/fs369/fs369_package.hdf5"
OUT = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/results/fs369_explore_s3.json"

store = pd.HDFStore(F, mode="r")
storer = store.get_storer("fs369_wholelife_events")
print("storer type:", type(storer).__name__, "nrows:", storer.nrows)
nrows = int(storer.nrows)

# stim times per electrode (int64 ns)
st = pd.read_hdf(F, key="fs369_wholelife_stimulations")
stim_ns_all = st["time_of_stim"].values.astype("datetime64[ns]").astype("int64")
e_all = st["electrode"].values.astype(np.int64)
stim_by_e = {}
for e in np.unique(e_all):
    stim_by_e[int(e)] = np.sort(stim_ns_all[e_all == e])

# top-3 protocols (by exact setting match, from stage 2)
set_cols = ['a1','a2','enable_amp','enable_chargerecovery','d1','d2','nb_stim_pulse',
            'post_stim_amp_settle','post_stim_charge_recov_off','post_stim_charge_recov_on',
            'post_trigger_delay','pre_stim_amp_settle','pulse_train_period','refractory_period',
            'stim_polarity','stim_shape']
protos = [
    ("P1_1.5uA_pos", dict(a1=1.5,a2=1.5,d1=300.0,d2=300.0,nb_stim_pulse=2.0,stim_polarity=1.0,stim_shape=0.0)),
    ("P2_2.5uA_neg", dict(a1=2.5,a2=2.5,d1=300.0,d2=300.0,nb_stim_pulse=2.0,stim_polarity=0.0,stim_shape=0.0)),
    ("P3_0.8uA_100us_neg", dict(a1=0.8,a2=0.8,d1=100.0,d2=100.0,nb_stim_pulse=2.0,stim_polarity=0.0,stim_shape=0.0)),
]
proto_stims = {}  # name -> dict electrode -> sorted ns array
for name, spec in protos:
    m = np.ones(len(st), dtype=bool)
    for k, v in spec.items():
        m &= (st[k].values == v)
    print(name, "n stims:", int(m.sum()))
    d = {}
    for e in np.unique(e_all[m]):
        d[int(e)] = np.sort(stim_ns_all[m & (e_all == e)])
    proto_stims[name] = d

MS = 1_000_000
DAY = 86_400_000_000_000

CH = 4_000_000
elec_counts = collections.Counter()
day_counts = collections.Counter()
half_counts = collections.Counter()  # (electrode, half)
volt_min = np.inf; volt_max = -np.inf; volt_sum = 0.0; volt_n = 0
volt_sample = []
tmin = np.iinfo(np.int64).max; tmax = np.iinfo(np.int64).min
mono_viol = 0; adj_dup = 0
art2 = 0; art10 = 0; art_tot = 0
per_elec_art = collections.Counter(); per_elec_tot = collections.Counter()
resp = {name: {"pre": 0, "post": 0, "n": 0} for name, _ in protos}
interval_counter = collections.Counter()
prev_last_t = None; prev_last_e = None

n_chunks = (nrows + CH - 1) // CH
t_start = time.time()
for ci in range(n_chunks):
    s0, s1 = ci * CH, min((ci + 1) * CH, nrows)
    df = store.select("fs369_wholelife_events", start=s0, stop=s1,
                      columns=["electrode", "time_of_event", "max_voltage_uv"])
    t = df["time_of_event"].values.astype("datetime64[ns]").astype("int64")
    e = df["electrode"].values.astype(np.int64)
    v = df["max_voltage_uv"].values.astype(np.float64)
    n = len(df)
    tmin = min(tmin, int(t.min())); tmax = max(tmax, int(t.max()))
    # monotonicity & adjacent duplicates (global order as stored)
    if prev_last_t is not None:
        if t[0] < prev_last_t: mono_viol += 1
        if t[0] == prev_last_t and e[0] == prev_last_e: adj_dup += 1
    mono_viol += int((np.diff(t) < 0).sum())
    adj_dup += int((((t[1:] == t[:-1]) & (e[1:] == e[:-1]))).sum())
    prev_last_t, prev_last_e = int(t[-1]), int(e[-1])
    # aggregates
    for ee, cc in zip(*np.unique(e, return_counts=True)):
        elec_counts[int(ee)] += int(cc)
    days = t // DAY
    for dd, cc in zip(*np.unique(days, return_counts=True)):
        day_counts[int(dd)] += int(cc)
    volt_min = min(volt_min, float(v.min())); volt_max = max(volt_max, float(v.max()))
    volt_sum += float(v.sum()); volt_n += n
    volt_sample.extend(v[::997].tolist())  # ~0.1% sample
    # interval subsample for periodicity check (every 200th interval, per chunk)
    iv = np.diff(t)[::200]
    for val in iv[:40000]:
        interval_counter[int(val)] += 1
    # per-electrode work
    for ee in np.unique(e):
        m = e == ee
        te = np.sort(t[m])
        per_elec_tot[int(ee)] += int(m.sum())
        # artifact: time since previous same-electrode stim
        sbt = stim_by_e.get(int(ee))
        if sbt is not None and len(sbt):
            idx = np.searchsorted(sbt, te, side="right") - 1
            has = idx >= 0
            dt = np.empty(len(te), dtype=np.int64); dt[has] = te[has] - sbt[idx[has]]; dt[~has] = np.iinfo(np.int64).max
            c2 = int((dt <= 2*MS).sum()); c10 = int((dt <= 10*MS).sum())
            art2 += c2; art10 += c10; per_elec_art[int(ee)] += c10
        art_tot += int(m.sum())
        # stim response for the 3 protocols
        cmin, cmax = te[0], te[-1]
        for name, _ in protos:
            pbt = proto_stims[name].get(int(ee))
            if pbt is None or not len(pbt): continue
            rel = pbt[(pbt - 500*MS <= cmax) & (pbt + 50*MS >= cmin)]
            if not len(rel): continue
            pre = np.searchsorted(te, rel - 50*MS) - np.searchsorted(te, rel - 500*MS)
            post = np.searchsorted(te, rel + 50*MS) - np.searchsorted(te, rel)
            resp[name]["pre"] += int(pre.sum()); resp[name]["post"] += int(post.sum())
            resp[name]["n"] += int(len(rel))
    # half split needs global midpoint: do later from tmin/tmax; skip here
    del df, t, e, v
    if (ci+1) % 6 == 0 or ci == n_chunks - 1:
        print(f"chunk {ci+1}/{n_chunks} done, {time.time()-t_start:.0f}s", flush=True)

store.close()
tmid = (tmin + tmax) // 2
# half counts need another pass? No — approximate: skip, do per-electrode early/late via day_counts instead.
res = {
    "nrows": nrows,
    "tmin_ns": tmin, "tmax_ns": tmax,
    "tmin_utc": str(pd.to_datetime(tmin, utc=True)), "tmax_utc": str(pd.to_datetime(tmax, utc=True)),
    "mono_violations": int(mono_viol), "adjacent_duplicates": int(adj_dup),
    "electrode_counts": {str(k): v for k, v in sorted(elec_counts.items())},
    "day_counts": {str(k): v for k, v in sorted(day_counts.items())},
    "volt_min": volt_min, "volt_max": volt_max, "volt_mean": volt_sum/volt_n,
    "volt_sample_n": len(volt_sample),
    "artifact_within_2ms": int(art2), "artifact_within_10ms": int(art10), "artifact_total_events": int(art_tot),
    "per_elec_art10": {str(k): v for k, v in sorted(per_elec_art.items())},
    "stim_response": resp,
    "top_intervals": interval_counter.most_common(15),
    "elapsed_s": time.time()-t_start,
}
vs = np.array(volt_sample)
res["volt_p1"], res["volt_p50"], res["volt_p99"] = float(np.percentile(vs,1)), float(np.percentile(vs,50)), float(np.percentile(vs,99))
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("saved", OUT)
print("total events:", nrows)
print("span:", res["tmin_utc"], "->", res["tmax_utc"])
print("mono violations:", mono_viol, "adjacent dups:", adj_dup)
print("artifact 2ms:", art2, f"({art2/art_tot*100:.3f}%)", "10ms:", art10, f"({art10/art_tot*100:.3f}%)")
print("stim response:", json.dumps(resp, indent=1))
print("top intervals (ns):", interval_counter.most_common(15))
print("electrodes:", len(elec_counts))
