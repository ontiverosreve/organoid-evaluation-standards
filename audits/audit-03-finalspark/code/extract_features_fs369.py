"""Step 2: trial-level feature extraction for the amplitude cohort.

Trials = stimulation TIMESTAMPS (n=120): every timestamp stimulates all 32
electrodes simultaneously with a single amplitude, so a per-stim-row trial
would duplicate the feature matrix 32x. Per timestamp:
  - X_full:  50 x 32  spike counts, 1 ms bins, [0,50) ms post-stim
  - X_base:  50 x 32  spike counts, 1 ms bins, [-550,-500) ms pre-stim (null)
Labels: amplitude in {0.8, 1.0, 1.5} (chance = 1/3).
Events table scanned in 4M-row chunks via HDFStore.select(start, stop).
"""
import pandas as pd, numpy as np, time, collections

F = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/data/fs369/fs369_package.hdf5"
OUT = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/results/fs369_trial_features.npz"

sel = pd.read_csv("/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/results/fs369_stim_cohort.csv",
                  parse_dates=["time_of_stim"])
ts = (sel.groupby("time_of_stim")
         .agg(amp=("a1", "first"), date=("time_of_stim", lambda s: s.iloc[0].date()))
         .reset_index().sort_values("time_of_stim").reset_index(drop=True))
assert (ts.groupby("time_of_stim")["amp"].nunique() == 1).all()
n = len(ts)
T_ns = ts["time_of_stim"].values.astype("datetime64[ns]").astype("int64")
amp2lab = {0.8: 0, 1.0: 1, 1.5: 2}
y = ts["amp"].map(amp2lab).values.astype(np.int64)
day = ts["date"].astype(str).values
print(f"n trials: {n}")

X_full = np.zeros((n, 50, 32), dtype=np.int32)
X_base = np.zeros((n, 50, 32), dtype=np.int32)
MS = 1_000_000

store = pd.HDFStore(F, mode="r")
nrows = int(store.get_storer("fs369_wholelife_events").nrows)
CH = 4_000_000
n_chunks = (nrows + CH - 1) // CH
t0 = time.time()
hits_full = np.zeros(n, dtype=np.int64)
hits_base = np.zeros(n, dtype=np.int64)
for ci in range(n_chunks):
    s0, s1 = ci * CH, min((ci + 1) * CH, nrows)
    df = store.select("fs369_wholelife_events", start=s0, stop=s1,
                      columns=["electrode", "time_of_event"])
    t = df["time_of_event"].values.astype("datetime64[ns]").astype("int64")
    e = df["electrode"].values.astype(np.int64)
    # candidate trials whose [-550,+50]ms window overlaps this chunk
    lo = np.searchsorted(t, T_ns - 550 * MS)
    hi = np.searchsorted(t, T_ns + 50 * MS)
    cand = np.nonzero(hi > lo)[0]
    for i in cand:
        tsl = t[lo[i]:hi[i]]; esl = e[lo[i]:hi[i]]
        rel = tsl - T_ns[i]
        m_full = (rel >= 0) & (rel < 50 * MS)
        if m_full.any():
            b = (rel[m_full] // MS).astype(np.int64)
            ei = (esl[m_full] - 64).astype(np.int64)
            ok = (ei >= 0) & (ei < 32)
            np.add.at(X_full[i], (b[ok], ei[ok]), 1)
            hits_full[i] += int(m_full.sum())
        m_base = (rel >= -550 * MS) & (rel < -500 * MS)
        if m_base.any():
            b = ((rel[m_base] + 550 * MS) // MS).astype(np.int64)
            ei = (esl[m_base] - 64).astype(np.int64)
            ok = (ei >= 0) & (ei < 32)
            np.add.at(X_base[i], (b[ok], ei[ok]), 1)
            hits_base[i] += int(m_base.sum())
    del df, t, e
    if (ci + 1) % 6 == 0 or ci == n_chunks - 1:
        print(f"  chunk {ci+1}/{n_chunks}, {time.time()-t0:.0f}s", flush=True)
store.close()

print("scan done in", round(time.time() - t0, 1), "s")
# sanity: evoked events per trial by amplitude
for lab, amp in enumerate([0.8, 1.0, 1.5]):
    m_ = y == lab
    print(f"amp {amp}: trials={m_.sum()} mean evoked[0,50)ms/trial={hits_full[m_].mean():.1f} "
          f"mean base[-550,-500)ms/trial={hits_base[m_].mean():.1f} ratio={hits_full[m_].mean()/max(hits_base[m_].mean(),1e-9):.2f}")
print("trials with zero evoked events:", int((hits_full == 0).sum()))

np.savez_compressed(OUT, X_full=X_full, X_base=X_base, y=y, day=day,
                    amp=np.array([0.8, 1.0, 1.5]),
                    trial_time_ns=T_ns)
print("saved", OUT)
