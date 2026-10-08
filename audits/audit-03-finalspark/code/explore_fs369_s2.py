"""Stage 2: stim protocol clustering + schedule, events scan probe."""
import pandas as pd, numpy as np, time
F = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/data/fs369/fs369_package.hdf5"

st = pd.read_hdf(F, key="fs369_wholelife_stimulations")
set_cols = [c for c in st.columns if c not in ("electrode","time_of_stim")]
print("setting cols:", set_cols)
g = st.groupby(set_cols, dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
print("\n# distinct protocols:", len(g))
pd.set_option("display.width", 250)
print(g.to_string())

# schedule: daily stim counts
st["date"] = st["time_of_stim"].dt.date
daily = st.groupby("date").size()
print("\ndaily stim counts (nonzero days):")
print(daily.to_string())
print("\nn days with stims:", (daily>0).sum(), " zero-stim days:", (daily==0).sum())

# interval analysis: median inter-stim interval overall and per electrode
ts = st.sort_values("time_of_stim")["time_of_stim"]
dt = ts.diff().dropna().dt.total_seconds()
print("\ninter-stim interval (s): median", dt.median(), "mean", dt.mean(), "min", dt.min(), "max", dt.max())
ts_e = st.sort_values(["electrode","time_of_stim"]).groupby("electrode")["time_of_stim"].diff().dt.total_seconds()
print("per-electrode inter-stim interval median (s):\n", ts_e.groupby(st.sort_values(["electrode","time_of_stim"])["electrode"]).median().to_string())

# probe events table
t0=time.time()
store = pd.HDFStore(F, mode="r")
ev = store["fs369_wholelife_events"]
print("\nevents nrows:", ev.nrows, "table format:", type(store.get_storer("fs369_wholelife_events")).__name__)
print("events columns:", ev.colnames if hasattr(ev,'colnames') else store.select("fs369_wholelife_events", start=0, stop=5).columns.tolist())
chunk = store.select("fs369_wholelife_events", start=0, stop=200000)
print("200k chunk read took %.2fs" % (time.time()-t0))
print(chunk.head(3).to_string())
print(chunk.dtypes)
store.close()
