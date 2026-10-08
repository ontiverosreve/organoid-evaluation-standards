"""Stage 4: incubator anomalies + door openings by date + stim protocol x day breakdown."""
import pandas as pd, numpy as np
F = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/data/fs369/fs369_package.hdf5"

def load(tbl):
    return pd.read_hdf(F, key=f"fs369_wholelife_incubator_{tbl}")

temp = load("temperature"); hum = load("humidity"); co2 = load("CO2"); o2 = load("O2"); pres = load("pressure"); door = load("door_opening")

def exc(df, vcol, lo, hi, name):
    v = pd.to_numeric(df[vcol], errors="coerce")
    bad = df[(v < lo) | (v > hi)].copy()
    print(f"\n{name}: n={len(df)} range=[{v.min():.2f},{v.max():.2f}] mean={v.mean():.2f}")
    print(f"  excursions outside [{lo},{hi}]: {len(bad)}")
    if len(bad):
        bad["date"] = pd.to_datetime(bad["time"], utc=True).dt.date
        print(bad.groupby("date").size().to_string())
        print("  extreme rows:")
        print(bad.sort_values(vcol).head(3)[["time", vcol]].to_string(index=False))
        print(bad.sort_values(vcol).tail(3)[["time", vcol]].to_string(index=False))

exc(temp, "incubator_temperature", 33.5, 34.6, "TEMPERATURE (degC)")
exc(hum, "incubator_humidity", 70, 95, "HUMIDITY (%)")
exc(co2, "incubator_CO2", 4.0, 5.5, "CO2 (%)")
exc(o2, "incubator_O2", 15.5, 18.0, "O2 (%)")
exc(pres, "incubator_pressure", 900, 1000, "PRESSURE")

print("\nDOOR openings by date:")
door["dt"] = pd.to_datetime(door["time"], utc=True)
op = door[door["incubator_door_opening"] == True].copy()
op["date"] = op["dt"].dt.date
print(op.groupby("date").size().to_string())
print("total openings:", len(op))

# stim protocol x day for top protocols
st = pd.read_hdf(F, key="fs369_wholelife_stimulations")
st["date"] = st["time_of_stim"].dt.date
set_cols = [c for c in st.columns if c not in ("electrode","time_of_stim","date")]
st["proto"] = st.groupby(set_cols, dropna=False).ngroup()
top = st["proto"].value_counts().head(6)
print("\ntop-6 protocol ids:", top.to_dict())
ct = pd.crosstab(st["date"], st["proto"])
cols = top.index.tolist()
print(ct[cols].to_string())
# key params of top protocols
for pid in cols:
    row = st[st["proto"]==pid].iloc[0]
    print(f"proto {pid}: n={int(top[pid])} a1={row.a1} a2={row.a2} d1={row.d1} d2={row.d2} npulse={row.nb_stim_pulse} ptp={row.pulse_train_period} pol={row.stim_polarity} shape={row.stim_shape} elecs={st[st['proto']==pid]['electrode'].nunique()}")
# stim hours within a big day: 2025-02-28 hourly
d = st[st["date"].astype(str)=="2025-02-28"].copy()
d["hour"] = d["time_of_stim"].dt.hour
print("\n2025-02-28 stim count by hour:\n", d.groupby("hour").size().to_string())
d2 = st[st["date"].astype(str)=="2025-02-20"].copy(); d2["hour"]=d2["time_of_stim"].dt.hour
print("\n2025-02-20 stim count by hour:\n", d2.groupby("hour").size().to_string())
