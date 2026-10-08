"""Stage 1: metadata, stimulations, incubator tables (small), events probe."""
import pandas as pd, numpy as np, sys
F = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/data/fs369/fs369_package.hdf5"

print("=== METADATA ===")
md = pd.read_hdf(F, key="fs369_wholelife_metadata")
print(md.shape)
for c in md.columns:
    print(f"{c!r}: {md[c].iloc[0]!r}")

print("\n=== STIMULATIONS ===")
st = pd.read_hdf(F, key="fs369_wholelife_stimulations")
print(st.shape)
print(st.dtypes.to_string())
t0, t1 = st["time_of_stim"].min(), st["time_of_stim"].max()
print("time span (ns):", t0, "->", t1)
print("time span (UTC):", pd.to_datetime(t0, utc=True), "->", pd.to_datetime(t1, utc=True))
print("unique electrodes stimulated:", st["electrode"].nunique(), sorted(st["electrode"].unique().tolist()))
print("rows per electrode:\n", st["electrode"].value_counts().sort_index().to_string())

print("\n=== INCUBATOR TABLES ===")
for tbl in ["temperature","humidity","CO2","O2","pressure","door_opening"]:
    key = f"fs369_wholelife_incubator_{tbl}"
    df = pd.read_hdf(F, key=key)
    vcol = [c for c in df.columns if c not in ("time","incubator_id")][0]
    print(f"\n{tbl}: rows={len(df)}, value_col={vcol!r}")
    if len(df):
        print("  time span:", pd.to_datetime(df.time.min(),utc=True), "->", pd.to_datetime(df.time.max(),utc=True))
        v = pd.to_numeric(df[vcol], errors="coerce") if tbl!="door_opening" else df[vcol]
        try:
            print("  min/max/mean:", float(v.min()), float(v.max()), float(v.mean()))
        except Exception as e:
            print("  stats err:", e, "unique values:", df[vcol].unique()[:10])
        if tbl=="door_opening":
            print("  True count (openings):", int((df[vcol]==True).sum()), "False count:", int((df[vcol]==False).sum()))
            op = df[df[vcol]==True]
            if len(op):
                print("  first/last openings:", pd.to_datetime(op.time.min(),utc=True), "->", pd.to_datetime(op.time.max(),utc=True))
