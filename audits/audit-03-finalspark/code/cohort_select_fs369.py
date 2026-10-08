"""Step 1: select the sweep-grid amplitude cohort and report exact n per amplitude per day.

Cohort: d1=d2=300, stim_polarity=0, stim_shape=0, nb_stim_pulse=2,
        pulse_train_period=10000, a1=a2 in {0.8, 1.0, 1.5, 2.0, 4.0},
        stim date in {2025-02-20, 2025-02-21, 2025-02-28, 2025-03-06}.
"""
import pandas as pd, numpy as np, json

F = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/data/fs369/fs369_package.hdf5"
OUT = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/results/fs369_stim_cohort.csv"

st = pd.read_hdf(F, key="fs369_wholelife_stimulations")
m = (
    (st["d1"] == 300.0) & (st["d2"] == 300.0) &
    (st["stim_polarity"] == 0.0) & (st["stim_shape"] == 0.0) &
    (st["nb_stim_pulse"] == 2.0) & (st["pulse_train_period"] == 10000.0) &
    (st["a1"] == st["a2"]) &
    (st["a1"].isin([0.8, 1.0, 1.5, 2.0, 4.0]))
)
sel = st.loc[m].copy()
sel["date"] = sel["time_of_stim"].dt.date
days = [pd.Timestamp("2025-02-20").date(), pd.Timestamp("2025-02-21").date(),
        pd.Timestamp("2025-02-28").date(), pd.Timestamp("2025-03-06").date()]
sel = sel[sel["date"].isin(days)].copy()
sel = sel.sort_values("time_of_stim").reset_index(drop=True)

print("cohort n:", len(sel))
print("amps:", sorted(sel["a1"].unique()))
tab = pd.crosstab(sel["a1"], sel["date"], dropna=False)
print(tab.to_string())
print("\nmarginal by amp:\n", sel["a1"].value_counts().sort_index().to_string())
print("\nmarginal by day:\n", sel["date"].value_counts().sort_index().to_string())
print("\nper-electrode counts (should be ~equal across 64..95):")
print(sel["electrode"].value_counts().sort_index().to_string())
# amplitude schedule check: how interleaved are amplitudes within each day/electrode?
print("\nfirst 20 rows of cohort (stim order):")
print(sel[["time_of_stim", "electrode", "a1"]].head(20).to_string())

sel.to_csv(OUT, index=False)
print("\nsaved", OUT)

with open("/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/results/fs369_cohort_counts.json", "w") as f:
    json.dump({
        "n_total": int(len(sel)),
        "by_amp": {str(k): int(v) for k, v in sel["a1"].value_counts().sort_index().items()},
        "by_day": {str(k): int(v) for k, v in sel["date"].value_counts().sort_index().items()},
        "crosstab": {str(a): {str(d): int(c) for d, c in row.items()} for a, row in tab.iterrows()},
        "electrode_counts": {str(k): int(v) for k, v in sel["electrode"].value_counts().sort_index().items()},
        "time_min": str(sel["time_of_stim"].min()),
        "time_max": str(sel["time_of_stim"].max()),
    }, f, indent=1)
print("saved cohort counts json")
