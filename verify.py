"""Independent verification of the Stage C headline claims.

Reads runs.csv and blocks.csv from this directory and asserts:
  1. interleaved accuracy == 1.000 at each alpha (0.005, 0.01, 0.02)
     across seeds (7, 11, 22)
  2. chronological accuracy == 1.000 (alpha=0.005, seeds 7, 11, 22)
  3. label-shuffle null (20 reps): max <= 0.750 and 0/20 reach 1.0
  4. blocks.csv: 48 rows, 24/24 balance, exactly 6 flagged blocks
     (indices 4, 6, 21, 30, 39, 45), all with |z| > 2.0
Exits 0 if all checks pass, 1 otherwise.
"""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_FLAGS = {4, 6, 21, 30, 39, 45}
failures = []

def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), name, detail)
    if not cond:
        failures.append(name)

def load(name):
    with open(os.path.join(HERE, name), newline="") as fo:
        return list(csv.DictReader(fo))

runs = load("runs.csv")
blocks = load("blocks.csv")

inter = [r for r in runs if r["split_type"] == "interleaved"]
for a in ("0.005", "0.01", "0.02"):
    rows = [r for r in inter if r["alpha"] == a]
    seeds = sorted(r["reservoir_seed"] for r in rows)
    accs = [float(r["accuracy"]) for r in rows]
    check(f"interleaved alpha={a}: 3 seeds, all acc==1.000",
          len(rows) == 3 and seeds == ["11", "22", "7"] and all(x == 1.0 for x in accs),
          f"seeds={seeds} accs={accs}")

chrono = [r for r in runs if r["split_type"] == "chronological"]
caccs = [float(r["accuracy"]) for r in chrono]
check("chronological: 3 seeds, all acc==1.000",
      len(chrono) == 3 and all(x == 1.0 for x in caccs), f"accs={caccs}")

shuf = [r for r in runs if r["split_type"] == "label-shuffle"]
saccs = [float(r["accuracy"]) for r in shuf]
check("shuffle null: 20 reps, max<=0.750, none reach 1.0",
      len(shuf) == 20 and max(saccs) <= 0.750 and sum(x >= 1.0 for x in saccs) == 0,
      f"n={len(shuf)} max={max(saccs):.3f} n_at_1.0={sum(x >= 1.0 for x in saccs)}")

check("blocks.csv: 48 rows", len(blocks) == 48, f"n={len(blocks)}")
lab = [int(b["label"]) for b in blocks]
check("blocks.csv: 24/24 class balance",
      lab.count(0) == 24 and lab.count(1) == 24,
      f"control={lab.count(0)} drug={lab.count(1)}")
flagged = {int(b["block_index"]) for b in blocks if int(b["flagged"]) == 1}
zflag = [abs(float(b["probe_z"])) for b in blocks if int(b["flagged"]) == 1]
check("blocks.csv: exactly 6 flagged (4,6,21,30,39,45), all |z|>2.0",
      flagged == EXPECTED_FLAGS and all(z > 2.0 for z in zflag),
      f"flagged={sorted(flagged)}")

if failures:
    print(f"\n{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
print("\nAll headline claims verified.")
