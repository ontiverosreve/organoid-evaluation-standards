# Stage C release package — drug/control classification on organoid MEA

Machine-readable tables behind the headline result: **test accuracy 1.000**
for control vs. drug organoid blocks (Trujillo et al. 2019, Zenodo 4751759).

## Contents

- `config.yaml` — complete reproduction manifest (files + SHA-256, well,
  binning, reservoir, readout, splits, probe, shuffle-null convention).
- `environment.txt` — pinned Python/package versions.
- `runs.csv` — one row per run: alpha, seed, split type, ridge weighting,
  n_train, n_test, accuracy. Includes the 20-rep label-shuffle null.
- `blocks.csv` — one row per block (48): file, in-file index, start time,
  label, predicted label (alpha=0.005, seed 11), quality weight, probe
  z-score, flagged, lean direction.
- `verify.py` — independent check of the headline claims; exits non-zero
  on failure.

## Raw data

Zenodo record 4751759 (CC-BY-4.0): <https://zenodo.org/records/4751759>

- `LFP_Sp_161217.mat` (control) — 1,283,529,863 bytes —
  sha256 `f8328c074773cddd371134eda6cb14182eb34e734a1261513d05a5133a8ae95e`
- `LFP_Sp_161217_Drugs.mat` (drug) — 1,357,743,120 bytes —
  sha256 `e680bf7c1d5a3a2bf8d1f959ac6d440a5a83b6101913f234efdeead37cdc6780`

Verify with: `sha256sum LFP_Sp_161217*.mat`

## Reproduction

1. Bin spikes (well 4, 50 ms bins, per-channel standardization) into
   `binned_well4.npz`:
   `python3 ../bin_cache.py`
   (A cached copy already exists at `../binned_well4.npz`.)
2. Regenerate this package deterministically:
   `python3 ../make_release.py`
   It re-runs the pipeline (same seeds), aborts if the headline numbers
   do not reproduce, and rewrites every file in this directory.
3. Check the claims: `python3 verify.py`

The original pipeline is `../run_drug_classification.py`
(2026-09-19; the 20-rep shuffle-null convention is in `../diagnose2.py`).

## What verify.py asserts

- interleaved: accuracy exactly 1.000 at alpha = 0.005, 0.01, 0.02,
  each across seeds 7, 11, 22.
- chronological: accuracy exactly 1.000 (alpha=0.005, seeds 7, 11, 22).
- label-shuffle null (20 reps): max <= 0.750 and 0/20 reach 1.0.
- blocks.csv: 48 rows, 24/24 class balance, exactly 6 flagged blocks
  (indices 4, 6, 21, 30, 39, 45), all with |z| > 2.0.

## Caveats

Labels are file-level ("drug" is confounded with recording session) and
the drug class is pharmacologically heterogeneous. The regime probe flags
blocks for manual review and never inverts predictions.
