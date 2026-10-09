# Honest-controls audit #6: Liu & Buonomano 2025, Nature Communications (ex vivo temporal-pattern learning)

**Status:** complete 2026-10-08 · **Data:** public, OSF node qspnv ("Early vs Late Evoked Timing.zip",
102,669,179 bytes, zip integrity verified). No confidentiality restrictions.

**Target paper:** Liu B, Buonomano DV. "Ex vivo cortical circuits learn to predict and spontaneously
replay temporal patterns." *Nature Communications* 16:3179 (2025). DOI 10.1038/s41467-025-58013-z.

**Published claim audited (Fig. 2 only — the evoked-timing claim):** ex vivo cortical organotypic slices
trained 24 h with dual-optical stimulation (Early: 10 ms ISI vs Late: 370 ms ISI) autonomously learn
the temporal structure — red-alone stimulation evokes training-interval-specific timed dynamics
(Early median event time 150±26 ms vs Late 479±32 ms, p<1e-9).

**Why this was the top audit target:** the paper itself reports investigators were **not blind** to
training condition during data collection, and the seven training batches were prepared "over the
course of a year" — a textbook batch-identity confound setup. The audit asks: does the Early-vs-Late
difference survive a batch-identity null?

**Data in this release:** 65 whole-cell patch-clamp recordings (30 Early / 35 Late neurons),
`Figure2_DATA_batch1_2.mat` — struct array `compMETA` with per-cell voltage traces (`keepTrace`,
n×30000 samples at 0.1 ms), stimulus metadata, and the authors' precomputed metrics. Seven batch
dates (YYMMDD): 230328, 230329, 230401, 230429, 230501, 230502, 230504. Five batches contain both
protocols; two (230328, 230329) are Late-only. Cell types: 43 pyramidal, 22 chr2+.

**Out of scope:** the spontaneous-replay and prediction-error (omission) claims — the data for
those figures is not in this release. This audit covers the evoked-timing claim only.

## Pipeline fidelity

The authors' MATLAB detection pipeline was reimplemented in Python (`code/liu_pipeline.py`) from
the released `.m` scripts. Event-time detection reproduces the authors' precomputed values exactly
(max |diff| = 0.20 ms, r = 1.000000, n=65). Center-of-gravity reproduces exactly (max |diff| =
0.21 ms, r = 1.000000) **only on a 600 ms window** — the released `.m` script sets a 2000 ms
window, but the `.mat` values match the 600 ms window noted in the script comments. Provenance
note: the script in the release is stale relative to the analysis that produced the data file.

## Verdict

**The headline survives honest controls — the first audit in this series where it does.**
The batch-confound hypothesis this audit was designed to test **fails**: the Early-vs-Late timing
difference holds within every mixed batch, survives a within-batch permutation null (0/2000), and
transfers across batches (leave-one-batch-out mean 0.865). The honest qualifications are real but
narrower than the confound hunt: response amplitude also differs by protocol, and batch fingerprints
exist — but neither explains the timing effect.

## Findings

### 1. Headline replicates (precomputed values)
Early n=30: median 78.7 ms, mean 149.8, sd 142.8. Late n=35: median 486.7 ms, mean 478.5, sd 190.7.
Mann-Whitney p = 1.85e-08 (paper: 150±26 vs 479±32 ms, p<1e-9). ✓

### 2. Within-batch stratification (§3.1) — the confound test
| batch | nE | nL | medE | medL | diff | p |
|---|---|---|---|---|---|---|
| 230401 | 7 | 6 | 116 | 463 | -347 | 0.0140 |
| 230429 | 4 | 4 | 77 | 547 | -471 | 0.0286 |
| 230501 | 8 | 3 | 221 | 482 | -261 | 0.1333 |
| 230502 | 3 | 11 | 46 | 504 | -458 | 0.0055 |
| 230504 | 8 | 3 | 50 | 332 | -282 | 0.0242 |

Direction consistent in all five mixed batches; 4/5 individually significant (230501: n=11,
direction consistent). Stratified permutation test (labels shuffled *within* batches, 2000 reps):
**0/2000** as-or-more extreme than the observed -407.9 ms. Batch magnitude varies (-261…-471 ms);
batch does not explain the effect.

### 3. Batch-identity classifier null (§3.1)
Logistic regression on trace features, leave-one-out, predicting batch within protocol:
Early 0.433 (chance 0.20, n=30) — batch fingerprints *do* exist in the Early group;
Late 0.171 (chance 0.143, n=35) — near chance. Batches are not identical, but §2 shows the
protocol effect survives batch structure anyway.

### 4. Label-shuffle null (§3.2)
200 repetitions shuffling protocol labels: 0 as-or-more extreme than observed; max shuffle
median-difference 174.4 ms vs observed -407.9 ms.

### 5. Rate/amplitude-only baseline (§3.5)
| feature set | CV accuracy |
|---|---|
| amplitude-only (peak amp, area, baseline SD) | 0.704 ± 0.104 |
| timing-only (event time, CoG, peak time) | 0.923 ± 0.073 |
| all features | 0.908 ± 0.071 |

Amplitude alone carries protocol information (peak amplitude differs: medE 13.2 vs medL 20.4,
p=0.0008) — the "timed dynamics" claim partially rides on response magnitude. But timing
features dominate, and the amplitude difference runs the *wrong way* for a trivial
detection-threshold artifact: Late responses are *bigger* yet *later*. The timing effect is not
"earlier because bigger."

### 6. Cross-batch transfer (§3.7) — PASSES
Leave-one-batch-out protocol decoding (logistic regression, all features):

| held-out batch | n | acc |
|---|---|---|
| 230328 | 3 | 0.667 |
| 230329 | 5 | 0.800 |
| 230401 | 13 | 0.769 |
| 230429 | 8 | 1.000 |
| 230501 | 11 | 0.909 |
| 230502 | 14 | 1.000 |
| 230504 | 11 | 0.909 |

Mean 0.865 (chance 0.50); every batch above chance, including the two single-protocol
(Late-only) batches the classifier never saw that protocol-mix from. The protocol distinction
generalizes across year-spanning batches — the opposite of the transfer failures in Audits #3–#5.

### 7. Quality-confound checks
Traces kept per cell (medE 8.0 vs medL 7.0, p=0.43) and baseline noise (p=0.11) do not differ by
protocol; neither correlates with event time (r=-0.03, r=0.17). The effect holds in both cell
types (pyr: 123 vs 536 ms; chr2: 64 vs 380 ms).

### 8. Stimulation-parameter check (§4.4)
The test stimulus (red-alone) is matched across protocols (55/65 cells at 'red/max'); the ISI
difference (10 vs 370 ms) is the training manipulation itself, not a test confound. §4.4 satisfied.

## Caveats (carried, not buried)

- **Non-blinded trace selection.** `keepTrace` is investigator-selected ("selected data or all
  data" per the script); the paper states collection was not blinded. `TraceAll` is absent from
  the `.mat`, so selection bias cannot be tested. This is the one confound avenue this audit
  cannot close.
- **Scope.** Only the Fig-2 evoked-timing claim is testable here; replay and omission-response
  claims need data not in this release.
- **Effect magnitude varies by batch** (-261…-471 ms); the claim is about direction and
  transfer, both of which hold.
- Patch-clamp (n=65 cells), not MEA — the canonical reservoir pipeline was not applicable;
  the battery ran on pipeline-derived features plus the authors' own verified metrics.

## Cross-audit pattern

| Audit | Within-session effect | Cross-session transfer |
|---|---|---|
| Stage C (drug classifier) | 1.000 | fails session nulls |
| #1 Sharf (diazepam signature) | 1.000 | fails session nulls; dose ordering survives |
| #2 DishBrain (HCC learning) | +0.150, p=6.3e-13 | feedback-dependence survives with caveats |
| #4 Shao (pattern decoding) | 0.96/0.90 | training claim fails; transfer collapses |
| #3 FinalSpark (evoked/stim decoding) | evoked 2.7–4.6×, 0.73–0.97/day | transfer fails both datasets |
| #5 Van der Molen (backbone sequences) | stereotypy replicates, narrower | transfer fails; session ID decodable |
| **#6 Liu & Buonomano (timed prediction)** | **150 vs 479 ms, p=1.9e-08** | **PASSES — LOBO 0.865** |

Eight datasets, six labs. The battery does not manufacture skepticism: where the confound is
real it says so (seven of eight), and where the effect survives it — this time — it says that too.

## Reproducibility

- `code/` — `liu_pipeline.py` (verified MATLAB port), `audit06_main.py` (full battery),
  `make_plots.py`
- `results/` — `features.csv` (per-recording features), `inventory.csv`, `within_batch.csv`,
  `stratified_perm.csv`, `shuffle_null.csv`, `rate_baseline.csv`, `cross_batch_transfer.csv`,
  `pipeline_check.npz`, `run_log.txt`
- `plots/` — `event_time_by_batch.png`, `cross_batch_transfer.png`, `feature_sets.png`
- `data/` — **not included in this repo** (kept locally only); download the public
  archive `liu_buonomano_2025.zip` from OSF node `qspnv` ("Early vs Late Evoked
  Timing.zip") to reproduce: https://osf.io/download/k5w63/

*AI disclosure: this audit was performed with AI assistance; all numbers are machine-generated
from the public data and reproduce from the scripts above.*
