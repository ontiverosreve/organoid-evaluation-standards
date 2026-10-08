# Audit #3 — FinalSpark whole-life MEA datasets (FS369, FS437)

**Status:** complete 2026-10-07 · **CONFIDENTIAL inputs** — raw FinalSpark data never leaves
`data/` and is not part of any public release. This report, the analysis scripts,
derived result CSVs, and plots are aggregates only.

## What was audited

Two whole-life multielectrode-array recordings of living neurosphere cultures,
provided by FinalSpark (Ewelina Kurtys, 2026-10-06) as offline datasets under
confidentiality terms (cite FinalSpark, as-is, no redistribution):

| | FS369 | FS437 |
|---|---|---|
| Culture | NS Batch 3, 12 neurospheres, 4×8 MEA (32 ch) | Same batch, **different physical MEA/culture** |
| Span | 2025-02-19 → 2025-03-12 (21.2 days, continuous) | 2025-06-05 → 2025-06-11 (5.7 days) |
| Spike events | 94,154,693 | 2,643,769 |
| Stimulations | 130,113 (7 session days, 60 protocols) | 802,657 (daily protocol sweeps + 1 dense day, 54 protocols) |
| Death | 2025-03-12 "Almost no activity" (gradual die-off) | 2025-06-11 "No activity" (abrupt, +1,050 test stims into the silent culture) |

There is no published paper claim attached to this data; the audit asks what
structure in it survives honest controls. Methods: canonical Stage C pipeline
(150-node leaky reservoir, spectral radius 0.9, sparsity 0.1, alphas
[0.005, 0.01, 0.02], seeds [7, 11, 22], quality-weighted ridge readout λ=1e-3)
plus logistic-regression baselines, per the Open Evaluation Standard v0.3
(session-identity nulls, shuffle nulls, chronological controls, rate-only
controls, stimulation-artifact blanking, §3.7 cross-session transfer).

## Verdict

**Evoked responses are real; nothing transfers.** In both cultures, electrical
stimulation reliably evokes short-latency spiking that survives every null we
applied — but no decoder learns a stimulation signature that survives a change
of day, and recording-session identity is decodable from background activity
alone. This is the fifth independent confirmation of the series' central
finding: within-session effects hold, cross-session generalization does not.

## Findings

### 1. Stimulation-amplitude decoding works within a day, fails across days (FS369)

Decoding amplitude {0.8, 1.0, 1.5} µA from the [0,50] ms post-stimulus window
(n=120 whole-array volley trials; chance 1/3):

| Analysis | Reservoir | LogReg |
|---|---|---|
| Pooled 5-fold CV | 0.536 ± 0.059 | 0.725 ± 0.101 |
| Per-day CV | 0.563–0.856 | 0.867–0.967 |
| Artifact-blanked [10,50] ms | 0.509 | 0.700 |
| Pre-stim baseline null | 0.367 (chance) | 0.325 (chance) |
| 20-rep label shuffle | true 0.558 vs null 0.337 ± 0.065 | — |
| **Cross-day 02-20 → 02-28** | **0.322 (chance)** | 0.533 |
| **Cross-day 02-28 → 02-20** | **0.330 (chance)** | 0.367 (chance) |

The decoder survives 0–10 ms artifact blanking (stimulation artifact contaminates
only 0.004% of events — the hardware blanking works), beats the shuffle null,
and sits at chance on pre-stim windows. But the canonical reservoir fails §3.7
transfer in **both** directions, and even logistic regression transfers weakly
one way only. The information is distributed across the array (per-electrode
decoding 0.339 vs all-32 0.536/0.725). Likely mechanism: the decoder leans on
evoked burst magnitude, which *decreases* with amplitude (87.9 → 79.7 → 66.0
events/trial) and drifts across days — a day-drifting scalar cannot transfer.

### 2. Lifespan session-identity: the Stage C mirror at 21-day scale (FS369)

From 5-minute firing-rate blocks (6,092 blocks × 32 electrodes):

| Analysis | Accuracy | Chance |
|---|---|---|
| Epoch (early/mid/late) | **0.9867** | 1/3 |
| Calendar-day ID | **0.9215** | 1/22 |
| Epoch, collapsing electrodes only (64–71) | 0.8981 | 1/3 |
| Epoch, persistent electrodes only (72–87) | 0.8012 | 1/3 |

Session separates near-perfectly — the confound demonstration, not a discovery.
The ablation refines it: die-off electrodes carry more signal, but surviving
electrodes still decode epoch at 0.80, so lifespan signal is die-off *plus*
genuine rate change, not die-off alone.

### 3. No detectable medium-change effect (FS369)

One timestamped medium change (2025-02-28T14:01:34) sits 1.5 h before the largest
stim session. Volley-level evoked ratios collapse post-change (median 125.2x →
0.94x, p=1.3e-14) — but the decisive within-day control kills the attribution:
block 1 of the 02-28 session, recorded **2.4 h after the medium change** at 10 s
volley spacing, shows the same ~100x responses as pre-change days, while block 2
at 1 Hz drive shows 0.94x (p=3.9e-08). The collapse is stimulation-rate
saturation, not the medium. No within-session habituation (p=0.47).

### 4. Evoked responses survive the honest null; protocol identity barely decodes (FS437)

| Protocol | Observed ratio | Stim-shuffled null (20 reps) |
|---|---|---|
| P1 (3 µA) | **4.17×** | 1.00 ± 0.03 |
| P2 (1 µA) | **2.73×** | 1.01 ± 0.05 |
| P3 (5 µA) | **4.56×** | 1.00 ± 0.05 |

The null permutes stim times preserving per-electrode ISI distributions — the
dense regime (median ISI 0.2 s) makes naive pre-stim baselines stim-contaminated,
so this is the real test, and the responses survive it decisively. Dose ordering
holds (high-amplitude ≈ 4.2–4.6×, low ≈ 2.7×). But protocol identity decodes at
only 0.43 vs 0.33 chance (survives artifact blanking) — a strong evoked
response is not a strong protocol fingerprint.

### 5. Cross-day transfer fails; session identity wins (FS437, §3.7)

- Forward (train Jun 7–8 → test Jun 9): P2 recall **0.16 vs shuffled null 0.42** — worse than null.
- Reverse "1.00" P2 recall is a classifier-bias artifact (in-distribution P1 recall only 0.28).
- Day-ID decodes from **background activity alone at 0.51** vs 0.33 chance — protocol decoding (0.43) does not beat the session confound.
- The canonical reservoir shows a **zero-state collapse mode** on sparse trials (77–78% of test trials → near-zero states, argmax degenerates) — a methodological caveat for the pipeline on sparse per-stim data, recorded here.

### 6. Dead-culture negative control passes (FS437)

1,050 stimulations delivered 2025-06-11 into the silent culture (after the last
event at 06:38): **0 events** in any post-stim or baseline window; the
live-trained decoder sits at chance confidence (0.347 vs 0.333). The pipeline
does not hallucinate responses where none exist.

## Cross-audit pattern

| Audit | Within-session effect | Cross-session transfer |
|---|---|---|
| Stage C (drug classifier) | 1.000 | fails session nulls |
| #1 Sharf (diazepam signature) | 1.000 | fails session nulls; dose ordering survives |
| #2 DishBrain (HCC learning) | +0.150, p=6.3e-13 | feedback-dependence survives with caveats |
| #4 Shao (pattern decoding) | 0.96/0.90 | training claim fails; transfer collapses |
| **#3 FinalSpark (this audit)** | **evoked 2.7–4.6×, amp. decoding 0.73–0.97/day** | **transfer fails both datasets** |

Five datasets, four labs, one confound: session identity.

## Caveats

- FS369's 60 GB raw file was not downloaded (disk); all analyses use the package
  (events/stimulations/incubator/metadata), which the provider documents as the
  standalone analysis unit. Artifact conclusions rest on event timestamps, not
  raw waveforms.
- 16.7% of FS369 event timestamps sit on a 100 µs lattice — sub-millisecond
  analyses should treat with care; the 2/10 ms artifact windows are unaffected.
- Voltage clips symmetrically at ±6,390 µV (ADC saturation) in both datasets.
- FS437's dense stimulation (1.46 stims/s) means no stim-free baseline exists;
  all "baseline" comparisons there are relative, and the stim-shuffled null is
  the load-bearing control.
- 20-rep nulls cap empirical p at 0.0476; observed effects sit far outside null
  ranges throughout.

## Reproducibility

- `code/` — analysis scripts (incl. `audit_lib.py`, the canonical pipeline
  copied unchanged from Audit #1) and exploration scripts.
- `results/` — derived CSVs (decoding accuracies, transfer tests, nulls,
  evoked ratios, dead-culture control) plus `run_log.txt`.
- `plots/` — summary figures.
- `data/` — **confidential inputs, never published**: download provenance and
  verification in `STAGING.md`. Raw FinalSpark data and share links are excluded
  from any public package.

*AI disclosure: this audit was performed with AI assistance; all numbers are
machine-generated from the provider's data and reproduce from the scripts above.*
