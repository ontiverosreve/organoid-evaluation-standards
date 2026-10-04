# Honest-controls audit #4: Shao et al. 2025, PLOS Computational Biology (cultured-network pattern recognition)

**Target paper:** Shao W-W, Shao Q, Xu H-H, Qiao G-J, Wang R-X, Ma Z-Y, et al. (2025)
"Repetitive training enhances the pattern recognition capability of cultured neural networks."
*PLOS Computational Biology* 21(4): e1013043. https://doi.org/10.1371/journal.pcbi.1013043

**Published claims audited:**
1. Cultured networks classify distinct electrical stimulation patterns from their
   evoked responses: two-pattern accuracy 93% after day 1 → 98.2% after day 3;
   six-pattern accuracy ~71% after day 1 → 82.5% after day 3
   (Fig 4d, mean ± s.e.m., n = 10, *p < 0.05).
2. "Repeated training increased recognition accuracy for each stimulation pattern"
   (Abstract) — accuracy improves with training days.

**IMPORTANT — what this data is:** primary cortical neurons from E18 **rat**
embryos (Wistar; Methods §4.2), plated as **2D monolayers on planar 60-electrode
MEAs** (MEA2100-Mini, 25 kHz). These are **not 3D brain organoids** — the same
honesty note as Audit #2 (DishBrain), whose "organoids" were also 2D monolayers.

**Data:** https://github.com/YunDid/Pattern-recognition (`pattern_recognition/`;
cloned 2026-10-04; no license file in repo — paper is CC-BY). Layout:
`{two_pattern,six_pattern}/culture_1..10/day_1..3/{EVENT,SSD}/` (2.0 GB).
- `SSD/<pattern>.mat`: spike-sorted spike times (seconds) per unit; variables
  `..._ID_<electrode>_..._cluster<n>` (sorted) plus `_unsorted`.
- `EVENT/<pattern>_EVENT.mat`: one variable with the 400 stimulation pulse times
  (1 Hz, biphasic ±500 mV, 200 µs; 40 pulses × 10 rounds per pattern per day).
- two_pattern: 4 patterns/day (two L-family, two X-family, e.g. 2L/8L vs 5X/10X);
  six_pattern: 6 patterns/day (L/C/X/S families, e.g. 10L/12C/2L/4X/6L/8S).
  File naming is inconsistent across cultures (dashes, `spikeSortingData*`,
  `*_train_*`, `first_learning` variants); SSD↔EVENT pairs were matched by
  normalized stem, then by (pattern-number, family-letter) key, each EVENT file
  used at most once. See `results/file_pairs.csv` for the full pairing log
  (286 rows; 17 SSD files had no EVENT match and were excluded).

**Data-quality findings (all handled explicitly):**
1. **Stray files from other sessions.** `two_pattern/culture_4/day_1/SSD/` contains
   4 extra files (`10-X2.mat`, `3-L1.mat`, `5-X2.mat`, `8-L.mat`) whose variable
   names carry a different session/position tag and which have no EVENT files.
   A naive key-based match would have paired them to the wrong stim times and
   double-counted trials. They are excluded (two-pass exact-then-key matching).
2. **Incomplete culture-days.** Several culture-days lack EVENT files for some
   patterns (e.g. two_pattern c5d2: 1/4, c7d3: 2/4; six_pattern c7/c8: no day 3,
   c9: no day 2). Only SSD↔EVENT pairs are analyzed; per-culture-day pattern
   counts are recorded in every result row. Headline paired tests use only
   culture-days with full pattern sets.
3. **Suspected filename typo.** two_pattern c8d2: SSD `5X.mat` paired to EVENT
   `5L_EVENT.mat` (number-only match, flagged in `file_pairs.csv`). Kept, flagged.
4. **six_pattern c3d3** contains a 7th family letter (`14-Y1`); treated as a 6th
   pattern class like the others.
5. **six_pattern c5d2/c5d3, c6d2/c6d3**: one pattern has 390 (not 400) pulses.
   Stratified CV handles the mild imbalance.
6. **Text/code classifier mismatch.** The paper text says logistic regression;
   the repo's `Classification Accuracy.py` uses `svm.SVC(kernel='rbf')`. Both
   were run (see below) — this mismatch explains the paper's headline numbers.

**Method:** per-trial features = per-unit spike counts in post-stimulus windows,
built from the 400 stim pulses per pattern file (sorted clusters only):
- `paper`: [10, 50] ms — **the paper's own window** (their `EvokedResponseMatrix.py`
  counts spikes with `stim+0.01 <= t <= stim+0.05`, i.e. they already blank the
  first 10 ms);
- `noblank`: [0, 50] ms; `artifact`: [0, 2] ms (artifact-only control);
  `early`: [2, 50] ms; `wide`: [10, 200] ms; `vwide`: [10, 500] ms.
- Classifier (main): L2 logistic regression (C=1.0, lbfgs) on standardized
  features — the paper's stated method. Sensitivity: the repo's RBF-SVM
  (`C=1.0, gamma='auto'`) with their chronological block scheme
  (5 blocks of 80 trials/pattern; train first 60, test last 20).
- CV: stratified 5-fold (main) + the block scheme as sensitivity.
- Session-identity nulls: (C1) culture-ID decoding pooled across days
  (10-way, electrode-aggregated features); (C2) day-ID decoding within culture;
  (C3) cross-day transfer train-day1→test-day3 and reverse on shared units,
  patterns aligned by (pattern-number, family-letter) key;
  (C4) 20 label-shuffles per culture-day as empirical chance.
- Training claim: paired t-test of day-3 minus day-1 accuracy across cultures,
  plus a check whether accuracy gains track mean firing-rate gains
  (excitability confound).

---

## VERDICT

**The pattern-decoding effect is real, but the "training improves it" claim does
not survive honest re-analysis.**

1. **Patterns are genuinely decodable from evoked responses.** In the paper's own
   [10, 50] ms window, logistic regression reaches 0.96 (two_pattern 4-way,
   chance 0.25) and 0.90 (six_pattern 6-way, chance 0.167), far above
   permutation nulls (0.25/0.167). Binary L-vs-X decoding is ~0.99. The cultures
   do respond differentially to stimulation patterns — this part replicates.
2. **No significant day-1 → day-3 improvement in either experiment.**
   - two_pattern (logreg, 5-fold): n=5 full-set cultures, day1 0.973 → day3 0.944,
     Δ=−0.030, paired t=−1.13, p=0.32. (Lenient n=9 incl. partial days:
     Δ=−0.007, p=0.71.) Day-1 accuracy is already at ceiling — the paper itself
     notes "high accuracy achieved on the first day."
   - six_pattern (logreg, 5-fold): n=7, day1 0.896 → day3 0.913, Δ=+0.018,
     paired t=0.76, p=0.48.
   - RBF-SVM replication (their code's classifier): two_pattern n=6,
     Δ=−0.052, p=0.22; six_pattern n=8, Δ=+0.084, p=0.23. Not significant either.
   - Accuracy gains do not track excitability gains (corr(Δrate, Δacc) = −0.09 /
     +0.26, both n.s.) — but there are no gains to explain.
3. **The paper's headline six-pattern numbers come from the weaker pipeline.**
   With logistic regression (their stated method), six-pattern day-1 accuracy is
   **0.90, not ~0.71** — the "headroom" for training gains mostly disappears.
   Their 71%→82.5% is reproduced only with RBF-SVM (day1 0.68 → day3 0.76 in my
   rebuild), and even there the paired gain is not significant (p=0.23).
4. **The stimulation artifact is informative — the 10 ms blank is load-bearing.**
   The [0, 2] ms artifact-only window decodes patterns at **0.985** (both
   experiments): the raw stimulation artifact carries near-perfect pattern
   identity (different stimulation electrodes → different artifact waveforms,
   picked up as "spikes"). The paper blanks the first 10 ms, and decoding in
   [10, 50] ms remains strong (0.96/0.90), so the claim does not *ride* the
   artifact — but the most informative milliseconds are the excluded ones, and
   any analysis that failed to blank them would be decoding the stimulator,
   not the network.
5. **Session identity dominates the features.** Culture-ID decodes at 0.94/0.90
   (chance 0.10); day-ID within culture at 0.94 (chance 0.33). Cross-day transfer
   of pattern decoders is weak: two_pattern 0.47/0.47, six_pattern 0.29/0.27 —
   far below within-day accuracy, and one culture (c1) transfers *below* chance
   (0.205), i.e. day-3 responses actively mislead a day-1 decoder. Pattern
   representations are not stable across days, which cuts against interpreting
   any day-to-day accuracy change as learning.

**Bottom line:** unlike the Stage C and Sharf classifier claims (which collapsed
under session-identity controls), the *decoding* here is real — but the paper's
central claim, that repeated training *improves* recognition, does not replicate:
accuracy is at ceiling from day 1 (two-pattern) or shows no significant gain
(six-pattern), the headline numbers depend on an undisclosed classifier switch
(RBF-SVM in code vs logistic regression in text), and the features are dominated
by culture/day identity with poor cross-day transfer.

**Anomalies flagged for manual review (not silently dropped):**
- (a) Fig 4d reports **n=10** for the six-pattern day-1→day-3 comparison, but at
  most **8** cultures have complete 6-pattern data on both days (c7/c8 have no
  day 3). It is unclear how n=10 was reached.
- (b) Text/code classifier mismatch: paper text says logistic regression; the
  released `Classification Accuracy.py` uses RBF-SVM. The headline 71%→82.5%
  figures are reproducible only via the SVM path.
- (c) two_pattern/culture_4/day_1's 4 stray dashed files (different session tag,
  no EVENT files) — excluded; see finding 1.
- (d) two_pattern c8d2's `5X.mat`→`5L_EVENT.mat` pairing (probable filename typo).

## Experiments

- `file_pairs.csv` — all 286 SSD↔EVENT pairings with flags.
- `decoding_pattern.csv` — per-(experiment, culture, day, window, scheme)
  pattern-decoding accuracy (N-way + binary L-vs-X rows).
- `decoding_culture_id.csv` — C1: 10-way culture-ID decoding (0.9398 / 0.8993).
- `decoding_day_id.csv` — C2: day-ID decoding per culture (mean 0.937 / 0.935).
- `cross_day_transfer.csv` — C3 (patid-aligned; superseded).
- `cross_day_transfer_v2.csv` — C3 redone with (pattern-number, family-letter)
  key alignment: two_pattern d1→d3 0.469 / d3→d1 0.465 (n=9);
  six_pattern 0.293 / 0.265 (n=8).
- `permutation_nulls.csv` — C4: 20 shuffles per culture-day (≈chance everywhere).
- `training_claim.csv` — per-culture-day accuracy + mean firing rate.
- `svm_replication.csv` — RBF-SVM + block scheme per culture/day (the paper's
  code path): two_pattern day1 0.944 → day3 0.892 (p=0.22); six_pattern
  day1 0.678 → day3 0.761 (p=0.23).

## Key numbers

Pattern decoding, paper window [10,50] ms, logistic regression, stratified 5-fold:

| experiment | day 1 mean | day 3 mean | Δ | paired p | n |
|---|---|---|---|---|---|
| two_pattern (4-way) | 0.973 | 0.944 | −0.030 | 0.32 | 5 |
| six_pattern (6-way) | 0.896 | 0.913 | +0.018 | 0.48 | 7 |

(Paper reported: two-pattern 0.93→0.982; six-pattern ~0.71→0.825, n=10, p<0.05.)

Window comparison (mean accuracy, 5-fold CV, full pattern sets):

| window | two_pattern | six_pattern |
|---|---|---|
| [0,2] ms (artifact-only) | 0.985 | 0.985 |
| [0,50] ms (no blank) | 0.999 | 0.999 |
| [2,50] ms | 0.998 | 0.998 |
| [10,50] ms (paper) | 0.960 | 0.902 |
| [10,200] ms | 0.964 | 0.927 |
| [10,500] ms | 0.968 | 0.931 |

Session-identity nulls:

| test | two_pattern | six_pattern | chance |
|---|---|---|---|
| culture-ID (C1) | 0.940 | 0.899 | 0.10 |
| day-ID within culture (C2, mean) | 0.937 | 0.935 | 0.33 |
| cross-day transfer d1→d3 (C3v2) | 0.469 | 0.293 | 0.25/0.167 |
| cross-day transfer d3→d1 (C3v2) | 0.465 | 0.265 | 0.25/0.167 |

## Limitations

- Exact replication of the paper's trial features is impossible: the repo's
  classification script reads precomputed CSVs (`StimData/`) that are not in the
  release, so features were rebuilt from the raw .mat files following their
  `EvokedResponseMatrix.py` window ([10, 50] ms). The RBF-SVM replication
  reproduces their *code* path; residual differences (their day-3 82.5% vs my
  76.1%) likely reflect the unavailable CSV features.
- Spike sorting taken as given (NeuroExplorer/SpyKING CIRCUS per Methods);
  sort quality not re-validated. Unsorted spikes excluded from features.
- Cross-day unit matching uses (electrode ID, cluster index); ~70–80% of units
  match across days within a culture.
- six_pattern cultures 7/8 lack day 3 and culture 9 lacks day 2, capping the
  paired day-1→day-3 test at n=8 (paper reports n=10 — see anomaly (a)).
- Analysis is private; nothing published, no authors contacted.

## Files

- `code/build_features.py` — SSD↔EVENT pairing, per-trial feature building, npz cache
- `code/run_audit.py` — main decoding analyses + session-identity nulls
- `code/fix_xfer.py` — C3 redo with key-based pattern alignment
- `code/svm_replication.py` — the paper's RBF-SVM code path
- `code/make_plots.py` — figures
- `cache/` — per-(experiment, culture, day) feature matrices (not for sharing)
- `results/` — CSVs listed above; `run_log.txt`
- `plots/` — `day1_vs_day3.svg`, `window_comparison.svg`, `cross_day_transfer.svg`
