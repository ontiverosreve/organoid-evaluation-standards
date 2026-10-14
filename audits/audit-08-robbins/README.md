# Audit #8 — Robbins et al., "Goal-Directed Learning in Cortical Organoids" (Zenodo deposit 17684862)

**Status: Published 2026-10-14.**
Date: 2026-10-14. Auditor: OSO honest-controls program (Audit #8).
Subject: Robbins et al. closed-loop cart-pole deposit, Zenodo record 17684862,
DOI 10.5281/zenodo.17684862, **CC-BY-4.0** — processed HD-MEA + task trajectories
from cortical organoids trained on a cart-pole task via the BrainDance code
(deposited alongside the data). Prediction 003 was registered publicly before
this audit ran (PARTIAL 0.10 / 0.70 / 0.20, screen-informed, disclosed as such).
This audit builds on the first-pass screen (`ROBBINS-SCREEN.md`, 2026-10-14;
its corrected numbers stand and are the pooled-level baseline here).

**Headline verdict: PARTIAL.**
The deposit evidences a genuine within-session improvement under adaptive (RL)
stimulation that survives stratification, split-half replication, and the
deposit's own controls at pooled level, in a majority of organoids. It does not
evidence the across-session component: performance does not accumulate from
session to session under any metric, and each new session starts significantly
below where the previous one ended. Separately, the deposit's headline metric
is not the metric its documentation says it is.

---

## 1. Material and methods

All analyses run on the deposit as delivered (`aggregated_experiment_data.pkl`,
83,234,635 B, MD5 verified against the Zenodo listing in the screen) plus the
deposited BrainDance code. Analysis code: `audit08.py` (seed 20261014;
numpy 2.5.4 / scipy 1.18.1 / pandas 3.0.6); machine-readable results in
`audit08_results.json` and `audit08_per_organoid.csv`; full run log
`audit08_run.log`.

- **Units.** 426 cartpole runs (229 `rl`, 67 `random`, 130 `none`) across 38
  experiments, 16 deduplicated organoid IDs on 15 chips, 2023-11-24 → 2024-05-30.
  (The deposit description says 19 organoids / 432 cycles; the file contains
  16 organoid IDs / 426 runs. Unreconciled — an author question, carried from
  the screen.) 424 runs have ≥8 episodes and enter trend statistics (2 RL runs
  excluded, as in the screen). Reward = per-episode cart-pole balance time (s),
  integer-valued.
- **Per-run statistics.** Spearman ρ(episode order, reward); late−early contrast
  (mean of last episode quartile − first quartile); entry level (first-quartile
  mean) and exit level (last-quartile mean); run median / mean / SD / IQR of
  rewards; and the two candidate performance metrics (§2).
- **Strata.** Organoid × session-day (calendar date of run start): 49 strata
  exist; 46 contain RL runs. Day was used because the screen showed session-day
  identity decodable at 0.699 balanced accuracy within the best-covered
  organoid — pooled analyses alone are not accepted anywhere in this audit.
- **Nulls.** Episode-order permutation within runs (destroys trend, keeps the
  reward distribution); condition-label permutation within strata (keeps each
  run's internal statistics, shuffles which runs count as RL); run-order
  shuffles within organoids. 1,000–10,000 permutations as noted. Where an
  episode-permutation p-value is reported for a rank statistic, observed and
  null values use the same ordinal-rank statistic, so the randomization test
  is exact; descriptive Spearman values are the standard tie-averaged ones.

## 2. Metric audit — `top_decile` is not what the documentation says

The deposit's README defines `top_decile` as the "mean reward of top 10% of
trials" and names it the primary performance metric. Reconstruction from the
deposited reward logs, all 426 runs:

| Candidate formula | Max abs. diff vs stored | Mean abs. diff |
|---|---|---|
| **90th percentile of episode rewards (linear interpolation)** | **0.00** | **0.00** |
| Mean of top 10% of episodes (the documented formula) | 531.9 s | 22.3 s |

The stored `top_decile` **is the 90th percentile of the run's episode
rewards, exactly, on 426 of 426 runs.** The documented top-decile-mean differs
from the stored value on 100% of runs. This is a mislabelled headline metric:
anyone reusing the deposit (or the paper's numbers) as a "mean of the best
decile" is using a single quantile instead.

As a statistic, the 90th percentile is also a near-proxy for within-run reward
spread: across runs, Spearman(p90, reward SD) = **0.957** (RL runs: 0.959) and
Spearman(p90, IQR) = 0.954, versus 0.73 for the run median vs SD. (The
documented top-decile mean is even more spread-coupled, 0.994 vs SD.) Tail
metrics on this data mostly measure how variable a run is, not where its
typical episode sits.

**Does the mislabel change any verdict? No — and that was tested, not assumed**
(§4): the across-session chronology was recomputed under the stored p90, the
documented top-decile mean, the run median, and the run mean. All four agree.

## 3. Within-run learning — stratified, split-half, and against controls

**Pooled (from the screen, stands):** RL runs trend up within a run (median
Spearman ρ = +0.085, median late−early +1.25 s, both p < 0.001 vs episode-order
shuffle); the deposit's `random` and `none` controls show no pooled trend
(p = 0.47–0.96).

**Stratified by organoid × day (this audit, decisive version):** across the 40
strata containing ≥2 RL runs (221 of 229 RL runs covered), the median
stratum-median late−early contrast is **+1.54 s** (episode-permutation
p = 0.002) and the median stratum-median rank trend is **+0.096** (p = 0.001).
The within-run RL improvement is present when organoid and recording day are
held fixed — it is not a pooling artefact of which organoids or days
contributed RL runs.

**Held-out split-half replication:** splitting each run's episodes into
interleaved odd/even halves and estimating the trend independently in each
half (RL runs with ≥16 episodes, n = 225): median ρ = +0.092 (odd) and +0.076
(even); the two half-estimates correlate at 0.747 across runs; the combined
statistic beats the episode-permutation null at p = 0.001. Control halves sit
at ≈ 0 (random: +0.008 / −0.004; none: +0.034 / +0.051). A genuine monotone
trend replicates on disjoint episode subsets; an endpoint or selection
artefact of the full-run statistic would not be expected to.

**RL-specificity within strata — the one soft spot.** In the 31 strata that
contain both RL and control runs (all in `cycled` experiments, the only design
with same-day controls), the RL-minus-control difference in within-run trend
is positive but statistic-dependent: median difference in late−early
**+1.13 s** (label-permutation p = 0.046; RL higher in 61% of strata), but
median difference in Spearman ρ only **+0.019** (p = 0.22; RL higher in 55%).
Split by control type, the ρ contrast is +0.023 vs `random` (21 strata,
p = 0.26) and +0.005 vs `none` (31 strata, p = 0.46). So: pooled, RL runs are
clearly the only condition that trends; at the strictest comparison — same
organoid, same day — the advantage over controls is real on the endpoint
contrast and not established on the rank trend. One design-consistent
explanation is visible in the data: in `cycled` experiments the conditions
rotate on the same culture within a day, so control runs inherit whatever the
day's earlier RL runs did to the culture (e.g. organoid `c20215`'s control
runs themselves trend at median ρ = +0.173, near its RL runs' +0.222). The
deposit cannot separate "RL-specific learning" from "same-day carryover lifts
all conditions"; the within-day design that was meant to control for session
effects also contaminates them.

## 4. Retention — tested four ways, fails four ways

1. **Chronology under every metric** (median per-organoid Spearman of run
   order vs run performance, organoids with ≥5 RL runs; run-order shuffle
   nulls): p90 **−0.127** (p = 0.91), documented top-decile mean **−0.191**
   (p = 0.97), run median **−0.116** (p = 0.88), run mean **−0.111**
   (p = 0.88). Reward SD and IQR trends are likewise negative (−0.164,
   −0.152) — spread does not grow either, so there is no hidden
   variance-driven improvement for a tail metric to have missed.
2. **Chronological split-half** (11 organoids with ≥6 RL runs; second half of
   each organoid's RL runs minus first half): p90 median difference
   **−1.16 s** (Wilcoxon p = 0.76), median-reward difference **0.0 s**
   (p = 0.77), mean difference **−0.34 s** (p = 0.52); 45% of organoids
   positive under each metric — a coin flip.
3. **Carryover gap (the direct test).** For 211 consecutive pairs of RL runs
   within an organoid: median (next run's entry level − this run's exit
   level) = **−2.68 s**, against a median within-run gain (exit − entry) of
   **+1.86 s** — the next session starts below where the previous one began,
   giving back ≈ 1.4× the entire within-run gain (Wilcoxon p = 0.0009). This
   is **not** an overnight effect: for the 181 same-day pairs the gap is
   −2.81 s (p = 0.002). Whatever a session gains is gone by the next run on
   the same day. (The 30 overnight-or-later pairs: −1.53 s, p = 0.21,
   underpowered.) Pooled pairs are dominated by the heavily-run organoids;
   the organoid-level summary (median of per-organoid median gaps +0.27 s,
   56% positive) is noisier — but it does not rescue retention, because tests
   1 and 2, which already use the organoid as the unit, are unanimously null.
4. **Entry vs exit drift.** Entry levels do creep up across sessions (median
   per-organoid ρ = +0.17, shuffle p = 0.041) while exit levels creep down
   (−0.17, p = 0.96). Sessions start marginally higher and end marginally
   lower over the months of training — within-run gains shrink rather than
   compound. A rising entry level with a negative carryover gap is drift or
   maturation, not retained learning: retained learning is precisely the gap
   closing, and the gap is significantly negative.

## 5. Generality across organoids — a majority, not a phenomenon of all

Per-organoid RL trend (episode-permutation p-values per organoid; full table
in `audit08_per_organoid.csv`). Of the 14 organoids with ≥3 RL runs:

- **9 of 14** have a positive median RL ρ; **8 of 14** are individually
  significant on the rank trend (p < 0.05) and **7 of 14** on late−early.
- Of the 10 organoids that have control runs, **6** show RL median ρ above
  their own controls' median ρ.
- Strong positives: `p001237b` (52 RL runs, median ρ +0.149, late−early
  +9.8 s, p < 0.001 on both), `20217` (+0.249, p ≤ 0.001), `p001346`
  (+0.186, p < 0.001), `c20215` (+0.222, ρ p < 0.001), `23186` (+0.224,
  late−early +13.6 s), `22097` (+0.069, p ≤ 0.007), `23133` (+0.118,
  ρ p < 0.001).
- **Well-powered negatives:** `23138` (27 RL runs: median ρ −0.048,
  late−early **−9.2 s**) and `22064` (13 RL runs: median ρ −0.064; only 1 of
  13 runs trends up) trend flat-to-negative with ample data. `22194` (5 runs)
  trends negative in every run. A claim about "cortical organoids" as a class
  has to carry these: the effect is real in a majority of individuals and
  absent or reversed in a substantial, well-sampled minority.

## 6. Verdict — scored by sub-claim (Open Evaluation Standard language)

| Sub-claim | Verdict | Basis |
|---|---|---|
| Within-run improvement under RL stimulation | **SURVIVES** | Pooled trend p < 0.001; survives organoid × day stratification (p = 0.001–0.002, 40 strata / 221 runs); replicates on held-out episode halves (p = 0.001) |
| The improvement is specific to RL (vs the deposit's own controls) | **PARTIAL** | Decisive pooled; within organoid × day strata positive on endpoint contrast (p = 0.046) but not on rank trend (p = 0.22); same-day cycled controls are themselves exposed to the day's RL runs |
| Across-session retention / cumulative learning | **FAILS** | Null-to-negative under all four performance metrics; split-half null; carryover gap −2.68 s vs +1.86 s within-run gain (p = 0.0009), including within the same day |
| Generality across organoids | **PARTIAL** | Positive in 9/14 adequately sampled organoids, individually significant in 8/14; reversed in two of the best-sampled organoids (23138, 22064) |
| Headline metric integrity | **FAILS as documented** | Stored `top_decile` is the 90th percentile, not the documented top-decile mean (426/426 runs); verdicts above are nevertheless metric-independent |

**Headline: PARTIAL.** What the deposit shows is real but narrower than
"goal-directed learning in cortical organoids": organoid cultures, in a
majority of individuals, improve *within* a training session under adaptive
stimulation, and the improvement resets between sessions — including between
sessions on the same day. That is a within-session adaptive phenomenon, not
evidence of learning that persists, accumulates, or is retained across the
repeated training cycles the deposit's framing rests on.

## 7. Limitations of this audit

- Identity decoding was demonstrated at screen level from run-summary
  features, not re-run from unit-level spike trains; the stratified analyses
  in §3 are the control that makes the headline tests immune to that leakage,
  but a unit-level decode would sharpen the leakage estimate.
- The trainer-mechanism question — whether the TetanusTrainer's pattern-value
  updates, vs stimulation per se, drive the within-run RL component — was not
  re-simulated. The deposited `pattern_log`s would support a replay/shuffle
  analysis; it is the natural next step if the authors contest §3.
- Run lengths differ by condition (RL median 100 episodes vs 136 for `none`),
  and stored-metric levels (not trends) are episode-count-sensitive; all
  headline tests here are within-run trend statistics or within-stratum
  contrasts, which are length-robust, but level comparisons in the deposit
  should not be reused without matching episode counts.
- Two counts in the deposit description (19 organoids, 432 cycles) do not
  match the file (16 organoid IDs, 426 runs); nothing in the verdicts turns
  on the difference, but the authors should reconcile it.
- Carryover pairs pool across organoids with very different run counts;
  organoid-unit alternatives are reported alongside wherever they could
  change a conclusion (§4.3).

*Published 2026-10-14 in this repository. Author notification and any
response will be recorded alongside this report.*
