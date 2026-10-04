# Honest-controls audit #2: Kagan et al. 2022, Neuron (DishBrain Pong)

**Target paper:** Kagan, Brenner, Liani et al., *Neuron* 2022,
"In vitro neurons learn and exhibit sentience when embodied in a simulated game-world"
(https://www.cell.com/neuron/abstract/S0896-6273(22)00806-6)

**Published claim audited:** living neuronal cultures (human cortical cells, HCC;
mouse cortical cells, MCC) on high-density MEAs, embodied in closed-loop Pong,
show *learning* — average rally length increases from T1 (first 5 min) to T2
(last 15 min) of a session — not seen in controls (CTL: media-only MEA;
RST: rest, no sensory info; IS: in-silico random paddle). "Apparent learning
within five minutes of real-time gameplay not observed in control conditions."

**IMPORTANT — what this data is:** 2D neural-culture monolayers (human iPSC-derived
and mouse primary cortical cells on planar HD-MEAs), **not** 3D brain organoids.

**Data:** OSF project 5u6qv (https://osf.io/5u6qv/), `Data/in_vitro_cells_sentience_corr.csv`
(md5 `753a94e3fbf6958dd517f7053993e97a`, 26.3 MB; 59,351 rally-level rows),
plus `Code/..._Main_Analysis.ipynb` (the paper's analysis code, used as reference).
License: **CC-BY-NC-ND-4.0** (non-commercial, no-derivatives sharing; this private
audit is analysis, not redistribution).

**Data-quality findings (all handled explicitly):**
1. The released CSV contains ~15% near-duplicate rally rows (same
   chip/date/session/rounded-elapse/hit_count, elapse differing by ~ms). We
   deduplicate on `(chip_id, date, session_num, round(elapse_seconds,1), hit_count)`,
   keeping the first: 59,351 → ~50,900 rows.
2. The CSV's precomputed `half` column is inconsistent with a 300 s split
   (~12% of early-minute rallies labeled half=1 and vice versa). We recompute
   `half = elapse_seconds > 300` ourselves (paper: T1 = 0–5 min, T2 = 6–20 min).
3. **The release is a 720-session superset; the paper's exact 399-session
   Figure-5 sample is not identified in the release** (no session list).
   Session counts in release vs paper: MCC 182 vs 101, CTL 80 vs 80 (exact match),
   HCC 296 vs 138, RST 118 vs 42, IS 48 vs 38. All headline analyses below use the
   full release; Exp A2 repeats on final-design tags as a sensitivity check
   (n=374: MCC 90, CTL 80, HCC 96, RST 60, IS 48) with the same pattern
   (MCC p=0.0014, HCC p=6.6e-07, RST p=0.0023, CTL n.s., IS declines).

**Method:** rally-level data → per-session-half means of hit_count (average rally
length), ace fraction, long-rally fraction → paired t-tests T2 vs T1 within each
group (paper's own statistic), plus: equal-window controls, minute-by-minute
curves, attrition analysis, within-session time-shuffle and group-shuffle
permutation nulls, baseline/consistency checks, STIM-vs-no-feedback contrast,
and a bump-decomposition (is the effect sustained or a minutes-6–8 transient?).
False starts (sessions with only one half) dropped, following the paper's notebook.

---

## VERDICT

**The core biological effect replicates — but it is narrower and messier than presented.**

1. **HCC (human cells): robust, sustained learning signal.** T1→T2 rally-length
   increase survives every control: equal adjacent windows (0–5 vs 6–10 min,
   p≈1e-10), bump-excluded windows (0–5 vs 9–20 min, p≈9e-10), and the most
   consistent per-chip effect (5 chips significant-positive by sign test, none
   significant-negative; MCC by contrast has 4 positive-significant but also
   2 negative chips — a mixed picture).
2. **MCC (mouse cells): the "learning" is a minutes-6–8 transient, not sustained
   improvement.** Full-window T1→T2 is significant (p=0.039), but excluding
   minutes 6–9 it falls to p=0.093. The mouse learning curve is a bump, not a ramp.
3. **The feedback manipulation replicates cleanly:** within HCC, closed-loop
   stimulus sessions improve strongly (n=142, p≈1e-7) while no-feedback sessions
   do not (n=81, p=0.17). This is the strongest evidence the effect is
   feedback-driven rather than a generic time trend.
4. **Caveats that weaken the paper's framing:**
   - **RST (rest, no sensory info) also improves T1→T2** (p=0.02 / long rallies
     p=0.004) on the full release — the paper reported no learning in RST. The
     feedback-specificity claim is therefore weaker than stated. (Note RST sessions
     are ~10 min long, so their "T2" is really minutes 6–10.)
   - **Session durations differ drastically by group and the paper never mentions
     it:** median session length HCC 19.8 min, MCC 19.8 min, IS 19.9 min, CTL 19.7 min
     — but RST 9.9 min, and only 60% of CTL sessions reach minute 10. The "last
     15 min" T2 window is not the same window across groups.
   - **Heavy differential attrition:** CTL drops 80→48 sessions at minute 10;
     RST collapses 117→2 sessions after minute 10. Among CTL survivors, a
     first-5-vs-last-5 comparison spuriously "improves" (p=0.036) — a survivor-bias
     demonstration inside the control data.
   - **No monotonic learning curve:** per-minute linear trends are non-significant
     for every group (all p>0.09); the T1→T2 difference is a level shift
     concentrated at minutes 6–8, not gradual improvement.
   - **HCC starts worse than every other group at T1** (0.68 vs 0.80–0.84),
     so part of its "improvement" is catching up (regression-to-the-mean
     concern); the STIM-vs-NF contrast mitigates but does not eliminate this.
   - The **deep-RL sample-efficiency comparison is not in this paper or dataset**
     (it belongs to the 2024 follow-up, separate OSF release) — out of scope here.

**Bottom line:** unlike the Stage C and Sharf classifier claims (which collapsed
under session-identity controls), DishBrain's within-session, feedback-dependent
improvement — at least for human cultures — survives honest re-analysis. What does
not survive is the tidy narrative: mouse "learning" is a transient bump, rest
controls drift upward too, the time windows aren't comparable across groups, and
the public release doesn't allow exact replication of the paper's sample. The
paper's core claim (biological cultures improve with closed-loop feedback;
no-feedback does not) stands; the strength and specificity claimed for it do not.

## Experiments

- `experiment_inventory.csv` — sessions/chips/rallies per group (release vs paper n).
- `experiment_inventory_durations.csv` — session-length distribution per group.
- `experiment_A_replication.csv` — paper's T1 (0–5) vs T2 (6–20) paired tests, full release.
- `experiment_A2_final_design.csv` — same, restricted to final-design tags (sensitivity).
- `experiment_B_equal_windows.csv` — first-5 vs last-5, and first-5 vs min-6–10.
- `experiment_C_minute_curves.csv` — per-group per-minute mean rally length + rally rate.
- `experiment_C_trends.csv` — per-group linear trend over minutes (all n.s.).
- `experiment_D1_timeshuffle.csv` — within-session time-shuffle permutation null
  (2,000 reps): HCC p=0.000, MCC p=0.042, RST p=0.019, CTL p=0.185 — the
  T1→T2 differences exceed arbitrary within-session splits.
- `experiment_D2_groupshuffle.csv` — group-label shuffle on the interaction
  contrast mean(t_MCC,t_HCC) − mean(t_CTL,t_RST,t_IS) (5,000 reps): observed 5.44,
  p<0.0002; bio vs (CTL,IS) only: observed 6.93, p<0.0002. The improvement is
  group-specific, not a grand-mean artifact. (An earlier min(t_MCC,t_HCC)
  statistic was discarded as uninformative — it is driven by the grand mean.)
- `experiment_E_baselines.csv` — T1 group means (HCC starts lowest).
- `experiment_E_chip_consistency.csv` — per-chip sign tests (≥4 sessions).
- `experiment_F_feedback.csv` — STIM-like vs no-feedback within HCC.
- `experiment_G_bump_decomposition.csv` — T1 vs min-9–20 and T1 vs min-6–9.

## Key numbers (paired T1→T2, hit_count, full release unless noted)

| group | n | T1 mean | T2 mean | Δ | t | p |
|---|---|---|---|---|---|---|
| MCC | 179 | 0.802 | 0.873 | +0.071 | +2.08 | 0.039 |
| CTL | 80 | 0.809 | 0.757 | −0.052 | −1.37 | 0.18 |
| HCC | 296 | 0.681 | 0.831 | +0.150 | +7.53 | 6.3e-13 |
| RST | 117 | 0.698 | 0.758 | +0.060 | +2.35 | 0.020 |
| IS | 48 | 0.845 | 0.748 | −0.096 | −2.89 | 0.0059 |

Bump decomposition (does the effect survive excluding minutes 6–9?):

| group | T1 vs min 9–20 Δ | p | T1 vs min 6–9 Δ | p |
|---|---|---|---|---|
| MCC | +0.063 | 0.093 (n.s.) | +0.307 | 0.063 |
| HCC | +0.118 | 8.7e-10 | +0.511 | 2.1e-08 |
| RST | +0.227 | 5.0e-04 | +0.083 | 0.013 |
| CTL | −0.005 | 0.93 | −0.097 | 0.045 (decline) |
| IS | −0.097 | 0.0045 (decline) | −0.075 | 0.12 |

Feedback contrast within HCC: STIM-like n=142, Δ=+0.142, p=1.3e-07; no-feedback
n=81, Δ=−0.051, p=0.17.

## Limitations
- Exact replication of the paper's 399-session sample is impossible from the
  public release (superset; no session list). Only CTL matches exactly.
- The feedback pkls' `control` column meaning is undocumented and no analysis code
  maps tags to the paper's STIM/SIL/NF conditions; Exp F uses tag-name heuristics
  on the main CSV (silent condition absent from release).
- `long_rally`/`ace` replicate the same pattern as hit_count (see CSVs); not all
  discussed above.
- No access to the deep-RL comparison data (2024 follow-up, separate release).
- Analysis is private; nothing published, no authors contacted.

## Files
- `data/` — downloaded OSF files (hashes verified against OSF API listing)
- `code/run_audit.py`, `code/run_part2.py`, `code/run_expF.py`
- `results/` — CSVs listed above
- `plots/` — `learning_curves.png` (per-minute rally length + rally rate),
  `t1_vs_t2.png` (T1 vs T2 bars by group)
