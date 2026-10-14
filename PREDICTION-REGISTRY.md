# Prediction Registry — Honest-Controls Audit Program

**Registered and posted publicly 2026-10-14.** Predictions below were written before any data from either target paper was seen, and are never edited after posting (registry rule 2).
**Author:** Reve Alan Ontiveros, OSO (independent researcher, Taylorsville, CA)

---

## What this ledger is

Before I audit a claim, I write down what I expect the audit to find — in
public, with a timestamp, before I have seen the data. Then the audit runs,
and the prediction gets scored, right or wrong, in the same public place.

Post-hoc audits can always be accused of motivated reasoning: I knew the
answer before I chose the controls. A prediction made before the data
arrives cannot be accused of that. This registry is the proof, one way or
the other.

**Rules of the ledger**

1. Predictions are posted before any data from the target paper is
   downloaded, requested-and-received, or analyzed by me.
2. Predictions are never edited after posting. Corrections and updates go
   in a new dated entry below the original.
3. Each prediction states a probability for each possible audit verdict —
   **survives**, **partial**, **fails** — using the verdict definitions in
   the Open Evaluation Standard (github.com/ontiverosreve/
   organoid-evaluation-standards). The three probabilities sum to 1.
4. When the audit completes, the prediction is scored with a **Brier
   score**: for each verdict, take (predicted probability − outcome)²,
   where the outcome is 1 for the verdict that occurred and 0 for the
   others, and sum. Lower is better. A confident, correct prediction
   scores near 0. A confident, wrong one scores near 2. Guessing
   "partial" at 1/3–1/3–1/3 scores 0.67 — the scoreboard punishes
   hiding behind the middle.
5. A running mean Brier score across all scored predictions is kept at
   the bottom of this file. Survivals I predicted are reported with the
   same prominence as failures I predicted.
6. If data never becomes available, the prediction stays open and is
   marked **unscored — no data**, not quietly deleted.

**Verdicts (from the Open Evaluation Standard, v0.4)**
- **Survives:** the headline claim passes the required null controls
  (session-identity, label-shuffle, chronological split, cross-session
  transfer where applicable) substantially as published.
- **Partial:** a real effect is present, but the published claim is
  narrower, weaker, or more conditional than stated — or the effect
  survives some controls and fails others.
- **Fails:** the headline claim is explained by a null control (session
  identity, artifact, leakage) and is withdrawn.

---

## Prediction 001 — Iannello et al. 2025

**Registered:** 2026-10-14. **Status: OPEN — no data seen.**
Data status at registration: code public (github.com/ludovico-iannello/
bio_reservoir_computing); full recordings not public. A data request was sent to the authors on
2026-10-08; it has not been answered, and no recordings have been
received or analyzed.

**The claim.** Iannello L. et al., "From Neurons to Computation:
Biological Reservoir Computing for Pattern Recognition," arXiv:2505.03510
(2025); published version in Proc. IEEE/CVF ICCV 2025 Workshop,
pp. 4771–4780. A pool of stem-cell-derived cultured neurons on a
4096-electrode HD-MEA works as a biological reservoir: stimulation
patterns (positional codes, oriented bars, digit images) are delivered
through a subset of electrodes, evoked spiking is recorded on the rest,
and a single-layer perceptron read out in the biological feature space
classifies the inputs.

**Predicted verdict: PARTIAL** — Survives 0.15 / Partial 0.60 / Fails 0.25

**Reasoning.**
- The claim is a *within-session decoding* claim: a perceptron trained
  and tested on evoked responses. Within-session decoding of stimulation
  patterns has held up in every dataset I have audited so far (Audit #4,
  Shao et al.: within-day decoding 0.96 against shuffle nulls). I expect
  the digit/pattern classification to be genuinely above its shuffle
  null. That is why "fails" is not my modal prediction.
- But three of my eight audited datasets showed the same shadow next to
  a real within-session effect: session identity decodable at 0.89–0.99
  from the same features, and a readout that does not cross a session
  boundary. A perceptron read out per session, on HD-MEA data with
  strong electrode- and day-level fingerprints, is exactly the geometry
  where that happens. If the authors' headline framing implies a
  general reservoir — one readout, many days, many cultures — I predict
  cross-session transfer lands near chance and the claim survives only
  in its per-session form.
- The "fails" weight (0.25) covers the stimulation-artifact route
  (§3.6 of the standard): with input patterns delivered electrically
  and a linear readout, artifact-blanked accuracy is the load-bearing
  control, and in Audit #4 the raw artifact alone decoded patterns at
  0.985. If the perceptron is substantially reading stimulation
  artifacts rather than evoked network responses, the claim fails, not
  trims.

**What would change my prediction.**
- Release of recordings with session/culture IDs showing the perceptron
  was trained on some sessions and tested on held-out sessions at
  above-null accuracy → move toward **survives**.
- A published artifact-blanking control in the paper or supplement
  that I have missed → move toward **survives**.
- Evidence that all classification comes from a single session with no
  replication → move toward **fails** (an unreplicated within-session
  result cannot carry a substrate claim).

**Scoring.** Scored when Audit #7a (or its successor) is published, or
marked unscored if no data is obtained within 18 months of registration.

---

## Prediction 002 — Kim et al. 2026 (MiV / Gazzola lab)

**Registered:** 2026-10-14. **Status: OPEN — no data seen.**
Data status at registration: code public (github.com/GazzolaLab/
MiV-OS); per the MiV-OS README the accompanying dataset is released
upon publication. A data request was sent to the corresponding author
(Prof. M. Gazzola, UIUC) on 2026-10-08; no data has been received or
analyzed.

**The claim.** Kim SH, Dou Z, Upadhyay G, et al., "Computing with
Living Neurons: Chaos-Controlled Reservoir Computing with Knowledge
Transplant," arXiv:2604.02552 (2026). Across 500+ identically prepared
mouse ESC-derived motoneuron cultures on 128-electrode MEAs (MiV
platform, optical input): (1) chaos-control stabilization improves
pattern-classification accuracy and model longevity ~300% over naive
reservoir computing; and (2) "Knowledge Transplant" — a readout map
moved from an expert culture to an attractor-equivalent student culture
— cuts training time ~80% while improving performance.

**Predicted verdict: PARTIAL** — Survives 0.20 / Partial 0.55 / Fails 0.25

**Reasoning.**
- Sub-claim (1), chaos control, is a *within-culture* comparison against
  the same culture's naive baseline. Within-session and within-culture
  effects are the category that has most often survived my controls
  (the DishBrain feedback contrast in Audit #2; evoked responses in
  Audit #3). I expect a real stabilization effect, though possibly
  smaller than 300% once session-matched nulls are applied — hence
  partial rather than survives.
- Sub-claim (2) is the boldest cross-substrate transfer claim in the
  current literature: a learned readout crossing from one living
  culture to another. That is §3.7 territory, and cross-boundary
  transfer has failed in 7 of my 8 audited datasets. The authors'
  defense is built in: transplant only between "attractor-equivalent"
  cultures, typed A–D by dynamics. That selection step is exactly why I
  do not predict an outright fail — matched-type transfer may genuinely
  beat chance. But it also caps the claim: if transplant works only
  inside a dynamical type, after searching 500+ cultures for a match,
  the headline ("knowledge transplant") is narrower than stated, and
  the honest verdict is partial.
- The 500+ culture structure is the best repeated-session design I have
  seen in this field. If anyone's transfer claim can survive, it is one
  built on that much replication. That is worth the 0.20 on survives —
  double what I would give a single-culture transfer claim.

**What would change my prediction.**
- A published transfer matrix (expert × student, by dynamical type)
  showing above-null transplant accuracy across types, not only within
  → move toward **survives**.
- Evidence that "attractor-equivalent" matching was done on the same
  recordings used to evaluate the transplant (matching on test data)
  → move toward **fails** for sub-claim (2).
- Release of the dataset with per-culture IDs on publication, as the
  README promises — the audit then runs §3.1 and §3.7 as designed and
  the prediction is scored on the result, whichever way it lands.

**Scoring.** Scored when the audit is published after dataset release,
or marked unscored if no data is obtained within 18 months of
registration.

---

## Prediction 003 — Robbins et al. 2025 (closed-loop organoid learning deposit)

**Registered:** 2026-10-14. **Status: OPEN — first-pass screen seen (disclosed below); full audit not run.**
Data status at registration: data and code are public (Zenodo record
17684862, CC-BY-4.0): processed HD-MEA features and task trajectories
from 19 cortical organoids, 432 training cycles on a cart-pole task.
**Disclosure against Rule 1:** before registering, I ran a first-pass
*screen* of the deposit (within-run learning curves under an
order-shuffle null, the deposit's own random/none control arms, and
identity decodability). The screen is not the audit: the audit's
decisive analyses — organoid × day stratified controls, an audit of
the deposit's `top_decile` performance metric, and retention tested
under the Standard's chronological-split rules — have not run. I am
registering with the screen disclosed rather than pretending it did
not happen; score this entry knowing it is screen-informed.

**The claim.** Cortical organoids in a closed loop learn a
goal-directed task: reinforcement-style stimulation feedback improves
cart-pole balancing within training, and the improvement constitutes
learning by the biological network — with performance gains presented
as accumulating over training cycles/sessions.

**Predicted verdict: PARTIAL** — Survives 0.10 / Partial 0.70 / Fails 0.20

**Reasoning.**
- The screen found the within-run component genuinely RL-specific:
  RL runs improve (median Spearman of episode vs reward +0.085,
  p < 0.001) while the deposit's own random-feedback and no-feedback
  arms are flat. That is why "fails" is not my modal prediction.
- But the retention component — improvement carrying across sessions —
  showed no signal at screen level (median ρ = −0.127 against
  chronological run order, shuffle p = 0.902). If the full audit
  confirms gains resetting between sessions, the claim survives only
  as within-session adaptation, which is narrower than "learning a
  task" as the deposit frames it: partial.
- Session-day identity is strongly decodable from the deposit's own
  features (balanced accuracy 0.699 vs 0.125 chance), so the audit's
  stratified controls are load-bearing: any pooled analysis that
  ignores organoid × day will overstate the effect.
- The residual 0.20 on fails covers the `top_decile` metric: if the
  headline number is an artifact of how the top decile of runs is
  selected, the claim fails rather than trims.

**What would change my prediction.**
- A chronological-split retention test in the full audit showing
  across-session improvement above the shuffle null → move toward
  **survives**.
- Evidence that the within-run effect concentrates in a minority of
  organoids or in the metric's selection step → move toward **fails**.

**Scoring.** Scored when Audit #8 is published.

---

## Prediction 004 — Pavlinek et al. 2026 (KCL KORDS collection)

**Registered:** 2026-10-14. **Status: OPEN — design inventory only; no claim test run.**
Data status at registration: the collection is public on King's Open
Research Data System (figshare, collection DOI 10.18742/c.8097928),
all parts CC0. I have downloaded only the well-average + metadata part
(2.1 MB) and inventoried its design; no analysis of any claim has been
run. The raw spike-level parts (~138 GB) have not been downloaded.

**The claims.** Pavlinek et al., *Cell Reports Methods* 2026 (DOI
10.1016/j.crmeth.2026.101371): in dissociated human cerebral organoids
on multi-well MEA, (1) a chemical-LTP protocol induces sustained,
potentiation-like changes in network activity (within-well baseline
plus 15–19 post-treatment recordings out to 144 h, against a vehicle
arm), and (2) acute 17β-estradiol (10 nM) modulates network activity
against vehicle in paired within-well recordings.

**Predicted verdict: PARTIAL** — Survives 0.35 / Partial 0.45 / Fails 0.20

**Reasoning.**
- This is the best-controlled manipulation design in my battery so
  far: within-well comparisons, a vehicle arm, and dense post-treatment
  time series. Designs like this are why "survives" gets 0.35 — more
  than double any transfer claim I have registered.
- The residual risks are the usual two, in a specific form. First,
  developmental drift: these networks change enormously day to day
  (the collection's own longitudinal arm records ~25 daily sessions
  per well), so a "sustained" post-treatment change must beat the
  trajectory the same well was already on, not just its baseline.
  Second, well/plate identity: with well-average features this rich
  (~55–85 parameters), session and well identity will be highly
  decodable, and any analysis pooling wells without stratification
  will inherit that confound.
- Partial is modal because methods-paper claims usually survive in a
  narrower form: a real effect on some parameters, at some timepoints,
  in some lines — with the headline stated more broadly than the
  parameter × time × line grid supports.

**What would change my prediction.**
- The chLTP effect holding against the vehicle arm at matched
  post-treatment times, in a majority of parameters and lines →
  move toward **survives**.
- The E2 effect collapsing to a plate or well-identity artifact under
  stratification → move toward **fails**.

**Scoring.** Scored when the audit is published.

---

## Scoreboard

| # | Target | Registered | P(Survives) | P(Partial) | P(Fails) | Verdict | Brier |
|---|--------|-----------|-------------|------------|----------|---------|-------|
| 001 | Iannello et al. 2025 | 2026-10-14 | 0.15 | 0.60 | 0.25 | — open — | — |
| 002 | Kim et al. 2026 | 2026-10-14 | 0.20 | 0.55 | 0.25 | — open — | — |
| 003 | Robbins et al. 2025 | 2026-10-14 | 0.10 | 0.70 | 0.20 | — open — | — |
| 004 | Pavlinek et al. 2026 | 2026-10-14 | 0.35 | 0.45 | 0.20 | — open — | — |

**Running mean Brier score:** — (no scored predictions yet)

*This program reports survivals as loudly as failures. If both of these
predictions score badly, that will be printed here too.*
