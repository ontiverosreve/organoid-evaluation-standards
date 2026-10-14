# Open Evaluation Standard for Organoid-Intelligence Claims — v0.5 (DRAFT)

**Status:** Draft for comment. Not ratified by any body. Intended to live at
`github.com/ontiverosreve/organoid-evaluation-standards`.

**Changes from v0.4:** adds §3.9 (specification robustness), §4.5 (batch
correction), and §4.6 (decision rules for wash-in / A-B-A claims); amends
§3 intro (eight → nine required controls), §4.1 (ordering endpoints:
preference, cross-preparation consistency, and the corrected Audit #1
evidence), §4.3 (cross-reference to §4.6), §5 (reporting for corrected and
wash-in claims), §6 (Audit #1 contrast corrected), and §7 (v0.5 entry;
evidence base corrected to eight datasets, seven labs, with the 2026-10-14
follow-up analyses). New or amended clauses are tagged **[New in v0.5]**,
**[Amended in v0.5]**, or **[Corrected in v0.5]**. The full changelog follows.

---

## Changelog — v0.5 (2026-10-14)

Every change below is driven by a worked analysis, per the amendment rule in
§7. Sources are the 2026-10-14 follow-up analyses run against the standard's
own evidence base.

1. **§3.9 Specification robustness [new].** Where a headline number depends
   on definitional analysis choices, the claim must report its specification
   range. *Evidence:* specification-curve pilot on Audit #5 (Van der Molen
   et al. 2025) — 72 reasonable specifications on 5 human-organoid
   recordings: backbone fraction median 10.7%, IQR 7.1–13.3%, range
   2.8–20.9%; no specification reached the paper's reported 28% ± 14% as a
   cross-file mean; definitional choices moved the estimate more than
   detection choices.
2. **§4.1 Dose-monotonicity [amended].** Ordering endpoints are now
   explicitly preferred over separation endpoints where the design contains
   an ordered variable recorded out of order; monotonicity must be
   sign-consistent across independent preparations where more than one
   exists. *Evidence:* endpoint re-analysis of Audit #1 (Sharf et al. 2022)
   — the drug-direction projection ordered by dose at mean Spearman
   ρ +0.93…+1.00 across all four organoids (all 36 runs ≥ +0.80), while no
   purely descriptive endpoint cleared the cross-organoid bar.
3. **§4.1 / §6 Audit #1 evidence [corrected].** The v0.4 text credited the
   paper's median inter-burst-interval (IBI) shortening as replicated. The
   re-analysis shows it is organoid-specific: ρ = −0.90 (2950), 0.00 (2953),
   −0.30 (2954), −0.40 (2957). What generalized is the dose-ordered
   projection, not the paper's chosen measure. §6's worked contrast is
   corrected accordingly.
4. **§4.5 Batch correction [new].** Correction cannot rescue a design in
   which condition is aliased with batch; any batch-corrected claim must
   report a same-condition control pair after the identical correction;
   correction must be fit inside the training fold and transfer evaluated
   leakage-safe; the mirror-centroid artifact (systematically below-chance
   accuracies after global-fit correction) is named. *Evidence:*
   batch-correction test on Audits #1 and #6 — Sharf's design matrix
   [batch | dose] is rank 5 of 6 (aliased by algebra); train-fit ComBat left
   the no-drug pair separating at 0.83 while dose-monotonicity appeared
   preserved (ρ +0.77…+0.97); global-fit ComBat inflated Liu
   leave-one-batch-out decoding to 0.905 vs 0.765–0.778 leakage-safe
   (uncorrected 0.865); where condition varies within batch, ComBat
   retained 91% of Liu's headline effect (p=5.4e-09) and drove
   batch-identity decoding to chance.
5. **§4.6 Decision rules for wash-in (A-B-A) claims [new].** Reversal is
   the load-bearing criterion and a claim may not rest on
   classification-beats-nulls alone; time-locking counts only with a
   negative-control window; every run must report a baseline-phase drift
   estimate and runs beyond a pre-registered drift bound are underpowered,
   not negative. *Evidence:* simulation stress test of the preregistered
   wash-in decision rules — 9,600 simulated 3-run experiments (28,800 runs)
   built from real Stage C states with fitted drift: experiment-level false
   alarms 0.0–6.5%, but classification alone passed 33–52% of no-effect
   runs under drift, time-locking as operationalized passed ~99–100%
   (non-discriminating), and random-walk drift at ≥1× session scale cut
   detection to ≤19% even for the largest planted effects.
6. **§4.3 Wash-in design [amended].** Cross-reference to §4.6 added: the
   design removes session confounding by construction, and its claims are
   judged under drift-aware decision rules.
7. **§3 intro [amended].** Nine required controls (was eight); verdict
   consequence for §3.9 added; pointer to §4.5 where batch correction is
   used anywhere in a pipeline.
8. **§5 Reporting checklist [amended].** Item 6 extended to specification
   ranges and corrected-claim control pairs; new item 9 requires wash-in
   drift and power-condition reporting.
9. **§7 Evidence base [corrected and extended].** The v0.4 table's count
   ("Six datasets, five labs") completed to the full audited base — eight
   datasets, seven labs — with Audit #5 and Audit #6 rows added, the v0.5
   version entry, and a follow-up-analyses table for 2026-10-14.

---

## 1. Purpose and scope

This standard is the shared answer key for claims about organoid intelligence —
computation, sensing, or drug response in living neural tissue. It exists for one
reason: **measurement before claims.** Every prior computing era got its benchmarks
before its hype cycle. Biocomputing has none, and the claims are getting bolder
while the yardstick is missing.

The standard does not tell you which model to use, which organoids to grow, or
what counts as interesting science. It tells you which controls a
classifier-based claim must survive before anyone is obliged to take its headline
number seriously — and what to report so others can check your work.

**Scope (v0.1):** classifier-based claims on extracellular electrophysiology
recordings (MEA, HD-MEA, shank probes) from neural tissue, including dose and
intervention claims. The controls are pipeline-agnostic: they apply to reservoir
computers, SVMs, CNNs, or anything else that turns spike data into labels.

**Scope note (v0.3):** repeated-session training claims — "accuracy improved from
day 1 to day 3", "the network learned the patterns across sessions" — are in
scope whenever the training is **open-loop** (stimulation delivered on a fixed
schedule, no behavior-contingent feedback). Audit #4 was exactly this kind of
claim and motivated §3.7. **Closed-loop learning claims** (behavior-contingent
feedback, e.g., "the culture learned Pong") remain **out of scope (for now)**:
the feedback loop introduces its own confound family — reward-correlated drift,
exploration-exploitation structure, closed-loop artifact coupling — and needs
its own controls (see §7).

**Out of scope (for now):** closed-loop learning claims (e.g., "the culture
learned Pong"), calcium-imaging-only claims, and simulated data. These need
their own control families and are marked as future extensions in §7.

The standard is deliberately adversarial to its author's own results. It was
written from a controlled failure: a classifier that reached 1.000 accuracy and
was then explained away entirely by its null controls (§6). A standard that
cannot kill its author's best result is decoration.

---

## 2. The failure mode this standard exists for

Whenever experimental conditions are confounded with recording sessions — the
normal case in retrospective electrophysiology, where each condition comes from
its own recording — any sufficiently expressive classifier can reach perfect
accuracy by learning **session fingerprints** (electrode drift, culture state,
ambient noise, burst statistics) instead of the effect of interest.

No standard train/test split detects this, because the confound is perfectly
correlated with the label in every split. The label-shuffle null, often treated
as the responsible control, can pass with room to spare while the result is
entirely artifactual. Only controls that ask what the classifier does when the
*session* is the label expose it.

The rule: **a claim that cannot survive its nulls is a claim about the nulls.**

This failure family is not unique to retrospective data. Running reservoir
computing on live human cultures on the CL1 closed-loop platform, Kagan's group
found independently that RC decoders required "strict artifact control and
trial-based cross-validation to distinguish network computation from artifactual
signal separability or temporal data leakage" before any architecture
comparison could be interpreted (Loeffler et al. 2026, bioRxiv
10.64898/2026.08.10.743829). The decoder reading the rig instead of the tissue
is the rule, not the exception.

---

## 3. Required null controls for classifier-based claims

All nine controls below are **required** for any published or preprinted claim
of the form "classifier X distinguishes conditions A and B in neural recordings."
Each control gets a verdict: **pass**, **fail**, or **not applicable (with
written justification)**. A fail on 3.1 or 3.2 withdraws the headline claim. A
fail on 3.3–3.6 qualifies it (see criteria). A fail on 3.7 withdraws the
*training* claim while leaving any within-session decoding claim to be judged
by 3.1–3.6 — the standard distinguishes a real effect with a wrong story from
no effect at all. A high collapse fraction on 3.8 invalidates the transfer test
itself: fix the readout and re-run before any verdict on 3.7 is drawn.
**[Amended in v0.5:]** a fail on 3.9 qualifies the headline number: report the
specification range, or restate the claim at the range. Where batch or session
correction is applied anywhere in the pipeline, §4.5 applies in addition to
everything in this section.

### 3.1 Session-identity control (required)

Train the *identical* pipeline to predict **recording-session identity** rather
than the condition label, on session pairs that share a condition. Minimum: one
same-condition, different-session pair with a time gap comparable to the gap
between your conditions (e.g., two drug-free recordings hours or days apart).

- **Pass:** session-only accuracy near chance, OR substantially below the
  headline condition accuracy with non-overlapping shuffle-null distributions.
- **Fail:** session-only accuracy at or near the headline accuracy. The pipeline
  reads session identity; the condition claim is withdrawn. This is what
  happened in Stage C (§6) and in Audit #1.

If no same-condition session pair exists in the data, the control is
**not applicable** — and the claim must then carry the explicit caveat:
*"session confounding cannot be excluded with the available recordings."*
Absence of the control is not a pass.

### 3.2 Label-shuffle null (required)

Shuffle the condition labels, re-run the full pipeline, repeat ≥20 times with
independent shuffle RNG seeds. Report the **mean, the maximum, and the count of
repetitions reaching the headline accuracy** — not the mean alone. A null that
reports only "mean ≈ chance" while one repetition in twenty hits 0.9 is not a
null; it is a warning.

- **Pass:** 0 of ≥20 repetitions reach the headline accuracy, and the maximum
  sits well below it.
- **Fail:** any repetition reaches the headline accuracy. The pipeline can
  manufacture the result from noise.

### 3.3 Chronological split (required, in addition to interleaved)

Report accuracy on both an interleaved split (train/test blocks mixed across
the recording) and a **chronological split** (train on early blocks, test on
late blocks). Interleaved splits leak slow within-recording drift into both
train and test; the chronological split tests whether the classifier survives
being evaluated on the recording's future.

- **Pass:** headline accuracy holds on both splits.
- **Fail (qualifying):** accuracy collapses on the chronological split. The
  classifier is reading within-recording temporal structure, not a
  condition-level effect.

For decoders with state carryover (e.g., reservoir time-bin decoding),
cross-validation folds must additionally respect trial/stimulation-epoch
boundaries — no trial may be split across train and test. Where time-bin
decoding is used, report holdout-over-trials alongside any holdout-over-time;
holdout-over-time alone leaks adjacent bins from the same trial across the
split and inflates accuracy.

### 3.4 Within-recording stability check (required)

Split a single recording into first half vs. second half and classify halves.
A session fingerprint should be *stable* within a recording: halves of the same
file should separate near chance (they are the same session). If halves
separate cleanly, the "session" is not a stable unit and the session-identity
control in 3.1 needs re-examination — the confound structure is more complex
than file identity.

- **Pass:** near-chance accuracy (consistent with shuffle null).
- **Fail (qualifying):** clean separation. Report it and revise the confound
  model before claiming anything about conditions.

### 3.5 Rate-only baseline (required)

Before any temporal-pattern claim, classify conditions using **block-mean firing
rate alone** (one feature per block). This separates "less spiking" from
"different spiking."

- If the rate-only baseline reaches the headline accuracy, the claim is a
  **rate claim**, not a temporal-pattern claim. Say so.
- If the rate-only baseline sits at chance while the full pipeline separates
  conditions, the pipeline is reading temporal structure — which is the
  stronger and more interesting claim, and the one that must then survive 3.1.

Preprocessing choices change what the classifier is allowed to read. Per-file
standardization, for example, removes mean-rate differences by construction.
**State exactly what your preprocessing removes**, and run the rate-only
baseline on the *unstandardized* data so the comparison is honest.

### 3.6 Stimulation-artifact control (required where inputs are delivered by electrical stimulation)

Report decoder accuracy with stimulation artifacts removed/blanked versus
retained. The headline claim must survive artifact removal; accuracy that
collapses when the stimulus waveform is blanked is a claim about the
stimulator, not the tissue.

- **Pass:** accuracy with artifacts blanked is consistent with the headline
  (within the shuffle-null spread of §3.2).
- **Fail:** accuracy collapses when the stimulus waveform is removed. The
  decoder is reading the input signal, not network computation. Withdraw the
  computation claim.

This control complements §3.1 rather than replacing it: 3.1 asks what the
pipeline reads across sessions when the condition is held fixed; 3.6 asks
what it reads within a run when the neural response is held fixed and only
the stimulation artifact is removed.

### 3.7 Cross-session transfer (required for repeated-session training/plasticity claims)

Where the claim is that repeated training, stimulation, or exposure *changed*
the network — accuracy improved across days, representations sharpened,
responses stabilized — the decoder must be tested **across the session boundary
the claim says was crossed**. Within-session decoding, however strong, cannot
establish learning: a classifier can reach 0.96 within a day on session
fingerprints alone.

Train the identical pipeline on session A and test on session B, then reverse
(train B, test A). Report both directions. Match units or channels across
sessions by electrode ID (or the closest available unit-matching rule, stated
explicitly), and report the fraction of units that matched.

- **Pass:** above-chance transfer in both directions, substantially above the
  §3.2 shuffle null. The representation the decoder learned in one session
  exists in the other — the thing that "improved" is a thing, not a
  fingerprint.
- **Fail:** transfer at or near chance. Day-to-day accuracy changes are then
  drift, reconfiguration, or session fingerprints — not learning. Below-chance
  transfer is stronger evidence still: a session-A decoder actively misled by
  session B means the sessions' representations point in different directions.
  Withdraw the training claim.

This control is the sibling of 3.1 and 3.6: 3.1 asks what the pipeline reads
across sessions when the condition is held fixed; 3.6 asks what it reads within
a run when the neural response is held fixed and only the artifact is removed;
3.7 asks whether what the pipeline learned in one session survives contact
with the next. A learning claim that cannot cross a session boundary is a claim
about sessions.

Failing 3.7 does not impugn within-session decoding. A decoder can genuinely
read patterns within a day while the training narrative fails — report both
results separately.

### 3.8 Transfer-test validity: the zero-state check
(required where the decoder carries internal state)

A §3.7 transfer test can fail for a methodological reason that has nothing to
do with the tissue: the decoder's internal state can collapse on out-of-session
data. A reservoir whose test-trial states sit near zero is not decoding badly —
it is not decoding at all, and its argmax is degenerate. Reporting that
accuracy as "no transfer" confuses a broken instrument with a negative result.

Before interpreting any §3.7 result from a state-carrying decoder (reservoir,
RNN, LSTM, or any readout over evolving hidden states):

1. Declare, **before** running the transfer, a state-norm threshold (e.g., a
   low quantile of the training-state norm distribution).
2. Report the **collapse fraction**: the fraction of test trials whose
   readout-state norm falls below that threshold, for each transfer direction.

- **Valid test:** collapse fraction near zero and transfer at chance — the
  honest fail. The representation genuinely does not cross the session
  boundary.
- **Invalid test:** a large collapse fraction. Audit #3: on cross-day transfer,
  77–78% of sparse test trials collapsed to near-zero reservoir states and the
  argmax degenerated, while the same pipeline decoded within-day at up to
  0.967. The transfer number measured the decoder's silence, not the tissue's
  change. Do not report the accuracy as evidence of no transfer; fix the
  readout (longer state integration, state-norm regularization, or a stateless
  decoder) and re-run §3.7.

The standard's adversarial rule applies to the test itself: a control that
cannot distinguish "no effect" from "broken instrument" is not a control.

### 3.9 Specification robustness **[New in v0.5]**
(required where a headline number depends on definitional analysis choices)

Some headline numbers are not read off the data; they are produced by a chain
of definitional choices — what counts as a burst, what counts as participation
in one, a minimum-count rule — each individually reasonable and jointly
decisive. A point estimate from a single specification hides how much of the
number is the choice. Where a headline quantity depends on such choices,
report a **specification range**: re-run the pipeline over a declared grid of
reasonable alternative settings for the load-bearing choices, and report the
distribution of the headline quantity across specifications (median,
interquartile range, full range). The grid must vary the *definitions*, not
only detection thresholds — in the worked case below, the definitional levers
moved the estimate roughly twice as much as the detection levers.

- **Pass:** the headline value sits inside the specification distribution, or
  the claim is restated at the range. The range is reported alongside the
  headline number, not in a supplement nobody opens.
- **Fail (qualifying):** reasonable alternative specifications move the
  headline outside its reported uncertainty or reverse the qualitative claim,
  and no range is reported. Audit #5 follow-up (2026-10-14): the Van der
  Molen et al. 2025 backbone fraction, re-run over 72 specifications (burst
  threshold ×3/×4/×5, minimum burst distance 500/800 ms, burst-edge fraction
  0.05/0.10/0.20, participation in 100% vs 90% of bursts, minimum spikes 2
  vs 3) on 5 human-organoid recordings from DANDI 001603 — median 10.7%,
  IQR 7.1–13.3%, range 2.8–20.9%. **No specification reached the paper's
  reported 28% ± 14% as a cross-file mean.** The levers that mattered were
  definitional: participation rule (100% of bursts: 8.4% vs 90%: 13.5%),
  minimum spikes (2: 13.5% vs 3: 8.3%), burst-edge width (0.05: 14.0% →
  0.20: 7.2%); burst threshold barely moved it (10.8/10.8/11.2%). A claim
  stated as a point, where the specification range is this wide, is a claim
  about the specification.

Specification robustness does not replace the null controls: a number can be
stable across specifications and still be a session fingerprint (§3.1), and
a number can pass every null and still be specification-fragile. Report both.

---

## 4. Required controls for dose and intervention claims

Dose/intervention claims ("compound X changes network dynamics") get everything
in §3 plus the following. The reason: a classifier that separates any two
sessions (§3.1 fail) can still be rescued by dose-response structure that
session identity cannot produce.

### 4.1 Dose-monotonicity test (required where ≥3 dose levels exist)

Project every condition's blocks onto the discriminative direction learned from
control vs. highest dose, and test whether the per-condition projections order
**monotonically in nominal dose** (Spearman rank correlation, reported across
all hyperparameter/seeds runs). This tests *ordering*, not separation — file
identity alone cannot produce a dose-ordered projection, because identity has
no dose axis.

- **Pass:** significant monotonic ordering, consistent across seeds.
- **Fail:** no ordering. The conditions differ (per §3) but not in a
  dose-dependent way; do not describe the effect as dose-dependent.

**[Amended in v0.5 — endpoint guidance.]** Prefer ordering endpoints of this
kind over separation endpoints wherever the design contains one: an ordered
variable (dose, time, wash-in phase) that was **recorded out of order**.
Ordering is the signal session identity cannot fake; separation is the signal
it fakes best. Two qualifications, both earned in Audit #1's 2026-10-14
endpoint re-analysis:

1. **Not every monotone endpoint generalizes — require consistency across
   preparations.** The paper's own endpoint, median inter-burst-interval
   shortening, ordered with dose on organoid 2950 (Spearman ρ = −0.90) but
   not on the held-out organoids (ρ = 0.00, −0.30, −0.40); within-burst CV
   ran in the paper's wrong direction on all four organoids; no purely
   descriptive endpoint was sign-consistent at |ρ| ≥ 0.80 in ≥3 of 4
   organoids. The drug-direction projection did generalize: mean ρ +0.967 /
   +1.000 / +1.000 / +0.933 across the four organoids, **all 36 runs
   ≥ +0.80, 30 of 36 at exactly +1.00**, under non-monotonic recording order
   in every organoid (§4.2). Where more than one independent preparation
   exists, a monotonicity pass on one of them is a pilot result, not a dose
   response.
2. **Ordering needs an axis.** The test has power only where the design
   contains an ordered variable recorded out of order. Stage C has no dose
   series at all, and its fallback time-monotone test was also null
   (within-session Spearman of block features against time: all p ≥ 0.121) —
   its sessions are temporally stable, so its 1.000 separation stays a
   between-session property with nothing for an ordering endpoint to grip.
   Do not approximate a monotonicity test a design cannot support; state
   that the axis is absent.

### 4.2 Non-monotonic administration order (required where feasible)

Run the dose series in an order that is **not monotonic in dose** (e.g.,
control → low → high → control → medium → control → highest). If doses are
administered in increasing order, slow time drift and cumulative carryover are
perfectly confounded with dose, and 4.1 proves nothing. Report the actual
recording order. If the data are retrospective and the order was monotonic,
say so and downgrade the monotonicity claim accordingly.

### 4.3 Wash-in design (gold standard, recommended)

The design that is **immune to session confounding by construction**: one
continuous recording, intervention applied mid-recording, pre/post blocks drawn
from the same session. There is no second session for the classifier to
fingerprint. Where live tissue and fluidics allow it, this is the design the
standard endorses. Retrospective file-per-condition data can never reach this
bar; that is a limitation of the data, not of the claim — but it must be
stated. **[Amended in v0.5:]** immune to *session* confounding is not immune
to confounding: the design trades the second session for within-run drift,
and claims from it are judged under the decision rules of §4.6.

### 4.4 Stimulation-parameter-matched control
(required where an intervention coincides with a stimulation-parameter change)

When a protocol change — medium swap, compound application, day boundary,
culture transfer — coincides with a change in stimulation parameters (rate,
amplitude, inter-stimulus interval, or electrode selection), no effect may be
attributed to the protocol change until a **within-day, parameter-matched
control** rules out the stimulation parameters as the cause. Stimulation rate
alone can move evoked responses by two orders of magnitude; a "protocol
effect" riding on a rate change is a rate effect until proven otherwise.

- **Pass:** the effect persists when stimulation parameters are held fixed
  across the protocol change (same-day blocks at matched rate/amplitude/ISI),
  or a within-day parameter sweep reproduces the attribution (varying only the
  rate reproduces the change).
- **Fail:** the effect tracks the stimulation parameter, not the protocol.
  Audit #3: an apparent medium-change effect — evoked-response ratios
  collapsing from a median 125.2× to 0.94× (p=1.3e-14) — was refuted by a
  within-day control: blocks recorded 2.4 h *after* the medium change at 10 s
  volley spacing showed the same ~100× responses as pre-change days, while
  1 Hz blocks showed 0.94× (U=18060, p=3.9e-08). The collapse was
  stimulation-rate saturation, not the medium.

Report the full stimulation-parameter schedule (rate, amplitude, ISI,
electrode map) alongside the protocol timeline in §5. A retrospective dataset
whose protocol changes are perfectly confounded with stimulation-parameter
changes, with no within-day matched blocks, cannot support the attribution —
say so.

### 4.5 Batch correction **[New in v0.5]**
(required where batch/session correction is used to support a claim)

Batch-effect correction imported from genomics (ComBat-style location/scale
adjustment, per-batch standardization) is not a null control and cannot
substitute for one. Correction separates batch from biology only where the
experimental design already does. A claim is credited with "batch-corrected"
robustness only if all three requirements below are met (2026-10-14
correction test on Audits #1 and #6; implementations validated on synthetic
ground truth before any real-data number was taken).

1. **The design must be non-aliased.** If each condition was recorded as its
   own batch — one dose per file, one condition per session — condition is an
   exact linear combination of the batch indicators and no location/scale
   method can attribute the signal. This is algebra, not opinion: in the
   Sharf et al. 2022 dose series, the design matrix [batch dummies | dose]
   has rank 5 of 6 columns, so a dose-protecting correction is not
   identifiable in that dataset at all. The remedy for an aliased design is
   experimental — interleave conditions within sessions (§4.2, §4.3) — not
   computational.
2. **Report a same-condition control pair after the identical correction.**
   Pass the §3.1 same-condition session pair (or an equivalent null) through
   the *identical* correction pipeline and report it next to the corrected
   headline. Partial correction preserves a shrunken copy of whatever
   per-batch structure was present — drug effect, drift, and fingerprint are
   the same axis in an aliased design, and shrinkage cannot tell them apart.
   Sharf, train-fit ComBat: dose pairs landed at 0.44–0.83 and
   dose-monotonicity *appeared* preserved (ρ +0.90 / +0.97 / +0.77 across
   seeds) — but the no-drug pair still separated at **0.83**, which gives
   the game away. Per-session z-scoring, which did bring the no-drug pair to
   chance (0.39–0.50), destroyed the dose ordering with it (ρ −0.47…−0.37):
   in an aliased design, correction removes dose and session together,
   because they are the same partition of the blocks. A preserved signal
   after partial correction of an aliased design is evidence of nothing
   except that the correction was partial.
3. **Fit the correction inside the training fold; evaluate transfer
   leakage-safe.** Correction parameters fit on all samples — including test
   samples and, in protected designs, their labels — leak. Liu &
   Buonomano 2025: global-fit ComBat produced a leave-one-batch-out (LOBO)
   decoding of 0.905, against 0.765 (per-batch z-scoring) and 0.778
   (train-fit ComBat with new-batch adjustment) leakage-safe, and 0.865 with
   no correction at all. Correction helps *attribute* an effect; it does not
   help *predict* across new batches, and a global-fit number that says
   otherwise is measuring the leak.

**The positive case, for the record.** Where condition varies within batch,
correction does its job and a real claim gets stronger: in Liu & Buonomano
(protocol varies within 5 of 7 recording-date batches), ComBat with the
protocol protected retained 91% of the headline effect (median event-time
difference −372.9 ms of −407.9 ms, **p=5.4e-09**, stratified permutation
0/2000) while driving batch-identity decoding to 0.100 / 0.057 — at or below
chance. Nobody can now call that result "just batch." That is what correction
is for; the requirements above exist so it is not also used as a solvent for
designs it cannot fix.

**Named artifact: the mirror-centroid failure.** Global-fit correction
followed by within-session cross-validation does not produce an honest
chance level. Centering each file on its own full-sample mean forces each
file's train and test centroids to be mirror images, so a readout trained on
the training blocks systematically anti-classifies the test blocks: in Sharf,
global-fit corrections collapsed every file pair from 1.000 to **0.00–0.25**,
below even the label-shuffle null (maxima 0.75–0.83). Below-chance accuracy
after correction is an artifact signature — do not read it as "correction
destroyed a real effect," and do not flip the labels and call it a discovery.

### 4.6 Decision rules for wash-in (A-B-A) claims **[New in v0.5]**
(required for claims from §4.3 designs)

A wash-in experiment replaces session confounding with within-run drift, so
its preregistered decision rules must be stated — and their behaviour under
drift known — before data collection. The rules below are calibrated on a
simulation stress test of the standard's own canonical wash-in decision
procedure (classification beats all nulls + time-locking + reversal, claimed
in ≥2 of 3 independent runs): **9,600 simulated 3-run experiments (28,800
runs)** whose block states were real Stage C reservoir states, with drift
fitted from real no-drug session pairs (median between-session centroid
distance 3.63 per-PC standardized units) under two temporal models
(mean-reverting and random walk) and planted effects δ = 0…4.5 (2026-10-14).
The unamended rules held experiment-level false alarms to 0.0–6.5% across
all drift conditions — the conjunction works. The component criteria,
however, are not equal, and the standard now weights them as measured:

1. **Reversal is the load-bearing criterion; classification alone is weak
   evidence.** At zero planted effect with realistic drift, the
   classification criterion (beats permutation and split-half nulls) passed
   in **33–52% of runs** — drift alone manufactures a baseline-vs-intervention
   difference half the time. The reversal criterion passed in only 16–51% of
   no-effect runs, and it is what keeps false claims down; per-run false
   passes reached 13.7%, compressed to ≤6.5% only by requiring the full
   conjunction in ≥2 of 3 runs. A wash-in claim may not rest on
   classification-beats-nulls alone, and any weakening of the reversal
   requirement is paid for directly in false alarms.
2. **Time-locking counts only with a negative-control window.** As commonly
   operationalized (readout crossing within a fixed window after onset),
   time-locking passed in **~99–100% of runs in every condition**, including
   pure drift — it discriminated nothing. A time-locking criterion is
   evidence only if it includes a negative-control window: no comparable
   crossing during the baseline phase. Until a criterion is operationalized
   that way, do not count it toward the claim.
3. **Report per-run drift; pre-register a power condition.** Detectability
   depends more on the temporal shape of within-run drift than on effect
   size. With mean-reverting drift at 0.5× session scale, a session-scale
   effect (δ ≈ 3, the size of the real Stage C baseline→drugs contrast) was
   detected in **84%** of experiments; at full session-scale mean-reverting
   drift, 41%; under random-walk drift at ≥1× session scale, **≤19% even at
   δ = 4.5** — a random walk defeats reversal by construction, because the
   state never comes back, drug or no drug. Therefore: (a) estimate each
   run's drift from its own baseline phase and report it with the result;
   (b) state in advance a drift bound above which runs are declared
   **underpowered, not negative**; (c) a drift-robust reversal variant
   (e.g., reversal tested against the drift-extrapolated baseline rather
   than the raw baseline) may be used, but it must be chosen and
   stress-tested before data collection, not after seeing data.

---

## 5. Reporting checklist

Every claim under this standard ships with:

1. **Data provenance:** dataset identifier, file names, sizes, SHA-256 hashes,
   license, and the exact subset used (well, organoid, channels, time ranges).
   Include culture architecture (2D monolayer / 3D organoid / modular
   microfluidic) and cell lineage (e.g., cortical, hippocampal) where known —
   architecture shapes decoding performance and must be part of the record.
2. **N blocks and files:** number of blocks per condition, block length,
   number of recording sessions per condition, and the **recording order**
   (which session was recorded when, at what dose). The
   **stimulation-parameter schedule** per block/session (rate, amplitude, ISI,
   electrode map) wherever stimulation was delivered — see §4.4.
3. **Preprocessing:** bin width, standardization scope (per-file? global?),
   channel exclusion criteria, and what the preprocessing removes (cf. 3.5).
   **[New in v0.5:]** where batch/session correction is applied: the method,
   whether the design is non-aliased (condition varies within batch), and
   whether correction parameters were fit inside the training fold (§4.5).
4. **Model and sweep:** architecture, all hyperparameters swept, all random
   seeds, and the rule used to pick the reported configuration (e.g.,
   "max mean accuracy; ties broken by first").
5. **Headline numbers with seeds:** accuracy per hyperparameter × seed, for
   every split (interleaved and chronological).
6. **Null-control numbers:** for 3.1–3.9 and 4.1–4.6, the same granularity —
   never a mean without its max and its seed-wise spread. Include the §3.8
   collapse fraction per transfer direction. **[New in v0.5:]** include the
   §3.9 specification range (median, IQR, range) for any headline number
   that depends on definitional choices, and, for any batch-corrected claim,
   the same-condition control pair's accuracy after the identical
   correction (§4.5).
7. **Machine-readable release:** configuration file, run tables (one row per
   run), block tables (one row per block: file, index, label, prediction,
   quality weight), and a **verification script** that independently re-checks
   the headline claims from the tables and exits non-zero on failure.
8. **Known limitations:** compound identity if unresolvable, missing session
   pairs, modest block counts — stated, not buried.
9. **Wash-in reporting [New in v0.5]:** for claims from §4.3 designs, the
   per-run baseline-phase drift estimate, the pre-registered drift bound and
   power condition, the negative-control-window result for any time-locking
   criterion, and the per-criterion pass rates (classification, time-locking,
   reversal) per run — not only the conjunction (§4.6).

Provenance over narrative: the release tables are the evidence. Numbers in
prose must be computable from the tables.

---

## 6. Worked example: the Stage C controlled failure

*Canonical application of the standard. This is the example future audits are
measured against.*

**Claim (withdrawn):** a 150-node reservoir computer distinguished post-drug
from baseline organoid MEA blocks (Trujillo et al., Zenodo 4751759, well 4, 48
ten-second blocks) at **1.000 test accuracy** across leak rates
α ∈ {0.005, 0.01, 0.02} and seeds {7, 11, 22}, on interleaved and chronological
splits. The 20-repetition label-shuffle null reached at most 0.750. Rate-only
baseline: ~0.56–0.63 — the reservoir was reading burst dynamics, not spike
counts.

**Control 3.1 (session identity): FAIL.** Four independent session-only null
controls — the identical pipeline trained to predict recording-session identity
rather than drug label — reached test accuracies up to 1.000. The pipeline
reliably classifies which recording session it is looking at; drug pairs are
not distinguishable from session identity alone.

**Verdict:** no drug-specific signal survives four independent null controls.
The headline claim is withdrawn, with receipts. The release package (config,
hashes, run tables, `verify.py`, eight checks passing) is public so anyone can
reproduce the failure.

**What the standard demands next:** session controls expose confounding; they
do not prove pharmacology absent. A drug-specific claim from this data would
require the authors' compound/well mapping or a wash-in recording (§4.3).
Neither exists, so no drug-specific claim is made. Contrast Audit #1 (Sharf et
al. 2022, diazepam dose series): the classifier signature failed §3.1
identically, but dose-monotonicity (§4.1) survived — the dose-ordered
projection at Spearman +0.90…+1.00 across 9 runs on organoid 2950, and, as the
2026-10-14 re-analysis established, at mean ρ +0.93…+1.00 across all four
organoids with every one of 36 runs ≥ +0.80, under non-monotonic
administration order per §4.2 — so the paper's core pharmacological claim was
**not** overturned, only the idea that 1.000 classifier accuracy proves it.
**[Corrected in v0.5:]** the v0.4 text also credited the paper's
inter-burst-interval shortening as replicated. It is not: median-IBI ordered
with dose on 2950 only (ρ = −0.90) and not on the held-out organoids
(0.00, −0.30, −0.40). The dose-response phenomenon generalizes; the paper's
chosen measure of it does not. The standard distinguishes these outcomes;
that is its job.

---

## 7. Versioning and amendment

- **v0.1** (2026-09-29): initial draft. Covers classifier-based claims on
  extracellular electrophysiology and dose/intervention claims.
- **v0.2** (2026-09-30): cites Loeffler et al. 2026 (bioRxiv
  10.64898/2026.08.10.743829) as independent confirmation of the failure
  family; adds §3.6 stimulation-artifact control; requires trial-respecting
  CV folds in §3.3 for state-carryover decoders; adds culture architecture
  and cell lineage to the §5 provenance checklist.
- **v0.3** (2026-10-04): adds §3.7 cross-session transfer control (required
  for repeated-session training/plasticity claims); clarifies in §1 that
  open-loop repeated-session training claims are in scope while closed-loop
  learning claims remain out of scope. Motivated by Audit #4 — honest-controls
  audit of Shao et al. 2025, "Repetitive training enhances the pattern
  recognition capability of cultured neural networks," *PLOS Computational
  Biology* 21(4):e1013043 (https://doi.org/10.1371/journal.pcbi.1013043):
  within-day pattern decoding was real (0.96 two-pattern / 0.90 six-pattern,
  far above shuffle nulls) but cross-day transfer collapsed (0.47 / 0.29,
  one culture below chance at 0.205), session identity dominated the features
  (culture-ID 0.94, day-ID 0.94), and the "repeated training improves
  recognition" narrative did not survive (paired day-3−day-1 gains n.s. in
  both experiments). §3.6 was battle-tested in the same audit: the [0,2] ms
  stimulation artifact alone decoded patterns at 0.985, confirming the blank
  is load-bearing. Audit:
  `github.com/ontiverosreve/organoid-evaluation-standards/tree/main/audits/audit-04-shao2025`.
- **v0.4** (2026-10-08): adds §3.8 transfer-test validity (zero-state collapse
  diagnostic for state-carrying decoders) and §4.4 stimulation-parameter-matched
  control; extends §5 provenance to the stimulation-parameter schedule.
  Motivated by Audit #3 — honest-controls audit of two FinalSpark whole-life
  MEA datasets (FS369, 21.2-day continuous; FS437, 5.7-day), the first audit
  of data with no published claim attached: stimulation-evoked responses
  survived every null (FS437 post-stim elevations 4.17× / 2.73× / 4.56× vs
  stim-shuffled null ≈ 1.0; FS369 within-day amplitude decoding 0.725 ± 0.101,
  artifact-blanked 0.700; dead-culture negative control clean), but
  cross-session transfer failed on both datasets (canonical reservoir
  0.322 / 0.330 vs 1/3 chance; FS437 forward P2 recall 0.16 vs 0.42 shuffled
  null), session identity decoded from background activity alone (day-ID
  0.9215 vs 1/22 chance; epoch 0.9867), and the apparent medium-change effect
  was refuted as stimulation-rate saturation (U=18060, p=3.9e-08). Audit:
  `github.com/ontiverosreve/organoid-evaluation-standards/tree/main/audits/audit-03-finalspark`.
- **v0.5** (2026-10-14): adds §3.9 specification robustness, §4.5 batch
  correction, and §4.6 wash-in decision rules; amends §4.1 (ordering-endpoint
  guidance) and corrects the Audit #1 endpoint evidence in §4.1 and §6;
  completes the evidence base to eight datasets, seven labs (Audit #5 and
  Audit #6 rows added). Motivated by four follow-up analyses of the existing
  evidence base, run 2026-10-14 (see the follow-up table below): a
  9,600-experiment simulation stress test of the wash-in decision rules
  (§4.6); a ComBat/z-scoring correction test on Audits #1 and #6 (§4.5);
  a 72-specification curve on Audit #5 (§3.9); and an endpoint and
  sensitivity re-analysis of Audits #1 and #6 (§4.1) — which also
  re-confirmed Audit #6's survival under every non-degenerate parameter
  variant (CoG window 600 ms: p=5.53e-07; 1,000 ms as released:
  p=1.07e-09; leave-one-batch-out 0.865 at both, never below 0.831 across
  variants) and under batch correction (91% of the effect retained,
  p=5.4e-09).
- Amendments are proposed as issues/pull requests on the public repository,
  with a worked example (pass or fail) attached. A control earns its place by
  killing a real headline, not by sounding prudent.
- Planned extensions: closed-loop learning claims, calcium-imaging claims,
  multi-lab replication criteria, and organoid-held-out generalization
  standards. The closed-loop extension should absorb decoding-method
  guidance (e.g., frequency-domain vs. time-bin decoding) and decoder-bias
  controls alongside the artifact and trial-respecting controls already
  established by Loeffler et al. 2026.

### Evidence base (v0.5)

| # | Dataset | Lab | Within-session effect | Cross-session transfer |
|---|---|---|---|---|
| 1 | Stage C (Trujillo et al., Zenodo 4751759) | Muotri, UCSD | drug classifier 1.000 | fails session nulls |
| 2 | Audit #1 (Sharf et al. 2022) | Sharf, UCSC | dose signature 1.000 | fails session nulls; dose ordering survives |
| 3 | Audit #2 (DishBrain, Kagan et al. 2022) | Cortical Labs | HCC learning +0.150, p=6.3e-13 | feedback-dependence survives with caveats |
| 4 | Audit #4 (Shao et al. 2025) | Shao, Tianjin Univ. | pattern decoding 0.96 / 0.90 | training claim fails; transfer collapses |
| 5 | Audit #3 (FinalSpark FS369) | FinalSpark | amplitude decoding 0.725 ± 0.101/day | transfer 0.322 / 0.330 = chance |
| 6 | Audit #3 (FinalSpark FS437) | FinalSpark | evoked 2.73–4.56× vs null ≈ 1.0 | forward P2 recall 0.16 vs 0.42 null |
| 7 | Audit #5 (Van der Molen et al. 2025) | Braingeneers, UCSC | backbone fraction 6.4% ± 7.5% vs reported 28% ± 14% | transfer fails; session-ID 0.89–0.99 |
| 8 | Audit #6 (Liu & Buonomano 2025) | Buonomano, UCLA | timing effect 149.8 vs 478.5 ms, p=1.85e-08 | **transfer PASSES (LOBO 0.865) — headline survives** |

### Follow-up analyses (2026-10-14)

| Analysis | Question | Result | Where it lands |
|---|---|---|---|
| Wash-in simulation stress test | Do the preregistered A-B-A decision rules hold under drift? | 9,600 experiments: false alarms 0.0–6.5%; classification alone passes 33–52% of null runs; time-locking non-discriminating (~99–100%); random-walk drift → ≤19% detection | §4.6 |
| Batch-correction test (Audits #1, #6) | Can ComBat-style correction rescue the failed claims? | Within-batch design vindicated (91% of Liu effect retained, batch-ID to chance); aliased design unrescuable (Sharf no-drug pair still 0.83 after correction; design rank 5/6) | §4.5 |
| Specification curve (Audit #5) | Is the 28% backbone fraction an analysis-choice artifact? | 72 specs: median 10.7%, range 2.8–20.9%; no spec reaches 28% as a cross-file mean; definitional levers dominate | §3.9 |
| Endpoint & sensitivity re-analysis (Audits #1, #6) | Which endpoints survive, and is the survivor robust? | Sharf IBI endpoint organoid-specific (−0.90 / 0.00 / −0.30 / −0.40); dose-ordered projection general (+0.93…+1.00, 36/36 runs ≥ +0.80); Audit #6 verdict unchanged under all variants | §4.1, §6 |

**Eight datasets, seven labs, one confound: session identity.** Within-session
effects hold; cross-session generalization fails in seven of eight — and the
eighth survives transfer, specification stress, and batch correction alike.

*The future is grown, not built. But it has to be grown honestly — and honesty
needs a checklist.*
