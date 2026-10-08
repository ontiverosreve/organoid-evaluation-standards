# Open Evaluation Standard for Organoid-Intelligence Claims — v0.3

**Status:** Draft for comment. Not ratified by any body. Intended to live at
`github.com/ontiverosreve/organoid-evaluation-standards`.

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

All seven controls below are **required** for any published or preprinted claim
of the form "classifier X distinguishes conditions A and B in neural recordings."
Each control gets a verdict: **pass**, **fail**, or **not applicable (with
written justification)**. A fail on 3.1 or 3.2 withdraws the headline claim. A
fail on 3.3–3.6 qualifies it (see criteria). A fail on 3.7 withdraws the
*training* claim while leaving any within-session decoding claim to be judged
by 3.1–3.6 — the standard distinguishes a real effect with a wrong story from
no effect at all.

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
stated.

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
   (which session was recorded when, at what dose).
3. **Preprocessing:** bin width, standardization scope (per-file? global?),
   channel exclusion criteria, and what the preprocessing removes (cf. 3.5).
4. **Model and sweep:** architecture, all hyperparameters swept, all random
   seeds, and the rule used to pick the reported configuration (e.g.,
   "max mean accuracy; ties broken by first").
5. **Headline numbers with seeds:** accuracy per hyperparameter × seed, for
   every split (interleaved and chronological).
6. **Null-control numbers:** for 3.1–3.7 and 4.1, the same granularity —
   never a mean without its max and its seed-wise spread.
7. **Machine-readable release:** configuration file, run tables (one row per
   run), block tables (one row per block: file, index, label, prediction,
   quality weight), and a **verification script** that independently re-checks
   the headline claims from the tables and exits non-zero on failure.
8. **Known limitations:** compound identity if unresolvable, missing session
   pairs, modest block counts — stated, not buried.

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
identically, but dose-monotonicity (§4.1, Spearman +0.90…+1.00 across 9 runs,
with non-monotonic administration order per §4.2) and the replicated
inter-burst-interval shortening survived — so the paper's core pharmacological
claim was **not** overturned, only the idea that 1.000 classifier accuracy
proves it. The standard distinguishes these two outcomes; that is its job.

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
- Amendments are proposed as issues/pull requests on the public repository,
  with a worked example (pass or fail) attached. A control earns its place by
  killing a real headline, not by sounding prudent.
- Planned extensions: closed-loop learning claims, calcium-imaging claims,
  multi-lab replication criteria, and organoid-held-out generalization
  standards. The closed-loop extension should absorb decoding-method
  guidance (e.g., frequency-domain vs. time-bin decoding) and decoder-bias
  controls alongside the artifact and trial-respecting controls already
  established by Loeffler et al. 2026.

*The future is grown, not built. But it has to be grown honestly — and honesty
needs a checklist.*
