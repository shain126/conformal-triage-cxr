# SIIM-CAIMI26 AI Builder Showcase — Submission

**Title:** A Conformal Triage Layer: Bounding Missed Abnormalities on Chest X-ray
Screening Worklists

**Repo:** `<REPO URL>` · **Demo:** `<DEMO URL>` · **Video:** `<VIDEO URL>`

---

### Problem Statement

Screening chest X-ray worklists are dominated by normal studies, yet every study is
read. AI abnormality scores were supposed to help, but deployed classifiers output
**uncalibrated** confidence — in our own run on real NIH data, model scores occupied
a narrow 0.52–0.73 band where a "0.63" carries nowhere near 63% probability of
disease. A radiologist therefore has no defensible basis for acting on a score, and
no way to state what a given cutoff costs in missed findings. The blocker is not
detection accuracy; it is the absence of a rule for *acting* on a score with a stated
safety budget.

### Approach (What You Built)

A model-agnostic, post-hoc **conformal triage layer** that wraps any binary CXR
classifier. The radiologist sets one dial: `alpha`, the missed-abnormal rate they can
tolerate. Using a labelled calibration batch, the layer sets threshold `t_low` at the
k-th smallest abnormal score, `k = floor(alpha(n+1))`, and auto-clears studies scoring
below it; the rest stay on the worklist ranked by suspicion. Exchangeability yields
exactly `P(cleared | abnormal) = k/(n+1) <= alpha` — distribution-free and
finite-sample. Crucially it uses only the *ranking* of scores, never their face value,
so it works on models whose confidence numbers cannot be trusted.

### Demo / Evidence of Function

A browser demo recomputes everything live from **real inference on 8,000 real NIH
ChestX-ray14 studies** (5,775 patients) scored by `torchxrayvision` DenseNet-121
(AUROC 0.746, prevalence 41.2%). Across 500 **patient-disjoint** calibration/test
splits the empirical miss rate lands *on* the theoretical bound at every level —
0.0098 at alpha=0.01, 0.0497 at alpha=0.05, 0.1003 at alpha=0.10 — cutting the
worklist 4.4% / 15.0% / 24.4% respectively. The layer spends its entire risk budget
rather than hiding under it.

### Clinical / Operational Impact

At alpha=0.05 the layer removes 15% of this cohort's worklist while clearing at most
5% of abnormals. Two sensitivity analyses set expectations honestly: resampling to
screening prevalence (5–10% abnormal, versus NIH's enriched 41%) raises the cut to
~19%, and the binding constraint is model quality — at AUROC 0.90 the same layer
would cut 51%. The miss rate never moves across either sweep. **Validity is
model-agnostic; only the payoff scales.** That is the argument for a reusable layer
rather than another retrained classifier: it converts future model improvements into
worklist reduction under an unchanged guarantee.

### Current Stage

Working prototype. Retrospective, binary normal/abnormal, not PACS-integrated. The
guarantee is per-study and marginal — not per-patient, not per-subgroup — and assumes
calibration and incoming studies are exchangeable, so scanner or case-mix shift
requires recalibration. Conformal prediction itself is established prior art; the
contribution is packaging it as something a non-ML radiologist can tune and defend.

### Feedback Sought

Is a bounded missed-abnormal rate the guarantee that would actually let you drop
studies from a worklist — or is workflow integration the real blocker? What alpha
would you sign your name to? And should the guarantee hold per-subgroup (age, sex,
view) before this is worth piloting?

---

*Word count of the six sections above: ~500.*
