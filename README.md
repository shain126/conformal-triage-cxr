# Conformal Triage Layer for Chest X-ray Screening

A model-agnostic, post-hoc wrapper that turns **any** binary chest X-ray abnormality
classifier into a screening triage layer with a **finite-sample safety guarantee**.

The radiologist sets one dial — the missed-abnormal rate `alpha` they can tolerate.
The layer then auto-clears as much of the worklist as that budget allows, and
provably no more.

**[▶ Live interactive demo](https://shain126.github.io/conformal-triage-cxr/)** ·
runs entirely in the browser, no install

![Demo: dragging the tolerated-miss-rate dial recomputes the worklist reduction and
miss rate live on real NIH data](conformal-triage-demo.gif)

*Dragging α from 0.8% to 20% on real NIH scores: worklist reduction rises 2.5% → 37%
as the miss rate tracks the budget you set.*

---

## The problem

Radiologists don't need another detector. Screening worklists are dominated by
normal studies, and the AI scores meant to help are typically **uncalibrated** — a
"0.87" from a CXR model is not an 87% probability of disease. So the scores can't
safely be thresholded, and every study still gets read.

The missing piece isn't accuracy. It's a defensible rule for *acting* on a score.

## How it works

Split conformal prediction, one-sided:

1. Score a batch of labelled studies (the calibration set).
2. Take the abnormal ones, sort their scores ascending, and set
   `t_low` = the k-th smallest, where `k = floor(alpha * (n + 1))`.
3. In deployment: **auto-clear** any study scoring `< t_low`. Everything else stays
   on the worklist, ranked by score, highest suspicion first.

Exchangeability then gives, exactly:

```
P( auto-cleared | study is abnormal )  =  k / (n + 1)  <=  alpha
```

Distribution-free, finite-sample. No asymptotics, no assumption that the model is
calibrated — **only the ranking of scores is used, never their face value.** That is
what lets it wrap a model whose confidence numbers you don't trust.

## Results on real data

Real inference, real studies — not a simulation.

| | |
|---|---|
| Model | `torchxrayvision` **densenet121-res224-all** (DenseNet-121) |
| Data | **NIH ChestX-ray14**, 8,000 studies (HuggingFace `arudaev/chest-xray-14-320`) |
| Label | abnormal iff the NIH finding string is not `No Finding` |
| Score | `max` over the 14 NIH findings = P(any abnormality) |
| Base AUROC | **0.746** · prevalence **41.2%** abnormal |
| Patients | 5,775 (up to 13 studies each) |

Averaged over 500 random **patient-disjoint** 50/50 calibration/test splits
(`python validate_real.py`). Splitting on patient matters: NIH contributes up to 13
studies per patient, and a naive study-level split leaks ~30% of test studies into
the same patient as a calibration study.

| alpha | exact bound k/(n+1) | empirical miss rate [p5, p95] | worklist cut | normals cleared |
|---:|---:|---:|---:|---:|
| 0.01 | 0.0097 | **0.0098** [0.005, 0.016] | 4.4% | 6.8% |
| 0.02 | 0.0197 | **0.0197** [0.012, 0.029] | 7.2% | 10.8% |
| 0.05 | 0.0497 | **0.0497** [0.038, 0.063] | 15.0% | 22.0% |
| 0.10 | 0.0997 | **0.1003** [0.084, 0.117] | 24.4% | 34.4% |

The empirical miss rate lands **on** the bound at every level, not comfortably under
it — the layer spends its whole risk budget and converts it into worklist reduction.

Note the `[p5, p95]` spread. A deployment gets exactly **one** calibration set, and
its realized miss rate lands *near* alpha, not exactly on it. The bound is on the
expectation. This is the honest caveat that the headline number hides.

### What drives the operational benefit

15% is a real number but a pessimistic one, and it's worth being precise about why.
NIH ChestX-ray14 is a research cohort **enriched to 41% abnormal**; a screening
population is mostly normal. And the open classifier is mid-grade at AUROC 0.746.

**Case mix** — real scores, deduplicated to one study per patient (5,775) and
resampled to each prevalence, alpha = 0.05:

| prevalence | worklist cut | miss rate |
|---:|---:|---:|
| 5% | 18.3% | 0.044 |
| 10% | 19.3% | 0.047 |
| 20% | 18.0% | 0.049 |
| 30% | 16.5% | 0.050 |
| 41% (as measured) | 14.6% | 0.049 |

**Model quality** — simulated sensitivity curve, alpha = 0.05, prevalence 10%:

| base AUROC | worklist cut | miss rate |
|---:|---:|---:|
| 0.75 (≈ today's open model) | 22.1% | 0.048 |
| 0.85 | 38.8% | 0.048 |
| 0.90 | 50.9% | 0.048 |
| 0.95 | 67.8% | 0.048 |

Two things to read off these tables:

- **Model quality, not case mix, is the binding constraint.** At AUROC 0.746 the
  layer can only clear ~22% of normals at alpha=0.05, which caps the cut near 20%
  however normal the population is.
- **The miss-rate column never moves.** Validity is model-agnostic; only the payoff
  scales. That is the argument for building this as a *layer* rather than retraining
  a classifier — it turns every future improvement in CXR models into worklist
  reduction under the same guarantee, for free.

### The guarantee you can actually say out loud

The bound above is **marginal** — it holds in expectation over calibration sets. But
a deployment draws *one* calibration set and keeps that threshold. Measured across
possible deployments, only **~50% of them come in at or under alpha**, because the
marginal rule targets the mean.

`calibrate_threshold_pac(..., delta=0.05)` targets the quantile instead, giving a
**training-conditional (PAC)** guarantee — *"with 95% confidence, this threshold
misses at most alpha of abnormal studies."*

| alpha | rule | mean miss | p95 miss | deployments ≤ alpha | worklist cut |
|---:|---|---:|---:|---:|---:|
| 0.05 | marginal | 0.0490 | 0.0619 | 53.0% | 14.9% |
| 0.05 | **PAC** | 0.0402 | 0.0520 | **92.0%** | 12.5% |
| 0.10 | marginal | 0.0994 | 0.1159 | 53.0% | 24.4% |
| 0.10 | **PAC** | 0.0875 | 0.1031 | **90.0%** | 22.4% |

It costs ~2 points of worklist cut. Note the PAC column reads slightly under 95%
because it is measured on a *finite* test set, which adds estimation noise on top of
the true miss rate; against an analytically-known score distribution the same rule
delivers **95.6%** (vs 51.7% marginal). Treat the column as a lower bound.

### Per-finding safety audit

The guarantee is marginal over all abnormal studies, so in principle it could clear
pneumothoraces at 15% while averaging out to alpha. That has to be *measured*:

| finding | miss rate | vs alpha |
|---|---:|---:|
| Hernia | 0.066 | 1.3× |
| Fibrosis | 0.058 | 1.2× |
| Nodule | 0.054 | 1.1× |
| … | … | … |
| **Pneumothorax** | **0.036** | **0.7×** |
| Effusion / Edema | 0.017 | 0.3× |

No finding blows past the budget — worst is 1.3× on the rarest class (Hernia, ~17
positives, mostly noise), and the most time-critical finding here (pneumothorax)
sits *below* alpha. This is measured, not guaranteed; per-subgroup validity would
need a Mondrian conformal variant.

## Running it

Zero-download path — simulated scores, numpy only, works offline:

```bash
pip install numpy
python conformal_triage.py
```

Real-model path — downloads the pretrained CXR model and streams NIH studies:

```bash
pip install numpy torch torchvision torchxrayvision scikit-image datasets pillow
python run_real_model.py --n 8000     # writes real_scores.npz  (~10 min, MPS/CUDA/CPU)
python validate_real.py               # the tables above
python build_demo.py                  # regenerates index.html with your scores
```

`index.html` is fully self-contained — the scores are embedded, and every number on
the page is recomputed in the browser. Open it directly, or serve it over HTTP if
your browser restricts `file://` scripts.

## Files

| file | purpose |
|---|---|
| `conformal_triage.py` | the layer: `calibrate_threshold`, `triage`, `evaluate`, `conformal_bound` |
| `run_real_model.py` | real inference — CXR model over NIH studies → `real_scores.npz` |
| `validate_real.py` | validation on real scores + prevalence and model-quality sweeps |
| `build_demo.py` | bakes scores into `demo_template.html` → `index.html` |
| `index.html` | the interactive demo (generated) |

## Scope and limitations

- **Guaranteed:** `P(cleared | abnormal) <= alpha`.
- **Not guaranteed:** `P(abnormal | cleared)` — the share of the cleared pile that is
  abnormal. That moves with prevalence and gets no promise. The two are named
  separately throughout the code because conflating them is the standard error here.
- **Assumes exchangeability** between calibration and incoming studies. Scanner
  changes, case-mix shift, or a model update all break it and require recalibration.
  The layer is honest but not self-monitoring.
- **The headline table's guarantee is marginal** over calibration draws — only ~50%
  of deployments land at or under alpha. Use `calibrate_threshold_pac` if you need
  the training-conditional version (see above); it is implemented and validated.
- **Per-subgroup validity is measured, not guaranteed.** The per-finding audit above
  shows no finding exceeding ~1.3× alpha, but nothing in the theory prevents it. A
  Mondrian (per-stratum) conformal variant would. Demographic subgroups (age, sex,
  view position) are not audited at all — this dataset export carries no
  demographics.
- **Binary screening triage only**, retrospective, not PACS-integrated. The guarantee
  is per-study and marginal — *not* per-patient, and *not* per-subgroup. Subgroup
  validity is a known gap and the natural next step.
- **NIH labels are NLP-derived** from radiology reports and are themselves noisy.
- **The pretrained model saw NIH data in training**, so the absolute AUROC — and
  therefore the workload numbers — are optimistic. This does not threaten validity:
  the guarantee needs only that calibration and test studies are exchangeable, which
  holds because we split one pool at random. A leaky classifier yields a
  better-looking worklist cut; it cannot make the miss-rate bound fail.

## Prior work

This is **not novel math**. Split conformal prediction and selective classification
are established; the contribution is packaging — a runnable, model-agnostic layer
with one dial a non-ML radiologist can set and defend.

- Vovk, Gammerman & Shafer. *Algorithmic Learning in a Random World* — foundational
  conformal prediction.
- *Conformal Triage for Medical Imaging AI Deployment*, medRxiv, 2024.
- *Risk-Sensitive Conformal Prediction for Catheter Placement Detection in Chest
  X-rays*, [arXiv:2505.22496](https://arxiv.org/abs/2505.22496), 2025.

## License / data

NIH ChestX-ray14 is released by the NIH Clinical Center for open research use; no
credentialing required. Model weights via `torchxrayvision` (MIT).
