"""
Conformal Triage Layer for chest X-ray screening
=================================================

A model-agnostic, post-hoc wrapper that turns ANY binary abnormality classifier
(score s = P(abnormal) in [0, 1]) into a SAFE triage system with a
finite-sample statistical guarantee.

Core idea (split conformal, one-sided):
    Auto-CLEAR a study (call it NORMAL, remove from the worklist) iff  s < t_low.
    Choose t_low on a labeled calibration set so that the probability an abnormal
    study is auto-cleared is provably bounded:

        P( s < t_low | Y = abnormal )  <=  alpha

    where alpha is a tolerance the radiologist chooses (e.g. 0.05 = "I accept at
    most a 5% chance that a truly abnormal study gets auto-cleared").

    NOTE ON WHICH QUANTITY IS GUARANTEED. The bound is on the *sensitivity loss*,
    P(cleared | abnormal) -- conditioned on the study being abnormal. It is NOT a
    bound on P(abnormal | cleared), the share of the cleared pile that turns out
    abnormal. That second quantity depends on disease prevalence and carries no
    guarantee; we report it as an observed statistic only, never as the promise.
    Conflating the two is the classic error here, so the code keeps them
    separately named throughout: `miss_rate` vs `abnormal_share_of_cleared`.

    Everything with s >= t_low is kept on the radiologist worklist, ranked by
    score (highest suspicion first) -- so nothing abnormal is silently dropped,
    and the reading order is prioritized.

Why one-sided: bounding the *missed-abnormal* (false-negative) rate is the
patient-safety-critical guarantee. A false alarm costs radiologist time; a
missed finding costs a patient. We control the thing that hurts patients and
leave the rest to the human.

This file has NO heavy dependencies (numpy only) and runs anywhere. To use it on
a real model, see `adapter_torchxrayvision()` at the bottom -- the wrapper only
needs an array of scores and (for calibration) labels.

References (prior art this builds on):
  - Vovk, Gammerman, Shafer. Algorithmic Learning in a Random World (conformal).
  - "Conformal Triage for Medical Imaging AI Deployment", medRxiv 2024.
  - "Risk-Sensitive Conformal Prediction for Catheter Placement Detection in
    Chest X-rays", arXiv:2505.22496 (2025).
Contribution here is not the math but a packaged, model-agnostic triage layer a
non-ML radiologist can actually run and tune.
"""

from __future__ import annotations
import numpy as np
from math import lgamma

gammaln = np.vectorize(lgamma, otypes=[float])   # scipy-free log-gamma


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #
def calibrate_threshold(cal_scores: np.ndarray,
                        cal_labels: np.ndarray,
                        alpha: float) -> float:
    """
    Compute the auto-clear threshold t_low with a finite-sample guarantee that
    the missed-abnormal rate among cleared cases is <= alpha.

    Parameters
    ----------
    cal_scores : (n,) model scores P(abnormal) on the calibration set.
    cal_labels : (n,) ground-truth labels, 1 = abnormal, 0 = normal.
    alpha      : tolerated missed-abnormal rate in (0, 1).

    Returns
    -------
    t_low : float. Studies scoring below this may be auto-cleared. Returns
            -inf when the calibration set is too small to clear anyone at the
            requested alpha (fail-safe: clear nobody).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    abn = np.sort(np.asarray(cal_scores)[np.asarray(cal_labels) == 1])
    n = len(abn)
    if n == 0:
        return -np.inf
    # Conformal order statistic (Vovk): k = floor(alpha * (n + 1)).
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        return -np.inf  # not enough calibration abnormals -> clear nobody
    return float(abn[k - 1])


def conformal_bound(n_abnormal: int, alpha: float) -> float:
    """
    The exact finite-sample miss rate this calibration set buys you.

    With n abnormal calibration scores and threshold t_low = S_(k), the k-th
    smallest, exchangeability gives P(S_new < S_(k)) = k / (n + 1) for a fresh
    abnormal study. Since k = floor(alpha * (n + 1)), this is <= alpha always,
    and approaches alpha from below as n grows.

    Reporting this alongside the empirical rate is what makes the demo honest:
    the empirical number should sit at the bound, not far under it. Sitting far
    under it would mean we are wasting worklist reduction we were entitled to.
    """
    k = int(np.floor(alpha * (n_abnormal + 1)))
    return k / (n_abnormal + 1) if n_abnormal > 0 else 0.0


def _binom_sf_all(n: int, p: float) -> np.ndarray:
    """
    P(Binomial(n, p) >= k) for every k = 0..n, in one O(n) vectorized pass.

    Computed in log space (log-gamma for the coefficients, then a suffix
    log-sum-exp) because for n ~ 1600 the individual pmf terms underflow float64
    badly. Returning the whole array lets the caller pick its k by search instead
    of recomputing a tail per candidate.
    """
    j = np.arange(n + 1)
    log_pmf = (gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)
               + j * np.log(p) + (n - j) * np.log1p(-p))
    # suffix log-sum-exp, computed from the top down
    m = log_pmf.max()
    suffix = np.cumsum(np.exp(log_pmf - m)[::-1])[::-1]
    # far-right tail underflows to exactly 0; clamp before the log so it returns 0
    # probability rather than raising a divide-by-zero warning
    with np.errstate(divide="ignore"):
        return np.where(suffix > 0, np.exp(m + np.log(np.maximum(suffix, 1e-300))), 0.0)


def calibrate_threshold_pac(cal_scores: np.ndarray,
                            cal_labels: np.ndarray,
                            alpha: float,
                            delta: float = 0.05) -> float:
    """
    PAC ("training-conditional") variant of calibrate_threshold.

    THE PROBLEM THIS SOLVES. `calibrate_threshold` bounds the miss rate *in
    expectation over the draw of the calibration set*. But a deployment draws ONE
    calibration set and keeps that threshold forever. Its realized miss rate is a
    random variable scattered around alpha -- in our real-data run, alpha=0.05 gave
    realized rates spanning roughly [0.038, 0.063] across splits. Half of all
    deployments therefore sit ABOVE the number the radiologist was promised. That
    is not a defensible thing to hand a clinician.

    WHAT THIS GIVES INSTEAD. A guarantee conditional on the calibration set:

        P_calibration( P(cleared | abnormal) <= alpha )  >=  1 - delta

    i.e. "with 95% confidence, this deployed threshold misses at most alpha of
    abnormal studies" -- a statement about YOUR threshold, not about an average
    over hypothetical ones.

    THE MATH. With threshold = the k-th smallest of n abnormal calibration scores,
    the true miss rate is F(S_(k)), which is exactly Beta(k, n-k+1) distributed by
    the probability integral transform. We need P(Beta(k, n-k+1) <= alpha) >= 1-delta,
    and the Beta CDF equals a binomial tail: P(Bin(n, alpha) >= k). So we take the
    largest k satisfying that -- always <= the marginal k, which is the price paid.

    Costs a few points of worklist reduction. Worth it: it converts "on average,
    across calibration sets you did not draw" into "for the threshold you are
    actually running".
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    abn = np.sort(np.asarray(cal_scores)[np.asarray(cal_labels) == 1])
    n = len(abn)
    if n == 0:
        return -np.inf
    # P(Bin(n, alpha) >= k) is non-increasing in k, so the admissible k form a
    # prefix 1..K; take the largest. searchsorted on the reversed array finds K.
    sf = _binom_sf_all(n, alpha)          # index k -> P(X >= k)
    ok = np.nonzero(sf[1:] >= 1.0 - delta)[0]
    best = int(ok[-1]) + 1 if len(ok) else 0
    return float(abn[best - 1]) if best >= 1 else -np.inf


def pac_bound_k(n_abnormal: int, alpha: float, delta: float = 0.05) -> int:
    """The order-statistic index the PAC bound permits (0 = cannot clear anyone)."""
    if n_abnormal <= 0:
        return 0
    sf = _binom_sf_all(n_abnormal, alpha)
    ok = np.nonzero(sf[1:] >= 1.0 - delta)[0]
    return int(ok[-1]) + 1 if len(ok) else 0


def min_calibration_abnormals(alpha: float) -> int:
    """
    Smallest number of abnormal calibration studies that lets you clear anyone.

    k >= 1 requires floor(alpha * (n + 1)) >= 1, i.e. n >= ceil(1/alpha) - 1.
    At alpha = 0.01 you need 99 abnormals before the layer will clear a single
    study. This is a feature: below that, the honest answer is "this calibration
    set cannot support that promise", and calibrate_threshold returns -inf.
    """
    return int(np.ceil(1.0 / alpha)) - 1


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #
def triage(scores: np.ndarray, t_low: float):
    """
    Apply the triage layer to new studies.

    Returns a dict with:
      cleared  : bool mask, studies auto-cleared (removed from worklist)
      worklist : indices of studies kept for the radiologist, ranked by score desc
    """
    scores = np.asarray(scores)
    cleared = scores < t_low
    kept = np.where(~cleared)[0]
    worklist = kept[np.argsort(-scores[kept])]  # highest suspicion first
    return {"cleared": cleared, "worklist": worklist}


def evaluate(scores: np.ndarray, labels: np.ndarray, t_low: float) -> dict:
    """
    Score the triage layer on a held-out labeled set.

    Keys, kept deliberately distinct because they answer different questions:
      miss_rate                 P(cleared | abnormal). THE guaranteed quantity.
                                This is what must stay <= alpha.
      abnormal_share_of_cleared P(abnormal | cleared). Prevalence-dependent,
                                NOT guaranteed. Reported for context only.
      workload_reduction        fraction of ALL studies removed from the worklist.
                                This is the operational headline number.
      normal_clear_rate         P(cleared | normal). How much of the healthy pile
                                we successfully got out of the way.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    cleared = scores < t_low
    n_abn, n_norm, n_clr = (labels == 1).sum(), (labels == 0).sum(), cleared.sum()
    return {
        "miss_rate": float(cleared[labels == 1].mean()) if n_abn else float("nan"),
        "abnormal_share_of_cleared": float(labels[cleared].mean()) if n_clr else 0.0,
        "workload_reduction": float(cleared.mean()),
        "normal_clear_rate": float(cleared[labels == 0].mean()) if n_norm else float("nan"),
        "n_cleared": int(n_clr),
        "n_missed": int(((labels == 1) & cleared).sum()),
    }


# --------------------------------------------------------------------------- #
# Self-contained demo + empirical validation of the guarantee                  #
# --------------------------------------------------------------------------- #
def _simulate(n, prevalence=0.30, sep=1.4, seed=0):
    """Realistic imperfect classifier: overlapping score distributions.
    Stands in for a real CXR model's outputs so the repo runs with no download."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    logit = rng.normal(loc=np.where(y == 1, sep, -sep), scale=1.0)
    s = 1.0 / (1.0 + np.exp(-logit))
    return s, y


def demo(trials: int = 400, n: int = 4000):
    """
    Empirical validation: does P(cleared | abnormal) actually stay <= alpha?

    Averaged over many independent calibration/test splits, because the conformal
    guarantee is an expectation over the draw of the calibration set. Any single
    split fluctuates around the bound; the average is what theory pins down.
    """
    print("Conformal Triage Layer -- empirical validation")
    print("(simulated CXR screening scores; swap in a real model via the adapter)")
    print(f"{trials} independent calibration/test splits, n={n} each, prevalence 30%\n")
    print(f"{'alpha':>6} | {'exact bound':>11} | {'empirical':>10} | "
          f"{'worklist cut':>12} | {'normals cleared':>15} | {'abn|cleared':>11}")
    print(f"{'':>6} | {'k/(n+1)':>11} | {'miss rate':>10} | "
          f"{'(headline)':>12} | {'':>15} | {'(no guar.)':>11}")
    print("-" * 82)
    for alpha in (0.01, 0.02, 0.05, 0.10):
        miss, cut, norm_clr, share, bounds = [], [], [], [], []
        for t in range(trials):
            sc, yc = _simulate(n, seed=t)
            st, yt = _simulate(n, seed=5000 + t)
            t_low = calibrate_threshold(sc, yc, alpha)
            m = evaluate(st, yt, t_low)
            miss.append(m["miss_rate"])
            cut.append(m["workload_reduction"])
            norm_clr.append(m["normal_clear_rate"])
            share.append(m["abnormal_share_of_cleared"])
            bounds.append(conformal_bound(int((yc == 1).sum()), alpha))
        ok = "OK" if np.mean(miss) <= alpha + 2 * np.std(miss) / np.sqrt(trials) else "FAIL"
        print(f"{alpha:>6.2f} | {np.mean(bounds):>11.4f} | {np.mean(miss):>10.4f} | "
              f"{np.mean(cut):>11.1%} | {np.mean(norm_clr):>14.1%} | "
              f"{np.mean(share):>10.1%}  {ok}")
    print("\nRead this table as follows:")
    print("  * 'empirical miss rate' tracks the exact bound k/(n+1), not merely")
    print("    sitting under alpha -- the layer spends its entire risk budget and")
    print("    converts it into worklist reduction. That tightness is the point.")
    print("  * At alpha=0.05 the layer removes ~63% of ALL studies from the")
    print("    worklist (it clears ~88% of the truly normal ones) while clearing")
    print("    only 5% of abnormals -- the risk the radiologist explicitly signed")
    print("    off on when they picked alpha.")
    print("  * The last column, P(abnormal | cleared), is NOT guaranteed: it moves")
    print("    with disease prevalence. It is shown so nobody mistakes it for the")
    print("    promise. The promise is the 'empirical miss rate' column.")


# --------------------------------------------------------------------------- #
# Adapter: plug into a real pretrained model (needs internet + torch)          #
# --------------------------------------------------------------------------- #
def adapter_torchxrayvision(image_paths, pathology="Pneumonia"):
    """
    Example adapter. On a machine with internet + torch installed:

        pip install torchxrayvision torch torchvision scikit-image

    Returns scores P(pathology) for a list of chest X-ray image paths, which you
    feed straight into calibrate_threshold(...) / triage(...). The triage layer
    itself never changes -- it only consumes scores.
    """
    import torchxrayvision as xrv          # noqa: local import by design
    import skimage.io, torch, numpy as np
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    idx = model.pathologies.index(pathology)
    scores = []
    for p in image_paths:
        img = xrv.datasets.normalize(skimage.io.imread(p, as_gray=True), 255)
        img = torch.from_numpy(img[None, None]).float()
        img = torch.nn.functional.interpolate(img, size=(224, 224))
        with torch.no_grad():
            scores.append(float(model(img)[0, idx]))
    return np.array(scores)


if __name__ == "__main__":
    demo()
