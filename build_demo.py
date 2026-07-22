"""
Generate index.html -- the self-contained interactive demo.

Embeds the real model scores from real_scores.npz directly into the page so the
demo is a single file with no server, no build step, and no network access. Falls
back to simulated scores if real_scores.npz is absent, so the repo always
produces a working demo.

    python build_demo.py
"""

from __future__ import annotations
import json
import os
import numpy as np

TEMPLATE = open(os.path.join(os.path.dirname(__file__), "demo_template.html")).read()


def load_data():
    if os.path.exists("real_scores.npz"):
        d = np.load("real_scores.npz")
        s, y, pat = d["scores"], d["labels"], d["patients"]
        src = "real"
    else:
        from conformal_triage import _simulate
        s, y = _simulate(8000, seed=1)
        pat = np.arange(len(s)).astype(str)   # simulated: every study its own patient
        src = "simulated"

    # Order studies by a shuffled PATIENT ranking, so that the demo's first-half /
    # second-half split is patient-disjoint by construction. A plain study-level
    # shuffle would leak ~30% of test patients into calibration (NIH has up to 13
    # studies per patient), which is the same bug validate_real.py fixes.
    rng = np.random.default_rng(7)
    uniq = np.unique(pat)
    rng.shuffle(uniq)
    order = {p: i for i, p in enumerate(uniq)}
    perm = np.argsort(np.array([order[p] for p in pat]), kind="stable")
    return s[perm], y[perm], pat[perm], src


def main():
    s, y, pat, src = load_data()
    from run_real_model import auc
    meta = {
        "source": src,
        "n": int(len(s)),
        "n_patients": int(len(np.unique(pat))),
        "prevalence": float(y.mean()),
        "auroc": float(auc(s, y)),
        "model": ("torchxrayvision densenet121-res224-all" if src == "real"
                  else "simulated classifier"),
        "dataset": ("NIH ChestX-ray14 (HuggingFace arudaev/chest-xray-14-320)"
                    if src == "real" else "synthetic screening cohort"),
    }
    # 6dp, not 4dp. At 4dp these 8000 real scores collapsed to ~1500 distinct
    # values with tie clusters up to 82 studies wide. Because the threshold rule is
    # a strict `<`, a whole tie cluster clears or doesn't as a block, so the demo's
    # numbers drifted from validate_real.py's. 6dp keeps all scores distinct for
    # ~10KB more page weight.
    _, codes = np.unique(pat, return_inverse=True)
    payload = {
        "meta": meta,
        "scores": [round(float(v), 6) for v in s],
        "labels": [int(v) for v in y],
        "groups": [int(g) for g in codes],   # patient id, for patient-disjoint splits
    }
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    with open("index.html", "w") as f:
        f.write(html)
    kb = os.path.getsize("index.html") / 1024
    print(f"wrote index.html ({kb:.0f} KB) from {src} scores: "
          f"n={meta['n']}, prevalence={meta['prevalence']:.1%}, AUROC={meta['auroc']:.3f}")


if __name__ == "__main__":
    main()
