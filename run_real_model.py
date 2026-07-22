"""
Real-model run: score real NIH ChestX-ray14 studies, then calibrate + validate
the conformal triage layer on those scores.

Model : torchxrayvision `densenet121-res224-all` (DenseNet-121, 18 pathologies)
Data  : NIH ChestX-ray14 via HuggingFace `arudaev/chest-xray-14-320`
        (320x320 repackaging of the open NIH release -- no credentialing)
Label : abnormal (1) iff the NIH finding string is anything other than "No Finding"
Score : P(any abnormality) = max over the 14 NIH findings the model predicts

Writes real_scores.npz (scores + labels) so the validation and the HTML demo can
be regenerated without re-running inference.

    python run_real_model.py --n 4000

HONEST CAVEAT, stated up front and repeated in the README:
`densenet121-res224-all` was trained on a mixture that INCLUDES NIH data, so some
of these images were likely seen in training. That makes the absolute
discrimination (AUC) optimistic, and therefore the workload-reduction number
optimistic too. It does NOT threaten the conformal guarantee: validity needs only
that calibration and test studies are exchangeable, which holds because we split
one pool at random. A leaky classifier gets you a better-looking worklist cut; it
cannot make the miss-rate bound fail. That separation -- performance is
model-dependent, validity is not -- is the whole point of the layer.
"""

from __future__ import annotations
import argparse
import numpy as np

# The 14 NIH ChestX-ray14 findings, as named by torchxrayvision.
NIH14 = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
         "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
         "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]


def score_studies(n: int, batch_size: int = 32, seed: int = 0):
    """Stream n studies from NIH ChestX-ray14 and score them with the CXR model."""
    import torch
    import torchxrayvision as xrv
    from datasets import load_dataset

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    cols = [model.pathologies.index(p) for p in NIH14]

    # Streaming: pulls shards lazily, so we never materialize the full dataset.
    ds = load_dataset("arudaev/chest-xray-14-320", split="validation", streaming=True)

    scores, labels, patients, findings, batch = [], [], [], [], []

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch))[:, None].float().to(device)
        with torch.no_grad():
            out = torch.sigmoid(model(x))[:, cols].max(dim=1).values
        scores.extend(out.cpu().numpy().tolist())
        batch.clear()

    for i, row in enumerate(ds):
        if i >= n:
            break
        # xrv expects a single-channel image normalized to [-1024, 1024], 224x224.
        img = np.array(row["image"].convert("L"), dtype=np.float32)
        img = xrv.datasets.normalize(img, 255)
        img = np.array(
            torch.nn.functional.interpolate(
                torch.from_numpy(img)[None, None], size=(224, 224),
                mode="bilinear", align_corners=False)[0, 0])
        batch.append(img)
        labels.append(0 if row["labels"].strip() == "No Finding" else 1)
        # NIH filenames are PATIENTID_STUDYINDEX.png. We keep the patient id because
        # one patient contributes up to 13 studies here, and splitting calibration
        # from test at the STUDY level would put near-duplicate images of the same
        # chest on both sides -- leakage that silently inflates every result.
        patients.append(row["filename"].split("_")[0])
        findings.append(row["labels"].strip())   # for per-finding safety audit
        if len(batch) == batch_size:
            flush()
            print(f"  scored {len(scores)}/{n}", end="\r", flush=True)
    flush()
    print(f"  scored {len(scores)}/{n}      ")
    k = len(scores)
    return (np.array(scores), np.array(labels[:k]), np.array(patients[:k]),
            np.array(findings[:k], dtype=object))


def auc(scores, labels):
    """Rank-based AUROC, no sklearn dependency."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties so the AUC is exact for duplicated scores
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = (labels == 1).sum(), (labels == 0).sum()
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="studies to score")
    ap.add_argument("--out", default="real_scores.npz")
    args = ap.parse_args()

    s, y, pat, fnd = score_studies(args.n)
    np.savez(args.out, scores=s, labels=y, patients=pat, findings=fnd)
    print(f"\nsaved {args.out}: {len(s)} studies from {len(np.unique(pat))} patients, "
          f"prevalence {y.mean():.1%}, AUROC {auc(s, y):.3f}")


if __name__ == "__main__":
    main()
