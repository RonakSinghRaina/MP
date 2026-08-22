"""
No-learning reference methods -- the comparison this project is missing.

WHY THIS EXISTS
---------------
`RFI_Project_Model_Comparison.md` compares the hybrid only against tf_unet.
That is not the comparison a referee will ask for. Two things are missing:

  1. A trivial reference. If a single global threshold already scores well,
     any learned model must clear that bar before its score means anything.
     On the seed=42 276x600 dataset a constant threshold reaches
     ROC AUC 0.93 / F1 0.74 -- *above* the tf_unet baseline this project
     reports (ROC AUC 0.67 / F1 0.39). A baseline that loses to a constant
     is a broken run, not an architectural limit, and cannot support a claim
     about architecture.

  2. The methods radio astronomers actually use. Nobody flags RFI with a
     plain U-Net; they use sigma clipping and SumThreshold
     (Offringa et al. 2010, MNRAS 405, 155). Those belong in the table.

Protocol matches `evaluate_hybrid_test.py` exactly: the threshold is chosen on
the VALIDATION split and applied ONCE to test.
"""
import os
import glob
import json
import argparse

import numpy as np
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef)


# ---------------------------------------------------------------------------
# Scoring functions. Each maps an image to a per-pixel "RFI-ness" score.
# ---------------------------------------------------------------------------
def raw_value(img):
    """The floor: the pixel value itself. No normalisation, no parameters."""
    return img.astype(np.float64)


def sigma_clip_score(img, n_iter=5, k=3.0):
    """
    Per-frequency-channel iterative sigma clipping -- the standard first-pass
    flagger. Each channel's clean level and spread are re-estimated using only
    currently-unflagged samples, so RFI stops poisoning its own baseline the
    way a plain median does.
    """
    x = img.astype(np.float64)
    keep = np.ones_like(x, dtype=bool)
    mu = np.zeros((x.shape[0], 1))
    sd = np.ones((x.shape[0], 1))
    for _ in range(n_iter):
        for f in range(x.shape[0]):
            v = x[f][keep[f]]
            if v.size < 10:
                v = x[f]
            mu[f, 0] = np.median(v)
            sd[f, 0] = np.median(np.abs(v - mu[f, 0])) * 1.4826
        sd = np.maximum(sd, 1e-6)
        keep = (x - mu) / sd < k
    return (x - mu) / sd


def sumthreshold_score(img, base_k=3.0):
    """
    SumThreshold-lite (Offringa et al. 2010). After sigma-clip normalisation,
    average the z-map over runs of length M along time and along frequency and
    keep the strongest response. This is the classical way to exploit the fact
    that RFI is *coherent* along an axis -- the same property the hybrid's
    anisotropic strip convolutions are motivated by, so it is the honest
    non-learned comparison for that component.
    """
    z = sigma_clip_score(img, n_iter=3, k=3.0)
    score = z.copy()
    for M in (2, 4, 8, 16, 32):
        thr = base_k / (1.5 ** np.log2(M))
        k1 = np.ones(M) / M
        rt = np.apply_along_axis(lambda r: np.convolve(r, k1, mode="same"), 1, z)
        rf = np.apply_along_axis(lambda c: np.convolve(c, k1, mode="same"), 0, z)
        score = np.maximum(score, np.maximum(rt, rf) * (base_k / max(thr, 1e-6)))
    return score


METHODS = {
    "raw_global_threshold": raw_value,
    "sigma_clip": sigma_clip_score,
    "sumthreshold": sumthreshold_score,
}


# ---------------------------------------------------------------------------
def _score_split(fn, img_files, mask_files):
    ys, ss = [], []
    for ip, mp in zip(img_files, mask_files):
        ss.append(fn(np.load(ip)).ravel())
        ys.append((np.load(mp) == 1).ravel())
    return np.concatenate(ys), np.concatenate(ss)


def run(name, fn, ds, n_val):
    vi = sorted(glob.glob(f"{ds}/val/images/*.npy"))[:n_val]
    vm = sorted(glob.glob(f"{ds}/val/masks/*.npy"))[:n_val]
    ti = sorted(glob.glob(f"{ds}/test/images/*.npy"))
    tm = sorted(glob.glob(f"{ds}/test/masks/*.npy"))
    if not vi or not ti:
        raise SystemExit(f"ERROR: no .npy pairs under {ds}/val or {ds}/test")

    # threshold chosen on VALIDATION only
    vy, vs = _score_split(fn, vi, vm)
    prec, rec, thr = precision_recall_curve(vy, vs)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    bi = int(np.argmax(f1))
    best_thr = float(thr[min(bi, len(thr) - 1)])
    del vy, vs

    # test touched once, at that fixed threshold
    ty, ts = _score_split(fn, ti, tm)
    fpr, tpr, _ = roc_curve(ty, ts)
    p2, r2, _ = precision_recall_curve(ty, ts)
    pred = (ts >= best_thr).astype(np.int8)
    tn, fp, fn_, tp = confusion_matrix(ty, pred, labels=[0, 1]).ravel()
    P = tp / (tp + fp + 1e-10)
    R = tp / (tp + fn_ + 1e-10)
    m = dict(name=name, roc_auc=float(auc(fpr, tpr)), pr_auc=float(auc(r2, p2)),
             threshold=best_thr, f1=float(2 * P * R / (P + R + 1e-10)),
             precision=float(P), recall=float(R),
             iou=float(tp / (tp + fp + fn_ + 1e-10)),
             mcc=float(matthews_corrcoef(ty, pred)),
             tn=int(tn), fp=int(fp), fn=int(fn_), tp=int(tp))
    print(f"\n--- {name} ---")
    print(f"  ROC AUC {m['roc_auc']:.4f}  PR AUC {m['pr_auc']:.4f}  F1 {m['f1']:.4f}  "
          f"P {m['precision']:.4f}  R {m['recall']:.4f}  IoU {m['iou']:.4f}  MCC {m['mcc']:.4f}")
    return m


def main():
    p = argparse.ArgumentParser(description="No-learning RFI-flagging baselines")
    _here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--dataset_dir", default=os.path.normpath(os.path.join(_here, "..", "Synthetic Dataset 276x600")))
    p.add_argument("--n_val", type=int, default=60, help="val images used to pick the threshold")
    p.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = [run(n, METHODS[n], args.dataset_dir, args.n_val) for n in args.methods]
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
