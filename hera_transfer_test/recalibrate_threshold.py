"""
Threshold recalibration on HERA -- no retraining, no weight changes.

WHY
---
The zero-shot test found the hybrid RANKS HERA pixels well (ROC AUC 0.90) but
SCORES badly at its own threshold (F1 0.14 vs an oracle 0.50). That gap is pure
miscalibration:

    the threshold 0.6113 was chosen on data where 15.08% of pixels are RFI.
    HERA is 2.75% RFI -- five times fewer targets.

At that threshold the model flags almost everything: recall 0.87 but precision
0.077. It is not blind, it is trigger-happy.

WHAT THIS DOES
--------------
Picks a new threshold on HERA's TRAIN split, then applies it ONCE to the TEST
split. The model's weights are never touched -- this changes one number.

That protocol matters. Choosing the threshold on test and reporting the result
would be reporting an oracle, which is not a real score. Train-then-test-once is
the honest version, and it is what the paper should quote.

EXPECTED
--------
F1 should rise from ~0.14 towards the ~0.45-0.50 oracle. It will land slightly
BELOW the oracle -- that difference is the honest cost of not being allowed to
peek at the test set.

USAGE
    source ~/torch-env/bin/activate
    cd "/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project"
    python3 hera_transfer_test/recalibrate_threshold.py

    # try a different normalisation
    python3 hera_transfer_test/recalibrate_threshold.py --norm percentile
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "hybrid_rfi_package"))
sys.path.insert(0, _HERE)
from hybrid_model import HybridRFINet, count_parameters          # noqa: E402
from test_hybrid_on_hera import NORMS                            # same normalisations


@torch.no_grad()
def predict_split(model, split_dir, norm_fn, device, stride=1, tag=""):
    """Run the model over a split. stride>1 subsamples pixels to save memory."""
    imgs = sorted(glob.glob(os.path.join(split_dir, "images", "*.npy")))
    msks = sorted(glob.glob(os.path.join(split_dir, "masks", "*.npy")))
    assert imgs and len(imgs) == len(msks), "no image/mask pairs in " + split_dir
    ys, ps = [], []
    for k, (ip, mp) in enumerate(zip(imgs, msks)):
        a = np.load(ip).astype(np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        x = torch.from_numpy(norm_fn(a).astype(np.float32))[None, None].to(device)
        p = F.softmax(model(x), dim=1)[0, 1].cpu().numpy()
        m = np.load(mp)
        if m.ndim == 3:
            m = m[..., 0]
        # float32, NOT float16. The softmax saturates near 1.0 and that is exactly
        # where the optimal threshold sits; float16's spacing there is 0.000488,
        # which collapses thousands of distinct scores into one value and makes
        # the threshold search coarse. float32 spacing is 6e-08. Memory is fine:
        # 36.7M pixels x 4 bytes = 147 MB.
        ps.append(p.ravel()[::stride].astype(np.float32))
        ys.append((m.astype(bool)).ravel()[::stride])
        if (k + 1) % 100 == 0:
            print(f"    [{tag}] {k+1}/{len(imgs)}", flush=True)
    return np.concatenate(ys), np.concatenate(ps)


def at_threshold(yt, yp, thr):
    pred = yp >= thr
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    P = tp / (tp + fp + 1e-10)
    R = tp / (tp + fn + 1e-10)
    return dict(threshold=float(thr),
                f1=float(2 * P * R / (P + R + 1e-10)),
                precision=float(P), recall=float(R),
                iou=float(tp / (tp + fp + fn + 1e-10)),
                mcc=float(matthews_corrcoef(yt, pred)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def main():
    p = argparse.ArgumentParser(description="Recalibrate the hybrid's threshold on HERA")
    p.add_argument("--data_dir", default=os.path.join(_HERE, "HERA_npy"))
    p.add_argument("--checkpoint", default=os.path.join(_ROOT, "hybrid_run_paperdim", "best.pt"))
    p.add_argument("--norm", default="log_per_image", choices=list(NORMS))
    p.add_argument("--old_threshold", type=float, default=None,
                   help="default: read from hybrid_run_paperdim/progress.json")
    p.add_argument("--train_stride", type=int, default=4,
                   help="subsample every Nth pixel when picking the threshold (memory)")
    p.add_argument("--out", default=os.path.join(_HERE, "results", "hera_recalibrated.json"))
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    targs = ck.get("args", {}) or {}
    model = HybridRFINet(1, 2, base=targs.get("base", 32),
                         depth=targs.get("depth", 4), dropout=0.0).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    old_thr = a.old_threshold
    if old_thr is None:
        pj = os.path.join(_ROOT, "hybrid_run_paperdim", "progress.json")
        old_thr = json.load(open(pj)).get("best_threshold", 0.5) if os.path.exists(pj) else 0.5

    print("=" * 74)
    print(" THRESHOLD RECALIBRATION on HERA  (weights unchanged)")
    print("=" * 74)
    print(f"  device          : {device}")
    print(f"  checkpoint      : epoch {ck.get('epoch','?')}, {count_parameters(model):,} params")
    print(f"  normalisation   : {a.norm}")
    print(f"  OLD threshold   : {old_thr:.4f}   (chosen on our 15.08%-RFI synthetic data)")
    print("=" * 74, flush=True)

    # ---- 1. choose the new threshold on TRAIN only
    print("\n[1/2] scoring HERA TRAIN split to choose a threshold ...", flush=True)
    ytr, ptr = predict_split(model, os.path.join(a.data_dir, "train"),
                             NORMS[a.norm], device, stride=a.train_stride, tag="train")
    print(f"      train RFI fraction: {100*ytr.mean():.4f}%")
    prec, rec, thr = precision_recall_curve(ytr, ptr)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    bi = int(np.argmax(f1))
    new_thr = float(thr[min(bi, len(thr) - 1)])
    print(f"      NEW threshold chosen on TRAIN: {new_thr:.4f}  (train F1 there {f1[bi]:.4f})")
    del ytr, ptr

    # ---- 2. apply it ONCE to TEST
    print("\n[2/2] applying it once to HERA TEST split ...", flush=True)
    yte, pte = predict_split(model, os.path.join(a.data_dir, "test"),
                             NORMS[a.norm], device, stride=1, tag="test")
    print(f"      test predictions: {len(np.unique(pte)):,} distinct values, "
          f"{100*(pte >= 0.9995).mean():.3f}% at >=0.9995 (saturation check)")
    fpr, tpr, _ = roc_curve(yte, pte)
    roc = float(auc(fpr, tpr))
    pr2, rc2, _ = precision_recall_curve(yte, pte)
    prauc = float(auc(rc2, pr2))
    f1t = 2 * pr2 * rc2 / (pr2 + rc2 + 1e-10)
    oracle = float(np.max(f1t))

    before = at_threshold(yte, pte, old_thr)
    after = at_threshold(yte, pte, new_thr)

    print("\n" + "=" * 74)
    print(f"  HERA test set: {len(yte):,} pixels, {100*yte.mean():.4f}% RFI")
    print(f"  ROC AUC {roc:.4f}   PR AUC {prauc:.4f}   (unchanged -- ranking, not threshold)")
    print("-" * 74)
    print(f"  {'':22s}{'threshold':>11}{'F1':>9}{'precision':>11}{'recall':>9}{'IoU':>9}{'MCC':>9}")
    print(f"  {'BEFORE (our thr)':22s}{before['threshold']:>11.4f}{before['f1']:>9.4f}"
          f"{before['precision']:>11.4f}{before['recall']:>9.4f}{before['iou']:>9.4f}{before['mcc']:>9.4f}")
    print(f"  {'AFTER (HERA train)':22s}{after['threshold']:>11.4f}{after['f1']:>9.4f}"
          f"{after['precision']:>11.4f}{after['recall']:>9.4f}{after['iou']:>9.4f}{after['mcc']:>9.4f}")
    print(f"  {'oracle (peeks@test)':22s}{'--':>11}{oracle:>9.4f}   <- upper bound, NOT reportable")
    print("-" * 74)
    gain = after['f1'] - before['f1']
    print(f"  F1 gain from recalibration alone: {gain:+.4f}"
          f"   ({100*gain/max(before['f1'],1e-9):+.0f}%)")
    print(f"  recovered {100*(after['f1']-before['f1'])/max(oracle-before['f1'],1e-9):.0f}% "
          f"of the gap to the oracle")
    print("=" * 74)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(norm=a.norm, roc_auc=roc, pr_auc=prauc, oracle_f1=oracle,
                   before=before, after=after, f1_gain=gain,
                   n_test_pixels=int(yte.size), test_rfi_fraction=float(yte.mean()),
                   checkpoint=os.path.abspath(a.checkpoint),
                   note="threshold chosen on HERA train, applied once to HERA test; "
                        "model weights unchanged"),
              open(a.out, "w"), indent=2)
    print(f"\nSaved: {a.out}")


if __name__ == "__main__":
    main()
