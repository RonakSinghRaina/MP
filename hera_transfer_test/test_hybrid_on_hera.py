"""
ZERO-SHOT TRANSFER TEST: does the hybrid, trained only on our synthetic data,
detect RFI in HERA data it has never seen?

No training. Inference only. The model is used exactly as saved.

WHY THIS IS WORTH RUNNING
-------------------------
Our synthetic benchmark is easy: the label is a noise-free threshold
(injected_RFI > 0.5*sigma), 90.9% of the RFI sits at or above the noise level, and
a constant global threshold alone scores F1 0.74 on it. So F1 0.98 there is not by
itself evidence of a strong detector.

HERA is a completely different simulator, built by a different group. If the model
transfers, our benchmark was capturing something real. If it does not, that is
also a genuine, reportable result -- and the honest counterweight to a 0.98.

HOW THE TWO DATASETS DIFFER (measured, not assumed)
---------------------------------------------------
                        our synthetic 276x600      HERA 512x512
    image size          276 x 600                  512 x 512
    value range         -85 .. 222                 0.000015 .. 415
    negative values     many                       NONE
    dynamic range       ~300, roughly linear       7 orders of magnitude (log-like)
    RFI fraction        15.08%                     2.77%   (5.5x rarer)
    test images         150                        140

Those are large gaps. In particular the model has never seen strictly-positive
data spanning seven decades.

WHY SEVERAL NORMALISATIONS ARE TESTED
-------------------------------------
This project already proved (BASELINE_FAILURE_DECOMPOSITION.md) that the wrong
input scaling can cost 0.213 F1 on its own. So running a single normalisation here
would be unsafe: a bad score might mean "the model cannot transfer" OR merely
"we fed it badly". Testing four removes that ambiguity. Report the best one, and
say which it was.

HOW TO READ THE OUTPUT
----------------------
ROC AUC is the number that matters most. It is threshold-free, so it answers
"does the model RANK RFI pixels above clean ones?" independently of calibration.

    ROC high + F1 low   -> the model SEES the RFI; only the threshold is wrong.
                           Fixable, and a genuinely positive transfer result.
    ROC low  + F1 low   -> the model is blind to HERA's RFI. Real negative result.
    ROC ~0.5            -> no better than guessing.

Two reference points are printed alongside, using the identical protocol:
a constant global threshold, and the same on log-scaled values. If the model
cannot beat a constant threshold on HERA, it has not transferred.

USAGE
    source ~/torch-env/bin/activate
    cd "/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project"
    python3 hera_transfer_test/test_hybrid_on_hera.py
"""
import os, sys, json, pickle, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "hybrid_rfi_package"))
from hybrid_model import HybridRFINet, count_parameters


# ---------------------------------------------------------------------------
# Normalisations. Each maps one raw HERA image to something the model can read.
# ---------------------------------------------------------------------------
def n_perimage(a):
    """Exactly what the model was trained with. The faithful zero-shot test."""
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


def n_log_perimage(a):
    """log10 first. Visibility amplitudes span decades; this is standard practice."""
    a = np.log10(np.maximum(a, 1e-6))
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


def n_percentile(a):
    """Robust clip to p1..p99, then scale. Immune to a single hot pixel."""
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def n_log_percentile(a):
    """log10 + robust clip. Usually the best choice for this kind of data."""
    a = np.log10(np.maximum(a, 1e-6))
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


NORMS = {"per_image": n_perimage, "log_per_image": n_log_perimage,
         "percentile": n_percentile, "log_percentile": n_log_percentile}


def metrics(yt, yp, fixed_thr=None):
    fpr, tpr, _ = roc_curve(yt, yp)
    roc = float(auc(fpr, tpr))
    pr, rc, _ = precision_recall_curve(yt, yp)
    prauc = float(auc(rc, pr))
    f1c = 2 * pr * rc / (pr + rc + 1e-10)
    out = dict(roc_auc=roc, pr_auc=prauc, max_f1=float(np.max(f1c)))
    if fixed_thr is not None:
        pred = (yp >= fixed_thr).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        P = tp / (tp + fp + 1e-10); R = tp / (tp + fn + 1e-10)
        out.update(f1_at_train_thr=float(2 * P * R / (P + R + 1e-10)),
                   precision_at_train_thr=float(P), recall_at_train_thr=float(R),
                   train_thr=float(fixed_thr))
    return out


def main():
    p = argparse.ArgumentParser(description="Zero-shot: hybrid model on HERA")
    p.add_argument("--pkl", default=os.path.join(_ROOT, "HERA_04-03-2022_all.pkl"))
    p.add_argument("--checkpoint", default=os.path.join(_ROOT, "hybrid_run_paperdim", "best.pt"))
    p.add_argument("--progress", default=os.path.join(_ROOT, "hybrid_run_paperdim", "progress.json"))
    p.add_argument("--out_dir", default=os.path.join(_HERE, "results"))
    p.add_argument("--n_images", type=int, default=0, help="0 = all 140 test images")
    p.add_argument("--norms", nargs="*", default=list(NORMS), choices=list(NORMS))
    p.add_argument("--export_npy", action="store_true",
                   help="also write HERA out as .npy in the train/val/test layout, "
                        "so it can be used for TRAINING later")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 74)
    print(" ZERO-SHOT TRANSFER: hybrid (trained on synthetic 276x600) -> HERA 512x512")
    print("=" * 74)
    print(f"  device      : {dev}")

    print(f"  loading     : {os.path.basename(a.pkl)} ...", flush=True)
    with open(a.pkl, "rb") as f:
        tr_x, tr_y, te_x, te_y = pickle.load(f)
    n = a.n_images if a.n_images > 0 else te_x.shape[0]
    te_x, te_y = te_x[:n], te_y[:n]
    print(f"  test images : {n} of {te_x.shape[1]}x{te_x.shape[2]}")
    print(f"  RFI fraction: {100*te_y.astype(bool).mean():.4f}%  "
          f"(our synthetic test set is 15.08%)")

    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    targs = ck.get("args", {}) or {}
    model = HybridRFINet(1, 2, base=targs.get("base", 32),
                         depth=targs.get("depth", 4), dropout=0.0).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  checkpoint  : epoch {ck.get('epoch','?')}, {count_parameters(model):,} params")

    train_thr = 0.5
    if os.path.exists(a.progress):
        train_thr = json.load(open(a.progress)).get("best_threshold", 0.5)
    print(f"  its threshold from OUR data : {train_thr:.4f}  (applied unchanged -- true zero-shot)")
    print("=" * 74, flush=True)

    yt = te_y.astype(bool).reshape(n, -1)
    results = {}

    for nm in a.norms:
        fn = NORMS[nm]
        preds = np.empty((n, te_x.shape[1] * te_x.shape[2]), dtype=np.float32)
        with torch.no_grad():
            for i in range(n):
                img = fn(te_x[i, :, :, 0].astype(np.float32))
                t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(dev)
                pr = F.softmax(model(t), dim=1)[0, 1].cpu().numpy()
                preds[i] = pr.ravel()
                if (i + 1) % 50 == 0:
                    print(f"    [{nm}] {i+1}/{n}", flush=True)
        m = metrics(yt.ravel(), preds.ravel(), fixed_thr=train_thr)
        m["pred_min"], m["pred_max"] = float(preds.min()), float(preds.max())
        m["pred_std"] = float(preds.std())
        results["hybrid_" + nm] = m
        print(f"\n  --- hybrid, {nm} ---")
        print(f"      ROC AUC {m['roc_auc']:.4f}   PR AUC {m['pr_auc']:.4f}")
        print(f"      F1 at its own threshold {m['f1_at_train_thr']:.4f}   "
              f"(P {m['precision_at_train_thr']:.3f} / R {m['recall_at_train_thr']:.3f})")
        print(f"      best-possible F1 (oracle) {m['max_f1']:.4f}")
        print(f"      output range [{m['pred_min']:.3f}, {m['pred_max']:.3f}] std {m['pred_std']:.4f}",
              flush=True)
        del preds

    # ---- reference points, identical protocol, no learning at all
    print("\n  --- reference points (no model) ---", flush=True)
    raw = te_x[:, :, :, 0].astype(np.float32).reshape(n, -1)
    for nm, sc in [("raw value", raw), ("log10 value", np.log10(np.maximum(raw, 1e-6)))]:
        m = metrics(yt.ravel(), sc.ravel())
        results["threshold_" + nm.replace(" ", "_")] = m
        print(f"      constant threshold on {nm:12s}: ROC {m['roc_auc']:.4f}  "
              f"best-possible F1 {m['max_f1']:.4f}")

    os.makedirs(a.out_dir, exist_ok=True)
    op = os.path.join(a.out_dir, "hera_zeroshot_metrics.json")
    json.dump({"n_test_images": int(n),
               "hera_rfi_fraction": float(te_y.astype(bool).mean()),
               "checkpoint": os.path.abspath(a.checkpoint),
               "train_threshold": float(train_thr),
               "results": results}, open(op, "w"), indent=2)

    best = max((k for k in results if k.startswith("hybrid_")),
               key=lambda k: results[k]["roc_auc"])
    br = results[best]
    thr_roc = max(results[k]["roc_auc"] for k in results if k.startswith("threshold_"))
    print("\n" + "=" * 74)
    print(f"  BEST hybrid setting : {best.replace('hybrid_','')}")
    print(f"    ROC AUC {br['roc_auc']:.4f}  |  F1(own thr) {br['f1_at_train_thr']:.4f}  "
          f"|  oracle F1 {br['max_f1']:.4f}")
    print(f"  best no-model reference ROC AUC: {thr_roc:.4f}")
    print()
    if br["roc_auc"] < 0.6:
        print("  READ: ROC near 0.5 -- the model is essentially blind to HERA's RFI.")
        print("        A real negative transfer result. Report it honestly.")
    elif br["roc_auc"] < thr_roc:
        print("  READ: the model does NOT beat a constant threshold on HERA.")
        print("        It has not transferred in any useful sense.")
    elif br["f1_at_train_thr"] < 0.5 * br["max_f1"]:
        print("  READ: ROC is decent but F1 at its own threshold is far below the")
        print("        oracle -- the model SEES the RFI, its calibration is just wrong")
        print("        for this data. That is a POSITIVE transfer result: fine-tuning")
        print("        or recalibrating the threshold should recover most of it.")
    else:
        print("  READ: the model transfers. Strong result -- worth a section in the paper.")
    print("=" * 74)
    print(f"\nSaved: {op}")

    if a.export_npy:
        base = os.path.join(_HERE, "HERA_npy")
        print(f"\nExporting .npy layout to {base} (for training later) ...", flush=True)
        for split, X, Y in [("train", tr_x, tr_y), ("test", te_x, te_y)]:
            di, dm = os.path.join(base, split, "images"), os.path.join(base, split, "masks")
            os.makedirs(di, exist_ok=True); os.makedirs(dm, exist_ok=True)
            for i in range(X.shape[0]):
                np.save(os.path.join(di, f"spectrogram_{i:04d}.npy"),
                        X[i, :, :, 0].astype(np.float32))
                np.save(os.path.join(dm, f"mask_{i:04d}.npy"),
                        Y[i, :, :, 0].astype(np.uint8))
            print(f"  {split}: {X.shape[0]} pairs")


if __name__ == "__main__":
    main()
