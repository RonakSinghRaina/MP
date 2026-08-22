"""
Final test-set evaluation for HybridRFINet.

USE THIS ONCE, AT THE END.
The threshold is taken from progress.json, where it was chosen on the
VALIDATION set during training -- never tuned on test. That is what makes
this a legitimate held-out result.
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_model import HybridRFINet, count_parameters
from train_hybrid import RFIPatchDataset


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Evaluate HybridRFINet on the test set (once)")
    _here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--dataset_dir", default=os.path.normpath(os.path.join(_here, "..", "Synthetic Dataset")))
    p.add_argument("--output_dir", default=os.path.normpath(os.path.join(_here, "..", "hybrid_run")))
    p.add_argument("--checkpoint", default=None, help="default: <output_dir>/best.pt")
    p.add_argument("--split", default="test")
    # DEFAULT CHANGED (audit): was 512. The reported 276x600 result was produced
    # with --patch_size 0 (whole images), but CLAUDE.md section 12 documents an
    # evaluation command that omits the flag. At the old default of 512 this
    # centre-crops every 276x600 image to 276x512 -- silently discarding 88 of
    # 600 time bins (14.7% of every image) -- and reports numbers that do not
    # match the published ones. Whole images are the only sane default; the
    # value actually used is now recorded in metrics.json.
    p.add_argument("--patch_size", type=int, default=0,
                    help="0 = evaluate on whole images (default, and what the reported "
                         "276x600 numbers used). Any other value centre-crops and will NOT "
                         "reproduce the published result.")
    p.add_argument("--threshold", type=float, default=None,
                    help="default: the validation-selected threshold from progress.json")
    p.add_argument("--per_image_csv", action="store_true",
                    help="also write per-image metrics, so the score can be reported with a "
                         "spread across images instead of one pooled number")
    p.add_argument("--strength_report", action="store_true",
                    help="break recall down by injected RFI strength (needs <split>/strength/, "
                         "written by dataset_generator_v3_strength.py). This is the single most "
                         "informative diagnostic for this task -- report it.")
    args = p.parse_args()

    from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                  confusion_matrix, matthews_corrcoef)

    ckpt_path = args.checkpoint or os.path.join(args.output_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: no checkpoint at {ckpt_path}")
        return

    img_dir = os.path.join(args.dataset_dir, args.split, "images")
    msk_dir = os.path.join(args.dataset_dir, args.split, "masks")
    if not os.path.isdir(img_dir):
        print(f"ERROR: {img_dir} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ck = torch.load(ckpt_path, map_location=device)
    targs = ck.get("args", {})
    model = HybridRFINet(1, 2,
                          base=targs.get("base", 32),
                          depth=targs.get("depth", 4),
                          dropout=0.0).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ck.get('epoch','?')}  "
          f"({count_parameters(model):,} parameters)")

    thr = args.threshold
    if thr is None:
        prog_path = os.path.join(args.output_dir, "progress.json")
        thr = json.load(open(prog_path)).get("best_threshold", 0.5) if os.path.exists(prog_path) else 0.5
        print(f"Using VALIDATION-selected threshold: {thr:.4f} (not tuned on test)")

    # Warn if this evaluation is not being run the way the model was trained.
    train_patch = targs.get("patch_size", None)
    if train_patch is not None and int(train_patch or 0) != int(args.patch_size or 0):
        print(f"WARNING: model was trained with patch_size={train_patch} but this evaluation "
              f"uses patch_size={args.patch_size}. Numbers will not be comparable to the "
              f"training log.")

    ds = RFIPatchDataset(img_dir, msk_dir, patch_size=(args.patch_size or None), random_crop=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    yt, yp, per_image = [], [], []
    for i, (x, y) in enumerate(loader):       # one image at a time (VRAM safe)
        probs = F.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
        t = (y.numpy() == 1).ravel()
        p = probs.ravel()
        yt.append(t); yp.append(p)
        per_image.append((os.path.basename(ds.image_files[i]), t, p))
    yt = np.concatenate(yt); yp = np.concatenate(yp)

    fpr, tpr, _ = roc_curve(yt, yp); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(yt, yp); pr_auc = auc(rec, prec)

    pred = (yp >= thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    mcc = matthews_corrcoef(yt, pred)

    print("\n" + "=" * 46)
    print(f" HybridRFINet -- {args.split} set ({len(ds)} images)")
    print("=" * 46)
    print(f"  ROC AUC      : {roc_auc:.4f}")
    print(f"  PR  AUC      : {pr_auc:.4f}")
    print(f"  F1           : {f1:.4f}   @ threshold {thr:.4f}")
    print(f"  Precision    : {precision:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"  IoU          : {iou:.4f}")
    print(f"  MCC          : {mcc:.4f}")
    print(f"  FPR          : {fp/(fp+tn+1e-10):.4f}   (false flagging of clean data)")
    print(f"  Confusion    : TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print("=" * 46)

    out = os.path.join(args.output_dir, f"eval_{args.split}")
    os.makedirs(out, exist_ok=True)

    payload = {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "f1": float(f1),
               "precision": float(precision), "recall": float(recall), "iou": float(iou),
               "mcc": float(mcc), "threshold": float(thr), "n_images": len(ds),
               "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
               # provenance -- so a reader can tell what actually produced these numbers
               "split": args.split,
               "patch_size": int(args.patch_size or 0),
               "checkpoint": os.path.abspath(ckpt_path),
               "checkpoint_epoch": ck.get("epoch"),
               "train_args": {k: v for k, v in targs.items()} if isinstance(targs, dict) else str(targs),
               "n_parameters": int(count_parameters(model)),
               "threshold_source": "cli" if args.threshold is not None else "progress.json (validation)",
               "torch_version": torch.__version__,
               "device": str(device)}

    # ---- per-image spread: one pooled number hides the variance across images
    if args.per_image_csv:
        import csv
        rows = []
        for name, t, p in per_image:
            pr = (p >= thr)
            itp = int((pr & t).sum()); ifp = int((pr & ~t).sum()); ifn = int((~pr & t).sum())
            P = itp / (itp + ifp + 1e-10); R = itp / (itp + ifn + 1e-10)
            rows.append(dict(image=name, n_rfi=int(t.sum()),
                             precision=P, recall=R,
                             f1=2 * P * R / (P + R + 1e-10),
                             iou=itp / (itp + ifp + ifn + 1e-10)))
        with open(os.path.join(out, "per_image_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        ious = np.array([r["iou"] for r in rows])
        f1s = np.array([r["f1"] for r in rows])
        payload["per_image"] = {"iou_mean": float(ious.mean()), "iou_median": float(np.median(ious)),
                                "iou_std": float(ious.std()),
                                "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
                                "n_images_with_no_rfi": int(sum(r["n_rfi"] == 0 for r in rows))}
        print(f"  per-image IoU: mean {ious.mean():.4f}  median {np.median(ious):.4f}  "
              f"sd {ious.std():.4f}")
        print(f"Saved: {out}/per_image_metrics.csv")

    # ---- recall vs injected RFI strength: the diagnostic that actually shows
    #      whether the model is reading signal amplitude or memorising shapes
    if args.strength_report:
        sdir = os.path.join(args.dataset_dir, args.split, "strength")
        if not os.path.isdir(sdir):
            print(f"  (skipping strength report: {sdir} not found)")
        elif args.patch_size:
            print("  (skipping strength report: only valid at --patch_size 0)")
        else:
            import glob as _glob
            sfiles = sorted(_glob.glob(os.path.join(sdir, "*.npy")))
            strengths = np.concatenate([np.load(f).astype(np.float32).ravel() for f in sfiles])
            if strengths.shape != yt.shape:
                print(f"  (skipping strength report: {strengths.shape} vs {yt.shape})")
            else:
                pred_b = pred.astype(bool)
                bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, np.inf)]
                rep = []
                for lo, hi in bins:
                    sel = yt & (strengths >= lo) & (strengths < hi)
                    n = int(sel.sum())
                    r = float((pred_b & sel).sum() / n) if n else float("nan")
                    lab = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
                    rep.append({"bin_sigma": lab, "n_rfi_px": n, "recall": r})
                    print(f"  strength {lab:>7} sigma : n={n:>10,}  recall={r:.4f}")
                covered = sum(b["n_rfi_px"] for b in rep)
                print(f"  (bins cover {covered:,} of {int(yt.sum()):,} RFI pixels)")
                payload["strength_recall"] = rep
                payload["strength_bins_cover_px"] = int(covered)

    json.dump(payload, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"\nSaved: {out}/metrics.json")


if __name__ == "__main__":
    main()
