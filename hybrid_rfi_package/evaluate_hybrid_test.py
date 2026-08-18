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
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=None,
                    help="default: the validation-selected threshold from progress.json")
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

    ds = RFIPatchDataset(img_dir, msk_dir, patch_size=(args.patch_size or None), random_crop=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    yt, yp = [], []
    for x, y in loader:                       # one image at a time (VRAM safe)
        probs = F.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
        yt.append((y.numpy() == 1).ravel())
        yp.append(probs.ravel())
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
    json.dump({"roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "f1": float(f1),
                "precision": float(precision), "recall": float(recall), "iou": float(iou),
                "mcc": float(mcc), "threshold": float(thr), "n_images": len(ds),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
               open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"\nSaved: {out}/metrics.json")


if __name__ == "__main__":
    main()
