"""
Matched-budget architecture ablation.

WHY THIS EXISTS
---------------
The project's headline claim is that the reported gap (tf_unet F1 0.39 vs
hybrid F1 0.98) is "attributable to architectural limitations of the plain
U-Net rather than training configuration". Nothing in the repository tests
that. The two models differ in architecture AND framework AND loss AND
normalisation AND padding AND output activation AND batch size AND width, all
changed together.

This script changes ONE thing at a time. Every variant gets:
  - identical data and identical iteration order (same seed)
  - identical loss (the project's own weighted CE + Dice, imported from
    train_hybrid.py so it cannot drift)
  - identical optimiser (Adam) and LR schedule
  - identical number of gradient steps
  - identical evaluation protocol: threshold picked on validation, applied
    once to test

Only the architecture changes. That is the experiment.

BUDGET
------
Defaults are reduced (base=16, 400 training images, 200 steps) so the whole
grid runs on CPU in a few hours. Absolute F1 will therefore be below the
full-budget 0.98 -- the ORDERING and the GAPS are what this measures, and they
are measured under a budget that is identical for every row.

For the publication table, rerun on the GPU at the real budget:
    --base 32 --n_train 700 --batch 8 --steps 1936
and repeat with --seed 0 1 2 to get error bars (see --seeds).
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "hybrid_rfi_package")))

from train_hybrid import RFIPatchDataset, CEDiceLoss          # noqa: E402  the project's own code
from hybrid_model import count_parameters                      # noqa: E402
from models_ablation import build, ALL_NAMES                   # noqa: E402


class _Head(Dataset):
    """First n items of a dataset, without shuffling the underlying file order."""

    def __init__(self, ds, n):
        self.ds, self.n = ds, min(n, len(ds))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.ds[i]


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        p = F.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
        yt.append((y.numpy() == 1).ravel())
        yp.append(p.ravel())
    return np.concatenate(yt), np.concatenate(yp)


def load_strength(dataset_dir, split="test"):
    """
    Per-pixel injected RFI amplitude in units of local noise sigma, flattened to
    match the concatenated test labels. Returns None if not available.
    """
    import glob as _glob
    sdir = os.path.join(dataset_dir, split, "strength")
    files = sorted(_glob.glob(os.path.join(sdir, "*.npy")))
    if not files:
        return None
    return np.concatenate([np.load(f).astype(np.float32).ravel() for f in files])


STRENGTH_BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, np.inf)]


def evaluate(model, val_loader, test_loader, device, strength=None):
    """Threshold on val, apply once to test -- same protocol as evaluate_hybrid_test.py."""
    vy, vp = _collect(model, val_loader, device)
    prec, rec, thr = precision_recall_curve(vy, vp)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    bi = int(np.argmax(f1))
    best_thr = float(thr[min(bi, len(thr) - 1)])
    val_oracle = float(f1[bi])
    del vy, vp

    ty, tp_ = _collect(model, test_loader, device)
    fpr, tpr, _ = roc_curve(ty, tp_)
    p2, r2, _ = precision_recall_curve(ty, tp_)
    pred = (tp_ >= best_thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(ty, pred, labels=[0, 1]).ravel()
    P = tp / (tp + fp + 1e-10)
    R = tp / (tp + fn + 1e-10)
    out = dict(roc_auc=float(auc(fpr, tpr)), pr_auc=float(auc(r2, p2)),
               threshold=best_thr, val_oracle_f1=val_oracle,
               f1=float(2 * P * R / (P + R + 1e-10)),
               precision=float(P), recall=float(R),
               iou=float(tp / (tp + fp + fn + 1e-10)),
               mcc=float(matthews_corrcoef(ty, pred)),
               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

    # Recall stratified by injected RFI strength. This is the precise test of
    # the strip convolutions' stated purpose: hybrid_model.py argues they exist
    # so that "a 1.5-sigma line becomes detectable" by integrating along
    # coherent structure. If that mechanism is real, use_strip=True should beat
    # use_strip=False specifically in the weakest bins. Pooled F1 cannot show
    # this, because the weak bins are a small share of the pixels.
    if strength is not None and strength.shape == ty.shape:
        pb = pred.astype(bool)
        rep = []
        for lo, hi in STRENGTH_BINS:
            sel = ty & (strength >= lo) & (strength < hi)
            n = int(sel.sum())
            rep.append({"bin_sigma": (f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"),
                        "n_rfi_px": n,
                        "recall": float((pb & sel).sum() / n) if n else float("nan")})
        out["strength_recall"] = rep
    return out


def train_one(name, a, loaders, cw, seed, device, strength=None):
    train_loader, val_loader, test_loader = loaders
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(name, base=a.base, depth=a.depth, dropout=a.dropout).to(device)
    n_par = count_parameters(model)
    crit = CEDiceLoss(cw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    print(f"\n=== {name}  seed={seed}  ({n_par:,} params) ===", flush=True)
    t0, step, losses = time.time(), 0, []
    model.train()
    while step < a.steps:
        for x, y in train_loader:
            if step >= a.steps:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.detach().item())
            step += 1
            if step % 25 == 0:
                print(f"  step {step}/{a.steps} loss={np.mean(losses[-25:]):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    secs = time.time() - t0
    m = evaluate(model, val_loader, test_loader, device, strength=strength)
    m.update(name=name, seed=seed, params=n_par, train_secs=secs,
             final_train_loss=float(np.mean(losses[-25:])))
    print(f"  -> ROC {m['roc_auc']:.4f}  F1 {m['f1']:.4f}  P {m['precision']:.4f}  "
          f"R {m['recall']:.4f}  MCC {m['mcc']:.4f}   ({secs / 60:.1f} min)", flush=True)
    if "strength_recall" in m:
        print("     recall by injected strength: " +
              "  ".join(f"{b['bin_sigma']}s={b['recall']:.3f}" for b in m["strength_recall"]),
              flush=True)
    return m


def main():
    p = argparse.ArgumentParser(description="Matched-budget architecture ablation")
    p.add_argument("--dataset_dir", default=os.path.normpath(os.path.join(_HERE, "..", "Synthetic Dataset 276x600")))
    p.add_argument("--out", default=os.path.join(_HERE, "..", "results", "ablation.json"))
    p.add_argument("--variants", nargs="*", default=ALL_NAMES, choices=ALL_NAMES)
    p.add_argument("--seeds", type=int, nargs="*", default=[42],
                   help="run each variant at each seed; >=3 gives error bars")
    p.add_argument("--base", type=int, default=16, help="feature width (32 = the published model)")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_val", type=int, default=80)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=200, help="identical gradient-step budget for every variant")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    meta = os.path.join(a.dataset_dir, "train", "metadata.jsonl")
    if not os.path.exists(meta):
        raise SystemExit(f"ERROR: {meta} not found. Generate the dataset first.")
    fr = [json.loads(l)["rfi_fraction"] for l in open(meta)]
    mean_rfi = float(np.mean(fr))
    cw = [0.5 / (1 - mean_rfi), 0.5 / mean_rfi]
    print(f"class weights {cw}  (mean train RFI {mean_rfi * 100:.2f}%)")

    tr = RFIPatchDataset(f"{a.dataset_dir}/train/images", f"{a.dataset_dir}/train/masks",
                         patch_size=0, random_crop=False, seed=a.seeds[0])
    va = RFIPatchDataset(f"{a.dataset_dir}/val/images", f"{a.dataset_dir}/val/masks",
                         patch_size=0, random_crop=False, seed=0)
    te = RFIPatchDataset(f"{a.dataset_dir}/test/images", f"{a.dataset_dir}/test/masks",
                         patch_size=0, random_crop=False, seed=0)
    loaders = (DataLoader(_Head(tr, a.n_train), batch_size=a.batch, shuffle=True, num_workers=0),
               DataLoader(_Head(va, a.n_val), batch_size=1, shuffle=False, num_workers=0),
               DataLoader(te, batch_size=1, shuffle=False, num_workers=0))

    strength = load_strength(a.dataset_dir, "test")
    if strength is None:
        print("NOTE: no test/strength/ maps found -- skipping the strength-stratified "
              "recall breakdown. Regenerate with dataset_generator_v3_strength.py to get it.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    results = []
    for seed in a.seeds:
        for name in a.variants:
            try:
                results.append(train_one(name, a, loaders, cw, seed, device, strength=strength))
            except Exception as e:                                   # keep the grid going
                print(f"  !! {name} (seed {seed}) failed: {e}", flush=True)
            json.dump({"config": vars(a), "results": results}, open(a.out, "w"), indent=2)

    print(f"\n{'variant':<16}{'params':>11}{'ROC':>9}{'F1':>8}{'P':>8}{'R':>8}{'MCC':>8}")
    for m in results:
        print(f"{m['name']:<16}{m['params']:>11,}{m['roc_auc']:>9.4f}{m['f1']:>8.4f}"
              f"{m['precision']:>8.4f}{m['recall']:>8.4f}{m['mcc']:>8.4f}")
    print(f"\nSaved: {a.out}")


if __name__ == "__main__":
    main()
