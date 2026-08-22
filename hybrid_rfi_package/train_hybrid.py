"""
Trains HybridRFINet on the synthetic RFI dataset.

NOTE ON THE DEFAULTS BELOW (corrected in audit)
------------------------------------------------
Items 1 and 2 describe the ORIGINAL 1024x1024 work and are NOT how the reported
276x600 result was produced. That run used `--batch_size 8 --patch_size 0`
(whole images) on the 276x600 dataset, for 22 epochs of a planned 40, stopped
by hand. The argparse defaults in this file (`patch_size=512`, `batch_size=1`,
`dataset_dir=../Synthetic Dataset`) are still the 1024x1024 ones, so running
this script with no flags does NOT reproduce anything that was published --
pass the flags shown in README.md.

EVERY FIX FROM THE BASELINE WORK IS PRESERVED HERE
----------------------------------------------------
1. batch_size = 1, patch_size = 512   -- fits a 6 GB RTX 3060 Laptop GPU.
2. Same 1024x1024 dataset, same .npy pipeline, unchanged.
3. Min-max normalization WITHOUT np.fabs()  -- the baseline's own
   normalization took absolute values first, which turned a strong NEGATIVE
   noise dip into a near-maximum-brightness pixel indistinguishable from
   real RFI while the mask still said "clean".
4. Adam @ lr=1e-3, NOT momentum @ 0.2 -- the paper's lr killed the baseline
   at iteration 4 (measured).
5. Class weighting computed from the ACTUAL dataset (~15% RFI pixels).
6. Validation-only model selection. The TEST SET IS NEVER TOUCHED here.
7. Resumability across laptop restarts via progress.json (true cumulative
   epochs, not restarting the count).
8. Best-checkpoint tracking separate from last-checkpoint.
9. VRAM preflight before committing to a long run.
10. Evaluation runs one image at a time (batching eval caused an OOM crash
    in the baseline even though training itself fit fine).
11. Collapse detection.
12. GroupNorm instead of BatchNorm -- batch_size=1 makes BatchNorm
    statistics pure noise.

LOSS
-----
Weighted cross-entropy + Dice. CE alone is dominated by the ~85% clean
pixels; Dice directly optimises overlap of the minority RFI class and is
insensitive to the background pixel count. The sum keeps CE's stable
per-pixel gradients while Dice pushes up recall on sparse RFI.
"""
import os
import sys
import glob
import json
import time
import shutil
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_model import HybridRFINet, count_parameters  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RFIPatchDataset(Dataset):
    def __init__(self, images_dir, masks_dir, patch_size=512, random_crop=True, seed=None):
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.npy")))
        assert len(self.image_files) == len(self.mask_files) and self.image_files, (
            f"No matching .npy pairs in {images_dir} / {masks_dir}")
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.rng = np.random.RandomState(seed)
        print(f"RFIPatchDataset: {len(self.image_files)} pairs from {images_dir}"
              + (f" ({patch_size}x{patch_size} patches)" if patch_size else " (full images)"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = np.load(self.image_files[idx]).astype(np.float32)
        mask = np.load(self.mask_files[idx]).astype(np.int64)

        if self.patch_size:
            p = self.patch_size
            ny, nx = img.shape
            if self.random_crop:
                y = self.rng.randint(0, max(1, ny - p + 1))
                x = self.rng.randint(0, max(1, nx - p + 1))
            else:
                y = max(0, (ny - p) // 2)   # deterministic centre crop for val
                x = max(0, (nx - p) // 2)
            img = img[y:y + p, x:x + p]
            mask = mask[y:y + p, x:x + p]

        # Min-max to [0,1]. NO fabs() -- negative values are real noise dips
        # and must stay on the low end, not get folded up to look like RFI.
        img = img - img.min()
        mx = img.max()
        if mx > 0:
            img = img / mx

        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def dice_loss(logits, target, eps=1.0):
    probs = F.softmax(logits, dim=1)[:, 1]
    t = (target == 1).float()
    inter = (probs * t).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class CEDiceLoss(nn.Module):
    def __init__(self, class_weights, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice_weight * dice_loss(logits, target)


# ---------------------------------------------------------------------------
# Helpers carried over from the baseline script
# ---------------------------------------------------------------------------
def compute_class_weights(dataset_dir, split="train"):
    meta = os.path.join(dataset_dir, split, "metadata.jsonl")
    if not os.path.exists(meta):
        print(f"WARNING: {meta} missing -- using unweighted loss.")
        return [1.0, 1.0]
    fr = [json.loads(l)["rfi_fraction"] for l in open(meta)]
    mean_rfi = float(np.mean(fr))
    w = [0.5 / (1 - mean_rfi), 0.5 / mean_rfi]
    print(f"Measured class balance: {mean_rfi*100:.1f}% RFI pixels -> class_weights = "
          f"[{w[0]:.3f}, {w[1]:.3f}]")
    return w


def load_progress(d):
    p = os.path.join(d, "progress.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"epochs_completed": 0, "best_f1": -1.0, "best_epoch": None}


def save_progress(d, prog):
    json.dump(prog, open(os.path.join(d, "progress.json"), "w"), indent=2)


@torch.no_grad()
def evaluate(model, loader, device, max_images=None):
    """One image at a time -- batching eval caused an OOM in the baseline."""
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    model.eval()
    yt, yp = [], []
    for i, (x, y) in enumerate(loader):
        if max_images and i >= max_images:
            break
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        yt.append((y.numpy() == 1).ravel())
        yp.append(probs.ravel())
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)

    collapsed = bool(np.allclose(yp, yp[0], atol=1e-6))
    if len(np.unique(yt)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "max_f1": 0.0, "best_threshold": 0.5, "collapsed": collapsed}

    fpr, tpr, _ = roc_curve(yt, yp)
    prec, rec, thr = precision_recall_curve(yt, yp)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    bi = int(np.argmax(f1))
    # threshold chosen on VALIDATION only -- never on test.
    #
    # precision_recall_curve returns prec/rec of length n+1 but thr of length n:
    # the extra point is the degenerate (precision=1, recall=0) endpoint, which
    # has no threshold. If argmax lands there, the old code silently substituted
    # 0.5 and reported it as "the validation-selected threshold" -- a number that
    # was never selected by anything. Clamp to the last real threshold and say so.
    if bi < len(thr):
        best_thr = float(thr[bi])
    else:
        best_thr = float(thr[-1]) if len(thr) else 0.5
        print(f"  WARNING: best F1 fell on the degenerate endpoint of the PR curve; "
              f"clamping threshold to {best_thr:.4f}. This usually means the model is "
              f"predicting almost nothing as RFI -- check before trusting this epoch.")
    return {"roc_auc": float(auc(fpr, tpr)), "pr_auc": float(auc(rec, prec)),
            "max_f1": float(f1[bi]), "best_threshold": best_thr, "collapsed": collapsed}


def preflight_vram(model, device, h, w, batch_size, dropout_ok=True):
    print(f"\nVRAM preflight: {h}x{w}, batch_size={batch_size} ...")
    try:
        model.train()
        x = torch.randn(batch_size, 1, h, w, device=device)
        y = torch.zeros(batch_size, h, w, dtype=torch.long, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.zero_grad(set_to_none=True)
        del x, y, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("VRAM check passed.\n")
        return True
    except torch.cuda.OutOfMemoryError:
        print(f"\n!! OUT OF GPU MEMORY at {h}x{w}, batch_size={batch_size} !!")
        if batch_size > 1:
            print(f"  try: --batch_size {max(1, batch_size // 2)}   <-- try this first")
        print(f"  try: --patch_size {max(128, min(h, w) // 2)}   (train on smaller crops instead of full images)")
        print(f"  try: --base 16   (smaller network)")
        return False


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train HybridRFINet on the synthetic RFI dataset")
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--dataset_dir", default=os.path.normpath(os.path.join(_here, "..", "Synthetic Dataset")))
    parser.add_argument("--output_dir", default=os.path.normpath(os.path.join(_here, "..", "hybrid_run")))
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--base", type=int, default=32, help="Base feature width")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--total_epochs", type=int, default=40)
    parser.add_argument("--epochs_per_chunk", type=int, default=2)
    parser.add_argument("--n_val_images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_vram_check", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true",
                        help="run on CPU without the interactive prompt (needed for "
                             "unattended/scripted runs, where input() would hang forever)")
    parser.add_argument("--deterministic", action="store_true",
                        help="make the run bit-reproducible: seeds cuDNN, disables "
                             "nondeterministic kernels. Slower, but required if you want the "
                             "published numbers to be exactly regenerable.")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="stop after N consecutive evaluations with no improvement in val "
                             "F1 (0 = disabled). Use this instead of stopping a run by hand: "
                             "the reported run was halted manually at epoch 22 of a planned 40, "
                             "which is not a criterion anyone can reproduce.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        # Without this, cuDNN picks convolution algorithms nondeterministically and
        # two runs of the same command give slightly different numbers. That is
        # tolerable for a demo and not tolerable for a published table.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"NOTE: could not enable fully deterministic algorithms: {e}")
        print("Deterministic mode ON.")

    if not torch.cuda.is_available():
        print("!! NO GPU DETECTED !! This will be extremely slow on CPU.")
        if not args.allow_cpu:
            if not sys.stdin.isatty():
                print("Not running interactively and --allow_cpu was not passed. Aborting "
                      "rather than hanging on a prompt nobody can answer.")
                return
            if input("Continue on CPU anyway? [y/N]: ").strip().lower() != "y":
                return
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"GPU detected: {torch.cuda.get_device_name(device)}")

    train_img = os.path.join(args.dataset_dir, "train", "images")
    val_img = os.path.join(args.dataset_dir, "val", "images")
    if not os.path.isdir(train_img):
        print(f"ERROR: {train_img} not found.")
        return
    if not os.path.isdir(val_img):
        print(f"\nERROR: {val_img} not found.")
        print("This model must NOT be selected using the test set. Create a validation")
        print("split first (takes seconds, does not touch test/):")
        print(f"  python3 make_val_split.py --dataset_dir \"{args.dataset_dir}\"")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_last = os.path.join(args.output_dir, "last.pt")
    ckpt_best = os.path.join(args.output_dir, "best.pt")
    log_path = os.path.join(args.output_dir, "training_log.csv")

    progress = load_progress(args.output_dir)
    if progress["epochs_completed"] >= args.total_epochs:
        print(f"Already completed {progress['epochs_completed']}/{args.total_epochs} epochs.")
        return

    train_ds = RFIPatchDataset(train_img, os.path.join(args.dataset_dir, "train", "masks"),
                                patch_size=args.patch_size, random_crop=True, seed=args.seed)
    val_ds = RFIPatchDataset(val_img, os.path.join(args.dataset_dir, "val", "masks"),
                              patch_size=args.patch_size, random_crop=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = HybridRFINet(1, 2, base=args.base, depth=args.depth, dropout=args.dropout).to(device)
    print(f"HybridRFINet parameters: {count_parameters(model):,}")

    if not args.skip_vram_check and device.type == "cuda":
        if args.patch_size > 0:
            probe_h = probe_w = args.patch_size
        else:
            # patch_size=0 means "use full images" -- probe with the REAL
            # dataset image dimensions, not 0. An earlier version of this
            # script passed patch_size (=0) straight into the preflight,
            # which built a 0x0 tensor and crashed instead of testing memory.
            sample = np.load(train_ds.image_files[0])
            probe_h, probe_w = sample.shape[:2]
        if not preflight_vram(model, device, probe_h, probe_w, args.batch_size):
            return

    cw = compute_class_weights(args.dataset_dir, "train")
    criterion = CEDiceLoss(cw).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)

    if progress["epochs_completed"] > 0 and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        print(f"Resumed at epoch {progress['epochs_completed']} "
              f"(best F1={progress['best_f1']:.4f} @ epoch {progress['best_epoch']})")

    # Record the exact configuration next to the outputs. Without this, the only
    # record of how a result was produced is whatever the write-up remembers.
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "n_train_images": len(train_ds),
                   "n_val_images_available": len(val_ds),
                   "steps_per_epoch": (len(train_ds) + args.batch_size - 1) // args.batch_size,
                   "n_parameters": count_parameters(model),
                   "class_weights": cw, "device": str(device),
                   "torch_version": torch.__version__}, f, indent=2)

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            # NOTE: one row per CHUNK, not per epoch. `epoch` is the cumulative
            # epoch count reached, and `train_loss` is the mean loss over the
            # whole chunk (epochs_per_chunk epochs), not over the single epoch
            # named in the first column. With the default epochs_per_chunk=2 a
            # 22-epoch run produces 11 rows. Do not plot this as a per-epoch
            # curve without saying so.
            f.write("epoch,train_loss_chunk_mean,epochs_in_chunk,roc_auc,pr_auc,"
                    "max_f1,best_threshold,is_best,secs\n")

    stale = 0
    while progress["epochs_completed"] < args.total_epochs:
        chunk = min(args.epochs_per_chunk, args.total_epochs - progress["epochs_completed"])
        print(f"\n=== Epochs {progress['epochs_completed']+1}-"
              f"{progress['epochs_completed']+chunk} of {args.total_epochs} ===")
        t0 = time.time()

        model.train()
        running = 0.0
        n_steps = 0
        for _ in range(chunk):
            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # guards the instability we saw
                optimizer.step()
                running += loss.item()
                n_steps += 1
                if (step + 1) % 100 == 0:
                    print(f"  step {step+1}/{len(train_loader)}  loss={loss.item():.4f}")
            scheduler.step()

        train_loss = running / max(1, n_steps)
        m = evaluate(model, val_loader, device, max_images=args.n_val_images)
        progress["epochs_completed"] += chunk
        secs = time.time() - t0

        if m["collapsed"]:
            print("\n!! COLLAPSE DETECTED -- model outputs a constant value everywhere. Stopping.")
            print(f"   Try a lower --learning_rate (currently {args.learning_rate}).")
            return

        is_best = m["max_f1"] > progress["best_f1"]
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": progress["epochs_completed"],
                    "args": vars(args)}, ckpt_last)
        if is_best:
            progress["best_f1"] = m["max_f1"]
            progress["best_epoch"] = progress["epochs_completed"]
            progress["best_threshold"] = m["best_threshold"]
            shutil.copyfile(ckpt_last, ckpt_best)

        save_progress(args.output_dir, progress)
        with open(log_path, "a") as f:
            f.write(f"{progress['epochs_completed']},{train_loss:.4f},{chunk},{m['roc_auc']:.4f},"
                    f"{m['pr_auc']:.4f},{m['max_f1']:.4f},{m['best_threshold']:.4f},"
                    f"{int(is_best)},{secs:.1f}\n")

        print(f"Epoch {progress['epochs_completed']}: loss={train_loss:.4f} (mean over {chunk} epochs)  "
              f"ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  F1={m['max_f1']:.4f}"
              f"{'  <-- NEW BEST' if is_best else ''}  ({secs:.0f}s)")

        # Pre-declared stopping criterion. The published run was stopped by hand
        # at epoch 22 of 40 "once gains had flattened" -- a judgement call nobody
        # else can reproduce, and one that also fixes the epoch budget the
        # baseline was then matched to. Set --early_stop_patience so the stopping
        # rule is part of the configuration instead of part of the operator.
        stale = 0 if is_best else stale + 1
        if args.early_stop_patience and stale >= args.early_stop_patience:
            print(f"\nEarly stop: {stale} consecutive evaluations without improvement "
                  f"(--early_stop_patience {args.early_stop_patience}).")
            break

    print(f"\nDone. Best val F1={progress['best_f1']:.4f} @ epoch {progress['best_epoch']}")
    print(f"NOTE: best val F1 above is the ORACLE (threshold-tuned) F1 on validation. "
          f"It is not directly comparable to a fixed-threshold test F1 -- see AUDIT_REPORT.md.")
    print(f"Best model: {ckpt_best}")
    print(f"Validation-selected threshold: {progress.get('best_threshold', 0.5):.4f}")
    print("The TEST set has not been touched. Evaluate on it once, at the very end.")


if __name__ == "__main__":
    main()