"""
How many parameters does this task actually need?

WHY THIS EXISTS
---------------
MARS (arXiv:2608.05546, 2026) reaches F1 0.978 on an RFI segmentation task with
270,769 parameters. HybridRFINet reaches 0.9808 with 9,304,186 -- 34x more for
roughly the same score. A reviewer will ask what the extra 9 million are doing.

There is already an internal hint: the matched-budget ablation found every U-Net
variant landing between F1 0.8785 and 0.9117, and the BEST variant was the full
model with ECA removed. Removing things did not hurt.

This script measures the accuracy-versus-size curve directly. It trains the SAME
HybridRFINet at several widths under an IDENTICAL protocol and reports F1 against
parameter count and inference speed.

WHAT IS HELD FIXED (so only width varies)
-----------------------------------------
  dataset          Synthetic Dataset 276x600, seed 42, 700/150/150
  images           full 276x600 (patch_size=0), no cropping
  loss             the project's own CEDiceLoss, imported from train_hybrid.py
  class weights    measured from train/metadata.jsonl (0.5848 / 3.4485)
  optimiser        Adam @ 1e-3, CosineAnnealingLR(T_max=epochs)
  epochs           22   (matches the published hybrid run)
  batch size       8    (matches the published hybrid run)
  depth            4
  dropout          0.2
  seed             42, reset identically before every width
  protocol         best checkpoint by val F1; threshold chosen on val;
                   test touched ONCE with that fixed threshold

Only --base changes. base=32 is the published model and acts as the control:
it should land near F1 0.98, which also confirms the harness is faithful.

NOTE ON COMPARABILITY
---------------------
The published run used CosineAnnealingLR(T_max=40) but stopped at epoch 22, so
its learning-rate schedule differs slightly from a clean T_max=22 run. That is
why base=32 is retrained here rather than reusing the stored 0.9808 -- every row
in the output table must come from the same harness to be comparable.

SIZES (measured, not estimated -- depth 4)
-----------------------------------------
  base  4  ->     152,582 params    61x smaller than the published model
  base  8  ->     593,842 params    15.7x
  base 16  ->   2,342,474 params     4.0x
  base 24  ->   5,245,922 params     1.8x
  base 32  ->   9,304,186 params     1.0x  <- the published model, exactly
  MARS reference:  270,769 params, F1 0.978

Widths must be divisible by 8 (or be 1, 2 or 4) because the model uses
GroupNorm(min(8, C), C). base 12 and base 20 cannot be built; the script
refuses them up front rather than crashing partway through.

READING THE RESULT
------------------
  small widths hold F1  ->  the model is oversized; report the efficiency finding
  F1 falls off sharply  ->  the capacity is genuinely needed; also worth knowing

RUNTIME -- READ THIS BEFORE STARTING
------------------------------------
The published base-32 run logged ~3086 s per 2-epoch chunk, so 22 epochs took
roughly 9 hours on this machine. base 16 and base 8 are cheaper but not
proportionally so (the strip convolutions are depthwise, so their cost falls
linearly with width, not quadratically). Budget for an overnight job.

The widths run in the order given, cheapest first, and each one's result is
written the moment it finishes. So you can start the full sweep and stop after
whichever width you like -- nothing is lost. If you only have one night, run
    --base 8
on its own: it is the single most informative run, because if 15.7x fewer
parameters already hold the F1, the answer is settled.

USAGE
    source ~/torch-env/bin/activate
    cd "/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project"
    python3 experiments/width_sweep/run_width_sweep.py

    # one width at a time, or a different grid
    python3 experiments/width_sweep/run_width_sweep.py --base 8
    python3 experiments/width_sweep/run_width_sweep.py --base 4 8 16 24 32

Resumable: rerun the same command and finished widths are skipped. Add --fresh to
force a redo. If VRAM is tight use --batch 4 (but then use it for EVERY width).
"""
import os
import sys
import json
import time
import glob
import shutil
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "hybrid_rfi_package"))

from hybrid_model import HybridRFINet, count_parameters          # noqa: E402
from train_hybrid import RFIPatchDataset, CEDiceLoss             # noqa: E402  project's own code


# ---------------------------------------------------------------------------
def class_weights(dataset_dir):
    """Exactly as train_hybrid.compute_class_weights does it."""
    meta = os.path.join(dataset_dir, "train", "metadata.jsonl")
    if not os.path.exists(meta):
        raise SystemExit("ERROR: {} not found.".format(meta))
    fr = [json.loads(l)["rfi_fraction"] for l in open(meta)]
    m = float(np.mean(fr))
    return [0.5 / (1 - m), 0.5 / m], m


@torch.no_grad()
def collect(model, loader, device):
    """Full-precision probabilities. float32, never float16 -- the softmax
    saturates near 1.0 and float16's spacing there is 0.000488, which would
    make the threshold search coarse exactly where the optimum sits."""
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        p = F.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
        yt.append((y.numpy() == 1).ravel())
        yp.append(p.ravel().astype(np.float32))
    return np.concatenate(yt), np.concatenate(yp)


@torch.no_grad()
def throughput(model, device, h=276, w=600, warmup=3, iters=15):
    """Inference images/sec at batch 1 -- the practical case for a live flagger."""
    model.eval()
    x = torch.randn(1, 1, h, w, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return iters / (time.time() - t0)


def evaluate(model, val_loader, test_loader, device):
    """Threshold on val, applied ONCE to test. Same protocol as evaluate_hybrid_test.py."""
    vy, vp = collect(model, val_loader, device)
    prec, rec, thr = precision_recall_curve(vy, vp)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    bi = int(np.argmax(f1))
    best_thr = float(thr[min(bi, len(thr) - 1)])
    val_f1 = float(f1[bi])
    del vy, vp

    ty, tp_ = collect(model, test_loader, device)
    fpr, tpr, _ = roc_curve(ty, tp_)
    p2, r2, _ = precision_recall_curve(ty, tp_)
    f1c = 2 * p2 * r2 / (p2 + r2 + 1e-10)
    pred = tp_ >= best_thr
    tn, fp, fn, tp = confusion_matrix(ty, pred, labels=[0, 1]).ravel()
    P = tp / (tp + fp + 1e-10)
    R = tp / (tp + fn + 1e-10)
    return dict(roc_auc=float(auc(fpr, tpr)), pr_auc=float(auc(r2, p2)),
                f1=float(2 * P * R / (P + R + 1e-10)),
                precision=float(P), recall=float(R),
                iou=float(tp / (tp + fp + fn + 1e-10)),
                mcc=float(matthews_corrcoef(ty, pred)),
                oracle_f1=float(np.max(f1c)), threshold=best_thr,
                val_oracle_f1=val_f1,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


# ---------------------------------------------------------------------------
def train_one_width(base, a, cw, device):
    out_dir = os.path.join(a.out_root, "base{:02d}".format(base))
    done = os.path.join(out_dir, "eval_test", "metrics.json")
    if os.path.exists(done) and not a.fresh:
        print("  base={} already done -- skipping (use --fresh to redo)".format(base), flush=True)
        return json.load(open(done))
    if a.fresh:
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # identical starting conditions for every width
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)

    train_ds = RFIPatchDataset(os.path.join(a.dataset_dir, "train", "images"),
                               os.path.join(a.dataset_dir, "train", "masks"),
                               patch_size=0, random_crop=False, seed=a.seed)
    val_ds = RFIPatchDataset(os.path.join(a.dataset_dir, "val", "images"),
                             os.path.join(a.dataset_dir, "val", "masks"),
                             patch_size=0, random_crop=False, seed=0)
    test_ds = RFIPatchDataset(os.path.join(a.dataset_dir, "test", "images"),
                              os.path.join(a.dataset_dir, "test", "masks"),
                              patch_size=0, random_crop=False, seed=0)
    tl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    sl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = HybridRFINet(1, 2, base=base, depth=a.depth, dropout=a.dropout).to(device)
    n_par = count_parameters(model)
    crit = CEDiceLoss(cw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    pfile = os.path.join(out_dir, "progress.json")
    prog = json.load(open(pfile)) if os.path.exists(pfile) else \
        {"epochs_completed": 0, "best_f1": -1.0, "best_epoch": None}
    last = os.path.join(out_dir, "last.pt")
    if prog["epochs_completed"] > 0 and os.path.exists(last):
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        # Restore the shuffling RNG as well. Without this, a resumed run resets
        # the seed to 42 at process start and therefore shuffles differently
        # from epoch N onward than an uninterrupted run would -- a small effect,
        # but it makes "seed 42" stop meaning one fixed thing.
        _rng = st.get("rng")
        if _rng:
            _t = _rng["torch"]
            torch.set_rng_state(_t.cpu() if hasattr(_t, "cpu") else _t)
            np.random.set_state(_rng["numpy"])
            if device.type == "cuda" and "cuda" in _rng:
                try:
                    torch.cuda.set_rng_state_all([x.cpu() for x in _rng["cuda"]])
                except Exception as e:
                    print("  note: CUDA RNG not restored ({})".format(e), flush=True)
            print("  RNG restored -- batch order continues as if never interrupted", flush=True)
        else:
            print("  NOTE: this checkpoint predates RNG saving, so the batch order from "
                  "here differs slightly from an uninterrupted run.", flush=True)
        print("  resuming base={} from epoch {}".format(base, prog["epochs_completed"]), flush=True)

    steps = (len(train_ds) + a.batch - 1) // a.batch
    print("\n" + "=" * 78)
    print(" WIDTH base={}   {:,} parameters   ({:.1f} MB)".format(base, n_par, n_par * 4 / 1e6))
    print("   {} train / {} val / {} test | batch {} | {} steps/epoch | {} epochs".format(
        len(train_ds), len(val_ds), len(test_ds), a.batch, steps, a.epochs))
    print("=" * 78, flush=True)

    csv = os.path.join(out_dir, "training_log.csv")
    if not os.path.exists(csv):
        open(csv, "w").write("epoch,train_loss,val_oracle_f1,is_best,minutes\n")

    ep0 = prog["epochs_completed"]
    t_start = time.time()
    while prog["epochs_completed"] < a.epochs:
        model.train()
        losses = []
        t0 = time.time()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.detach().item())
        sched.step()
        prog["epochs_completed"] += 1
        mins = (time.time() - t0) / 60.0

        vy, vp = collect(model, vl, device)
        pr, rc, th = precision_recall_curve(vy, vp)
        f1 = 2 * pr * rc / (pr + rc + 1e-10)
        vf1 = float(np.max(f1))
        del vy, vp
        is_best = vf1 > prog["best_f1"]
        rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "rng": rng,
                    "epoch": prog["epochs_completed"]}, last)
        if is_best:
            prog["best_f1"] = vf1
            prog["best_epoch"] = prog["epochs_completed"]
            shutil.copyfile(last, os.path.join(out_dir, "best.pt"))
        json.dump(prog, open(pfile, "w"), indent=2)
        open(csv, "a").write("{},{:.4f},{:.4f},{},{:.1f}\n".format(
            prog["epochs_completed"], np.mean(losses), vf1, int(is_best), mins))

        elapsed = (time.time() - t_start) / 60.0
        eta = elapsed / max(prog["epochs_completed"] - ep0, 1) * (a.epochs - prog["epochs_completed"])
        b = int(24 * prog["epochs_completed"] / a.epochs)
        print("  [{}{}] Epoch {}/{}  loss {:.4f}  |  val F1 {:.4f}{}  |  {:.1f}m done, ~{:.0f}m left".format(
            "#" * b, "." * (24 - b), prog["epochs_completed"], a.epochs,
            np.mean(losses), vf1, "  <-- BEST" if is_best else "",
            (time.time() - t_start) / 60.0, eta), flush=True)

    train_min = (time.time() - t_start) / 60.0
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                     map_location=device, weights_only=False)["model"])
    ips = throughput(model, device)
    m = evaluate(model, vl, sl, device)
    m.update(base=base, depth=a.depth, params=int(n_par),
             params_millions=round(n_par / 1e6, 4),
             model_mb=round(n_par * 4 / 1e6, 2),
             images_per_sec=round(ips, 2),
             epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed,
             train_minutes=round(train_min, 1),
             best_val_f1=prog["best_f1"], best_epoch=prog["best_epoch"])
    os.makedirs(os.path.join(out_dir, "eval_test"), exist_ok=True)
    json.dump(m, open(done, "w"), indent=2)
    print("  -> params {:,} | ROC {:.4f} | F1 {:.4f} | {:.1f} img/s | {:.1f} min".format(
        n_par, m["roc_auc"], m["f1"], ips, train_min), flush=True)
    del model, opt, sched, tl, vl, sl
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return m


def main():
    p = argparse.ArgumentParser(description="Accuracy vs parameter-count sweep for HybridRFINet")
    p.add_argument("--dataset_dir", default=os.path.join(_ROOT, "Synthetic Dataset 276x600"))
    p.add_argument("--out_root", default=os.path.join(_ROOT, "hybrid_run_width_sweep"))
    p.add_argument("--base", type=int, nargs="+", default=[8, 16, 32],
                   help="widths to try. 32 is the published model and acts as the control.")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=22, help="matches the published hybrid run")
    p.add_argument("--batch", type=int, default=8, help="matches the published hybrid run")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()

    # GroupNorm(min(8, C), C) means every channel count must be divisible by 8
    # (or be 1/2/4). Catch this now with a clear message instead of a crash
    # eight hours into the sweep.
    bad = [b for b in a.base if not (b in (1, 2, 4) or b % 8 == 0)]
    if bad:
        raise SystemExit(
            "ERROR: width(s) {} cannot be built.\n"
            "  Every channel count must be divisible by 8 (or be 1, 2 or 4).\n"
            "  Verified-good widths: 4, 8, 16, 24, 32, 64.".format(bad))

    if not os.path.isdir(os.path.join(a.dataset_dir, "train", "images")):
        raise SystemExit("ERROR: dataset not found at {}".format(a.dataset_dir))

    # A width's output folder is named by width alone, NOT by seed. A second
    # seed written into the same --out_root would find the first seed's
    # metrics.json, skip the width, and silently report the OLD seed's numbers
    # as if they were the new seed's. Refuse that outright.
    _default_root = os.path.join(_ROOT, "hybrid_run_width_sweep")
    if a.seed != 42 and os.path.abspath(a.out_root) == os.path.abspath(_default_root):
        raise SystemExit(
            "ERROR: seed {} would be written into the seed-42 folder.\n"
            "  Widths already finished there would be SKIPPED and their seed-42\n"
            "  numbers reported as if they were seed {}.\n"
            "  Give this seed its own folder, e.g.:\n"
            "    --seed {} --out_root hybrid_run_width_sweep_seed{}".format(
                a.seed, a.seed, a.seed, a.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cw, mean_rfi = class_weights(a.dataset_dir)
    os.makedirs(a.out_root, exist_ok=True)

    print("=" * 78)
    print(" ACCURACY vs PARAMETER COUNT -- HybridRFINet width sweep")
    print("=" * 78)
    print("  device        : {}".format(device))
    print("  dataset       : {}".format(a.dataset_dir))
    print("  widths        : {}".format(a.base))
    print("  held fixed    : depth {} | dropout {} | epochs {} | batch {} | Adam {} | seed {}".format(
        a.depth, a.dropout, a.epochs, a.batch, a.lr, a.seed))
    print("  class weights : [{:.4f}, {:.4f}]  (train RFI {:.2f}%)".format(cw[0], cw[1], 100 * mean_rfi))
    print("  reference     : published HybridRFINet = 9,304,186 params, F1 0.9808")
    print("                  MARS (arXiv 2608.05546) =   270,769 params, F1 0.978")
    print("=" * 78, flush=True)

    print("  sizes to be trained:")
    for b in a.base:
        _m = HybridRFINet(1, 2, base=b, depth=a.depth, dropout=a.dropout)
        _n = count_parameters(_m)
        print("    base {:>2}  ->  {:>10,} params  ({:>6.2f} MB)".format(
            b, _n, _n * 4 / 1e6))
        del _m
    print("=" * 78, flush=True)

    results = []
    for b in a.base:
        try:
            results.append(train_one_width(b, a, cw, device))
        except torch.cuda.OutOfMemoryError:
            print("  !! OUT OF VRAM at base={}. Rerun the WHOLE sweep with a smaller "
                  "--batch so every width is comparable.".format(b), flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print("  !! base={} failed: {}: {}".format(b, type(e).__name__, e), flush=True)
        json.dump(results, open(os.path.join(a.out_root, "sweep_results.json"), "w"), indent=2)

    if not results:
        print("\nNo widths completed.")
        return
    results.sort(key=lambda r: r["params"])
    print("\n" + "=" * 78)
    print(" RESULT -- accuracy against size")
    print("=" * 78)
    print("  {:>5} {:>12} {:>9} {:>9} {:>9} {:>9} {:>10}".format(
        "base", "params", "MB", "ROC AUC", "F1", "IoU", "img/s"))
    for r in results:
        print("  {:>5} {:>12,} {:>9.1f} {:>9.4f} {:>9.4f} {:>9.4f} {:>10.1f}".format(
            r["base"], r["params"], r["model_mb"], r["roc_auc"], r["f1"], r["iou"],
            r["images_per_sec"]))
    big = max(results, key=lambda r: r["params"])
    print("-" * 78)
    for r in results:
        if r is big:
            continue
        print("  base {:>2} vs base {:>2}: {:.1f}x fewer parameters for {:+.4f} F1  ({:.1f}x faster)".format(
            r["base"], big["base"], big["params"] / r["params"], r["f1"] - big["f1"],
            r["images_per_sec"] / max(big["images_per_sec"], 1e-9)))
    print("=" * 78)
    print("\n  READ: if a small width holds F1 within ~0.01 of base 32, the published")
    print("        model is oversized for this task and that is a reportable finding.")
    print("        If F1 falls away sharply, the capacity is genuinely required.")
    print("\n  CAVEAT: one seed per width. Differences below ~0.01 F1 are not established.")
    print("          Repeat with --seed 0 and --seed 1 before quoting exact numbers.")
    print("\nSaved: {}".format(os.path.join(a.out_root, "sweep_results.json")))


if __name__ == "__main__":
    main()
