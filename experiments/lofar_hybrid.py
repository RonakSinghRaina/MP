"""
The HybridRFINet on the REAL LOFAR dataset.

Same protocol as experiments/lofar_tfunet_baseline.py so the two numbers are
directly comparable, and comparable to Mesarcik et al. Table 2:

  train on   : the 6621 CLEAN training images (leaked + dead baselines removed)
  test on    : the 109 expert-labelled images (data[3], human ground truth)
  threshold  : selected on VALIDATION, applied to test -- never chosen on test
  headline   : pooled pixel-wise F1 over all 109 images
  guard      : every train/val image is fingerprinted against all 109 test
               images before training starts; the run aborts if one matches

ARCHITECTURE -- base 8, the project's own efficiency conclusion (PART 6)
------------------------------------------------------------------------
The width sweep on synthetic data:

  base  4    152,582 params   F1 0.9547   -0.0265  <- genuinely breaks
  base  8    593,842 params   F1 0.9749   -0.0063  <- DEFAULT here
  base 16  2,342,474 params   F1 0.9788   -0.0024
  base 32  9,304,186 params   F1 0.9812      --

PART 6's conclusion is that roughly 8.7M of base 32's 9.3M parameters buy
about half a percent of F1, and that the paper should be reframed around
that. So base 8 is the default: 15.7x fewer parameters for -0.0063 F1, a gap
inside the single-seed noise floor. base 4 is NOT viable -- the capacity floor
sits between 152k and 594k parameters.

Whether that efficiency finding *transfers to real data* is untested and is
worth a table row on its own: run --base 8 and --base 32 and compare. Real
LOFAR is a much harder task and may well have a different capacity floor.

BUDGET -- matched to the hybrid's own synthetic run
---------------------------------------------------
The synthetic run was 700 images/epoch x 40 epochs at batch 1 = 28,000
gradient steps. Defaults here reproduce exactly that: --iters_per_epoch 700,
--total_epochs 40, --batch_size 1. LOFAR has 6621 training images, so these
28,000 steps are drawn at random from all of them (~4.2 passes).

OUTPUT SIZE -- note when comparing to tf_unet
---------------------------------------------
This model uses same padding and returns 512x512. tf_unet uses valid padding
and returns 472x472. Both are reported: `pooled_f1` over the full 512x512, and
`pooled_f1_crop472` over the matching centre crop, which is the number to line
up against PART 12. RFI density differs by only 0.4% between the two regions
(PART 12.7), so the two rarely disagree much.

NORMALISATION
-------------
Default is `fixed`, because on full 512x512 LOFAR images fixed range beat
per-image by 0.5021 vs 0.2456 for tf_unet (PART 12.11). BUT the hybrid has
GroupNorm, which PART 1 argued is exactly why per-image normalisation does not
hurt it the way it hurt tf_unet -- so `--norm per_image` is the interesting
second arm here, not a formality.

CLASS WEIGHTS
-------------
Unlike tf_unet (where weighting was lethal via the ReLU trap, PART 1), the
hybrid is *designed* around weighted CE + Dice. Weights default to inverse
frequency measured on the actual training split. On LOFAR that is roughly
1:65, far more extreme than synthetic's ~1:6, so --class_weight_cap is
provided and --no_class_weight disables it entirely.

USAGE
-----
    ~/torch-env/bin/python experiments/lofar_hybrid.py

    # the published width, for the efficiency comparison
    ~/torch-env/bin/python experiments/lofar_hybrid.py --base 32 \
        --output_dir runs/lofar/hybrid_b32_fixed_seed0

RESUMING
    Re-run the identical command; it reads progress.json and continues.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_curve

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "hybrid_rfi_package"))
sys.path.insert(0, _ROOT)

from hybrid_model import HybridRFINet          # noqa: E402
from lofar_data import load_lofar, preprocess  # noqa: E402


# ---------------------------------------------------------------------------
def assert_no_leakage(train_images, test_images, indices, label):
    """All 109 test images are byte-identical to 109 training images (PART 11.5).
    Relying on the index file being right is an assumption; this is a check."""
    t0 = time.time()
    fp = {}
    for t in range(test_images.shape[0]):
        fp.setdefault(float(test_images[t, :, :, 0].sum(dtype=np.float64)), []).append(t)
    leaked = []
    for i in np.asarray(indices):
        i = int(i)
        f = float(train_images[i, :, :, 0].sum(dtype=np.float64))
        for t in fp.get(f, ()):
            if np.array_equal(train_images[i], test_images[t]):
                leaked.append((i, t))
                break
    if leaked:
        raise SystemExit(
            "\nABORTED -- TEST SET LEAKAGE in the {} split.\n{} image(s) are "
            "byte-identical to test images, e.g. {}\n".format(label, len(leaked), leaked[:5]))
    print("  leakage check     : {} clean, 0 of {} match a test image  ({:.1f}s)".format(
        label, len(indices), time.time() - t0), flush=True)


def normalise(img, mode, lo, hi):
    if mode == "per_image":
        return preprocess(img)                      # clip 20 sigma -> log -> min-max
    d = np.asarray(img, dtype=np.float64)
    if mode == "fixed_log":
        d = np.log(np.maximum(d, 1e-6))
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def calibrate(images, indices, mode, n=200, seed=0):
    rng = np.random.default_rng(seed)
    sel = rng.choice(indices, size=min(n, len(indices)), replace=False)
    los, his = [], []
    for i in sel:
        a = images[int(i), :, :, 0].astype(np.float64)
        if mode == "fixed_log":
            a = np.log(np.maximum(a, 1e-6))
        los.append(np.percentile(a, 0.5))
        his.append(np.percentile(a, 99.5))
    return float(np.mean(los)), float(np.mean(his))


def dice_loss(logits, target, eps=1.0):
    probs = F.softmax(logits, dim=1)[:, 1]
    t = (target == 1).float()
    inter = (probs * t).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


# ---------------------------------------------------------------------------
def batches(images, masks, indices, bs, mode, lo, hi, rng, n_steps):
    """Yield n_steps random batches, reading only what is needed from the memmap."""
    for _ in range(n_steps):
        sel = rng.choice(indices, size=bs, replace=False)
        X = np.stack([normalise(images[int(i), :, :, 0], mode, lo, hi) for i in sel])
        Y = np.stack([masks[int(i), :, :, 0] for i in sel]).astype(np.int64)
        yield (torch.from_numpy(X).unsqueeze(1).float(), torch.from_numpy(Y))


@torch.no_grad()
def predict_scores(model, images, masks, indices, mode, lo, hi, device, bs=2):
    model.eval()
    yt, yp = [], []
    idx = np.asarray(indices)
    for s in range(0, len(idx), bs):
        sel = idx[s:s + bs]
        X = np.stack([normalise(images[int(i), :, :, 0], mode, lo, hi) for i in sel])
        x = torch.from_numpy(X).unsqueeze(1).float().to(device)
        p = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        yp.append(p.reshape(len(sel), -1))
        yt.append(np.stack([masks[int(i), :, :, 0] for i in sel]).reshape(len(sel), -1))
    model.train()
    return np.concatenate(yt), np.concatenate(yp)      # (N, 512*512) each


def metrics_from(yt2d, yp2d, threshold, crop=None):
    """crop: if given, restrict to a centre crop of that size before scoring."""
    if crop:
        n = yt2d.shape[0]
        a = yt2d.reshape(n, 512, 512); b = yp2d.reshape(n, 512, 512)
        c = (512 - crop) // 2
        yt = a[:, c:512 - c, c:512 - c].ravel().astype(bool)
        yp = b[:, c:512 - c, c:512 - c].ravel()
    else:
        yt = yt2d.ravel().astype(bool); yp = yp2d.ravel()
    p = yp > threshold
    tp = int((p & yt).sum()); fp = int((p & ~yt).sum())
    fn = int((~p & yt).sum()); tn = int((~p & ~yt).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-12)
    fpr, tpr, _ = roc_curve(yt, yp)
    pc, rcv, _ = precision_recall_curve(yt, yp)
    f1c = 2 * pc * rcv / (pc + rcv + 1e-10)
    return dict(pooled_f1=float(f1), pooled_precision=float(pr), pooled_recall=float(rc),
                TP=tp, FP=fp, FN=fn, TN=tn, max_f1=float(np.max(f1c)),
                roc_auc=float(auc(fpr, tpr)), pr_auc=float(auc(rcv, pc)),
                threshold=float(threshold), positive_fraction=float(yt.mean()))


def best_threshold(yt2d, yp2d):
    yt = yt2d.ravel().astype(bool); yp = yp2d.ravel()
    pc, rc, th = precision_recall_curve(yt, yp)
    f1 = 2 * pc * rc / (pc + rc + 1e-10)
    k = int(np.argmax(f1[:-1])) if len(th) else 0
    return (float(th[k]) if len(th) else 0.5), float(f1[k])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=int, default=8,
                    help="8 = PART 6's efficiency pick (593,842 params) and the default. "
                         "32 = the published width (9,304,186). 4 genuinely breaks -- "
                         "PART 6 measured -0.0265 F1 there.")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--norm", choices=["fixed", "fixed_log", "per_image"], default="fixed")
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="1 matches the synthetic run. Measured on the 6 GB card: "
                         "base 8 fits batch 8 (2.9 GB); base 32 fits batch 2 (3.0 GB).")
    ap.add_argument("--total_epochs", type=int, default=40)
    ap.add_argument("--iters_per_epoch", type=int, default=700,
                    help="700x40 = 28,000 steps, identical to the synthetic run")
    ap.add_argument("--epochs_per_chunk", type=int, default=2)
    ap.add_argument("--n_val_images", type=int, default=150)
    ap.add_argument("--dice_weight", type=float, default=1.0)
    ap.add_argument("--class_weight_cap", type=float, default=0.0,
                    help="cap the RFI-class weight (0 = uncapped inverse frequency)")
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default=None)
    a = ap.parse_args()

    out = a.output_dir or os.path.join(_ROOT, "runs", "lofar",
                                       "hybrid_base{}_{}_seed{}".format(a.base, a.norm, a.seed))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = load_lofar()
    rng = np.random.default_rng(a.seed)
    idx = d.clean_train_idx.copy()
    rng.shuffle(idx)
    if a.limit_train:
        idx = idx[:a.limit_train]
    n_val = max(1, int(len(idx) * a.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = HybridRFINet(1, 2, base=a.base, depth=a.depth, dropout=a.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    print("=" * 74)
    print("  HybridRFINet on REAL LOFAR  |  base {}  |  norm = {}".format(a.base, a.norm))
    print("=" * 74)
    print("  parameters        : {:,}".format(n_par))
    print("  device            : {}".format(device))
    print("  train / val / test: {} / {} / 109".format(len(tr_idx), len(val_idx)))
    print("  steps/epoch       : {} x {} epochs = {:,} gradient steps".format(
        a.iters_per_epoch, a.total_epochs, a.iters_per_epoch * a.total_epochs))
    print("                      (synthetic run was 700 x 40 = 28,000)")

    assert_no_leakage(d.train_images, d.test_images, tr_idx, "train")
    assert_no_leakage(d.train_images, d.test_images, val_idx, "val")

    lo = hi = None
    if a.norm in ("fixed", "fixed_log"):
        lo, hi = calibrate(d.train_images, tr_idx, a.norm, seed=a.seed)
        print("  fixed range       : [{:.6g}, {:.6g}]".format(lo, hi))

    # class weights from the ACTUAL training split
    samp = rng.choice(tr_idx, size=min(400, len(tr_idx)), replace=False)
    mean_rfi = float(np.mean([d.train_masks[int(i), :, :, 0].mean() for i in samp]))
    if a.no_class_weight:
        cw = [1.0, 1.0]
    else:
        cw = [0.5 / (1 - mean_rfi), 0.5 / mean_rfi]
        if a.class_weight_cap > 0:
            cw[1] = min(cw[1], a.class_weight_cap)
    print("  measured RFI frac : {:.4f}%  -> class_weights [{:.3f}, {:.3f}]".format(
        mean_rfi * 100, cw[0], cw[1]))
    print("=" * 74, flush=True)

    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.total_epochs)

    prog_path = os.path.join(out, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else \
        {"epochs_completed": 0, "best_f1": -1.0, "best_epoch": None}
    ckpt, best = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        print("  resumed from epoch {}".format(prog["epochs_completed"]), flush=True)

    val_eval = val_idx[:a.n_val_images]
    t_start = time.time()
    while prog["epochs_completed"] < a.total_epochs:
        n_ep = min(a.epochs_per_chunk, a.total_epochs - prog["epochs_completed"])
        model.train()
        for _ in range(n_ep):
            tot, nb = 0.0, 0
            for x, y in batches(d.train_images, d.train_masks, tr_idx, a.batch_size,
                                a.norm, lo, hi, rng, a.iters_per_epoch):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = ce(logits, y) + a.dice_weight * dice_loss(logits, y)
                loss.backward()
                opt.step()
                tot += float(loss.item()); nb += 1
            sched.step()
            prog["epochs_completed"] += 1
            el = (time.time() - t_start) / 60
            eta = el / max(prog["epochs_completed"], 1) * (a.total_epochs - prog["epochs_completed"])
            print("  epoch {}/{}  loss {:.4f}  |  {:.1f}m elapsed  ETA {:.1f}m".format(
                prog["epochs_completed"], a.total_epochs, tot / max(nb, 1), el, eta), flush=True)

        yt, yp = predict_scores(model, d.train_images, d.train_masks, val_eval,
                                a.norm, lo, hi, device)
        vm = metrics_from(yt, yp, 0.5)
        print("     -> val max-F1 {:.4f}  ROC {:.4f}".format(vm["max_f1"], vm["roc_auc"]), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, ckpt)
        if vm["max_f1"] > prog["best_f1"]:
            prog["best_f1"], prog["best_epoch"] = vm["max_f1"], prog["epochs_completed"]
            torch.save(model.state_dict(), best)
            print("        new best checkpoint", flush=True)
        json.dump(prog, open(prog_path, "w"), indent=2)

    # ---- final evaluation on the 109 human-labelled images -------------------
    print("\nBest val max-F1 {:.4f} @ epoch {}".format(prog["best_f1"], prog["best_epoch"]))
    model.load_state_dict(torch.load(best, map_location=device))

    yt_v, yp_v = predict_scores(model, d.train_images, d.train_masks, val_eval,
                                a.norm, lo, hi, device)
    th, vf1 = best_threshold(yt_v, yp_v)
    print("Threshold selected on validation: {:.4f} (val F1 {:.4f})".format(th, vf1))

    yt_t, yp_t = predict_scores(model, d.test_images, d.test_masks, np.arange(109),
                                a.norm, lo, hi, device)
    m = metrics_from(yt_t, yp_t, th)
    m_crop = metrics_from(yt_t, yp_t, th, crop=472)
    m["pooled_f1_crop472"] = m_crop["pooled_f1"]
    m["max_f1_crop472"] = m_crop["max_f1"]
    m["roc_auc_crop472"] = m_crop["roc_auc"]
    m.update(model="HybridRFINet", base=a.base, depth=a.depth, dropout=a.dropout,
             parameters=n_par, norm=a.norm, fixed_range=[lo, hi], class_weights=cw,
             dice_weight=a.dice_weight, learning_rate=a.learning_rate,
             batch_size=a.batch_size, epochs=prog["epochs_completed"],
             gradient_steps=a.iters_per_epoch * prog["epochs_completed"],
             seed=a.seed, n_train=len(tr_idx), n_val=len(val_idx),
             val_selected_threshold=th, best_val_max_f1=prog["best_f1"],
             best_epoch=prog["best_epoch"],
             test_set="109 expert-labelled LOFAR baselines (data[3])")

    print("\n" + "=" * 74)
    print("  RESULT — HybridRFINet on real LOFAR, human ground truth")
    print("=" * 74)
    print("  pooled F1 (512x512)      : {:.4f}   precision {:.4f}  recall {:.4f}".format(
        m["pooled_f1"], m["pooled_precision"], m["pooled_recall"]))
    print("  pooled F1 (472 crop)     : {:.4f}   <-- compare with tf_unet, PART 12".format(
        m["pooled_f1_crop472"]))
    print("  max F1 (oracle, 512)     : {:.4f}   <-- compare with the paper's Table 2".format(
        m["max_f1"]))
    print("  ROC AUC                  : {:.4f}".format(m["roc_auc"]))
    print("  PR  AUC                  : {:.4f}".format(m["pr_auc"]))
    print("-" * 74)
    print("  benchmarks: sigma-clip 0.4103 | tf_unet lr1e-4 0.5482 | AOFlagger 0.5698")
    print("              paper's U-Net 0.5876 | RFI-Net 0.5979   (all max-F1)")
    print("=" * 74)

    os.makedirs(os.path.join(out, "eval_test"), exist_ok=True)
    dest = os.path.join(out, "eval_test", "metrics.json")
    json.dump(m, open(dest, "w"), indent=2)
    print("\nSaved: {}".format(dest))


if __name__ == "__main__":
    main()
