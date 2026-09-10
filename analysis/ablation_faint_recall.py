"""Does removing the strip convolutions hurt FAINT RFI specifically? (PART 17.5)

`experiments/run_ablation.py` says in its own comments that pooled F1 cannot
answer this, because the weak bins are a small share of the pixels. PART 15
measured the tiers on the headline model and found the model has essentially
solved bright RFI (recall 0.937) and catches only a quarter of the invisible
tier (0.254).

`hybrid_model.py` justifies the strip convolutions by claiming they let "a
1.5-sigma line become detectable" by integrating along coherent structure. If
that mechanism is real, `hybrid_full` should beat `no_strip` **in the faint and
invisible tiers**, even though the LOFAR ablation found no difference in pooled
F1 (+0.0013, 0.3 seed sd). This script is the precise test of that claim.

Tiers follow PART 15: each true-RFI pixel is bucketed by how bright it is
relative to that same image's own CLEAN pixels, in the normalised space the
model actually sees.

    ~/torch-env/bin/python analysis/ablation_faint_recall.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, os.path.join(_ROOT, "src", "hybrid_rfi_package"))

from hybrid_model import HybridRFINet          # noqa: E402
from lofar_data import load_lofar              # noqa: E402
from models_ablation import build              # noqa: E402

TIERS = [("bright   (>clean p99)", 99.0, np.inf),
         ("moderate (p75-p99)", 75.0, 99.0),
         ("faint    (p50-p75)", 50.0, 75.0),
         ("invisible (<clean p50)", -np.inf, 50.0)]


def normalise_fixed(img, lo, hi):
    d = np.asarray(img, dtype=np.float64)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def scores_for(run_dir, d, device):
    m = json.load(open(os.path.join(run_dir, "eval_test", "metrics.json")))
    lo, hi = m["fixed_range"]
    variant = m.get("variant", "hybrid")
    if variant == "hybrid":
        model = HybridRFINet(1, 2, base=m["base"], depth=m["depth"], dropout=m["dropout"])
    else:
        model = build(variant, base=m["base"], depth=m["depth"], dropout=m["dropout"])
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    probs = np.zeros((109, 512, 512), dtype=np.float32)
    for i in range(0, 109, 2):
        sel = range(i, min(i + 2, 109))
        X = np.stack([normalise_fixed(d.test_images[j, :, :, 0], lo, hi) for j in sel])
        x = torch.from_numpy(X).unsqueeze(1).float().to(device)
        p = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        probs[i:i + len(X)] = p
    del model
    torch.cuda.empty_cache()
    return probs, float(m["val_selected_threshold"]), m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_lofar()

    # ---- tier assignment, computed once from the data (model-independent) ---
    # Uses the same fixed range as the runs, so "brightness" is measured in the
    # space the model sees. All ablation runs share one fixed_range.
    any_run = sorted(glob.glob(os.path.join(_ROOT, "runs", "lofar", "abl_*")))[0]
    lo, hi = json.load(open(os.path.join(any_run, "eval_test", "metrics.json")))["fixed_range"]

    tier_id = np.full((109, 512, 512), -1, dtype=np.int8)
    truth = np.zeros((109, 512, 512), dtype=bool)
    for i in range(109):
        img = normalise_fixed(d.test_images[i, :, :, 0], lo, hi)
        msk = d.test_masks[i, :, :, 0].astype(bool)
        truth[i] = msk
        if not msk.any():
            continue
        clean = img[~msk]
        cuts = {p: np.percentile(clean, p) for p in (50.0, 75.0, 99.0)}
        v = img[msk]
        t = np.full(v.shape, 3, dtype=np.int8)          # invisible
        t[v >= cuts[50.0]] = 2                           # faint
        t[v >= cuts[75.0]] = 1                           # moderate
        t[v >= cuts[99.0]] = 0                           # bright
        tier_id[i][msk] = t

    counts = [int((tier_id == k).sum()) for k in range(4)]
    total = sum(counts)
    print("=" * 82)
    print("  RFI brightness tiers on the 109 expert-labelled test images")
    print("=" * 82)
    for (name, _, _), n in zip(TIERS, counts):
        print(f"  {name:<24} {n:>9,}  ({100.0 * n / total:5.2f}% of all RFI)")
    print(f"  {'TOTAL':<24} {total:>9,}")

    # ---- per-variant recall in each tier ------------------------------------
    runs = sorted(glob.glob(os.path.join(_ROOT, "runs", "lofar", "abl_*_seed0")))
    order = ["hybrid_full", "no_strip", "no_eca", "no_res", "plain_unet", "no_groupnorm"]
    rows = {}
    for r in runs:
        name = os.path.basename(r).replace("abl_", "").replace("_seed0", "")
        probs, th, m = scores_for(r, d, device)
        pred = probs > th
        rec = [float((pred & (tier_id == k)).sum()) / counts[k] if counts[k] else float("nan")
               for k in range(4)]
        rows[name] = dict(recall=rec, max_f1=m["max_f1"], thr=th,
                          prec=m.get("pooled_precision"), rc=m.get("pooled_recall"))

    print()
    print("=" * 82)
    print("  RECALL BY TIER  (at each variant's own validation-selected threshold)")
    print("=" * 82)
    hdr = f"{'variant':<14}{'max F1':>8}" + "".join(f"{n.split()[0]:>11}" for n, _, _ in TIERS)
    # (tier labels are truncated to their first word so the columns line up)
    print(hdr)
    for k in order:
        if k not in rows:
            continue
        v = rows[k]
        print(f"{k:<14}{v['max_f1']:>8.4f}" + "".join(f"{x:>11.3f}" for x in v["recall"]))

    print()
    print("  WARNING: the table above is NOT a fair comparison. Each variant uses")
    print("  its own validation-selected threshold, and those differ a lot")
    print("  (hybrid_full {:.3f} vs no_strip {:.3f}). A lower threshold raises"
          .format(rows.get("hybrid_full", {}).get("thr", float("nan")),
                  rows.get("no_strip", {}).get("thr", float("nan"))))
    print("  recall in EVERY tier for free. See the matched comparison below.")

    # ---- matched-budget comparison, the honest one --------------------------
    # Give every variant the same number of predicted positives as hybrid_full,
    # so any tier difference is about WHERE each model spends its detections,
    # not how many it makes.
    if "hybrid_full" in rows:
        ref_probs, ref_th, _ = scores_for(
            os.path.join(_ROOT, "runs", "lofar", "abl_hybrid_full_seed0"), d, device)
        budget = int((ref_probs > ref_th).sum())
        print()
        print("=" * 82)
        print("  MATCHED-BUDGET RECALL BY TIER")
        print("  every variant thresholded to emit the same {:,} positives".format(budget))
        print("=" * 82)
        print(hdr.replace("max F1", " prec "))
        matched = {}
        for r in runs:
            name = os.path.basename(r).replace("abl_", "").replace("_seed0", "")
            probs, _, _ = scores_for(r, d, device)
            # threshold at the budget-th largest score
            flat = probs.reshape(-1)
            th = np.partition(flat, -budget)[-budget]
            pred = probs >= th
            rec = [float((pred & (tier_id == k)).sum()) / counts[k] if counts[k] else float("nan")
                   for k in range(4)]
            prec = float((pred & truth).sum()) / max(int(pred.sum()), 1)
            matched[name] = rec
            print(f"{name:<14}{prec:>8.4f}" + "".join(f"{x:>11.3f}" for x in rec))

        if "no_strip" in matched:
            print()
            print("=" * 82)
            print("  THE STRIP-CONVOLUTION CLAIM, at matched budget")
            print("  hybrid_full minus no_strip, by tier")
            print("=" * 82)
            for (name, _, _), x, y in zip(TIERS, matched["hybrid_full"], matched["no_strip"]):
                d_ = x - y
                verdict = ("strips help here" if d_ > 0.02
                           else "strips HURT here" if d_ < -0.02 else "no effect")
                print(f"  {name:<24} {d_:+.4f}   {verdict}")
            print()
            print("  hybrid_model.py claims strips exist so a 1.5-sigma line becomes")
            print("  detectable. That predicts a clear positive in the faint/invisible")
            print("  rows. Anything within +/-0.02 is not a real effect at N=1.")


if __name__ == "__main__":
    main()
