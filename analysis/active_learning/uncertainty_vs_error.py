"""Does the hybrid's own uncertainty point at its mistakes? (topic 11, active learning)

Active learning only pays off if the patches the model is least sure about are
the patches where it disagrees with the expert. Measured here on the 109
expert-labelled LOFAR test images for the headline models
(runs/lofar/hybrid_b08_nocw_seed{0,1,2}), at each model's validation-selected
threshold:

  * share of all error pixels (FP + FN vs the expert) inside the k% most
    uncertain pixels / 64x64 patches (uncertainty = binary entropy of p);
  * the same for two controls at equal budget: RANDOM patches (expected k%) and
    the patches with the most PREDICTED RFI (errors can only occur near RFI, so
    uncertainty must beat "just pick patches that contain RFI" to be useful);
  * image level: Spearman correlation of mean uncertainty with error count.

    ~/torch-env/bin/python analysis/active_learning/uncertainty_vs_error.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "hybrid_rfi_package"))
from hybrid_model import HybridRFINet  # noqa: E402
from lofar_data import load_lofar      # noqa: E402

RUNS = os.path.join(ROOT, "runs", "lofar")
OUT = os.path.join(ROOT, "analysis", "active_learning", "uncertainty_vs_error.json")
PATCH = 64
BUDGETS = (0.05, 0.10, 0.20)


def normalise_fixed(img, lo, hi):
    d = np.asarray(img, dtype=np.float64)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def predict(model, images, lo, hi, dev):
    out = np.zeros((len(images), 512, 512), np.float32)
    for i in range(0, len(images), 2):                 # 2 at a time (CLAUDE.md s8)
        x = np.stack([normalise_fixed(im, lo, hi) for im in images[i:i + 2]])
        p = F.softmax(model(torch.from_numpy(x).unsqueeze(1).to(dev)), 1)[:, 1]
        out[i:i + len(x)] = p.cpu().numpy()
    return out


def entropy(p):
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def patch_sum(a):
    n = a.shape[0]
    return a.reshape(n, 512 // PATCH, PATCH, 512 // PATCH, PATCH).sum(axis=(2, 4)).reshape(-1)


def share_in_top(score, errors, frac):
    k = max(1, int(round(frac * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    return float(errors[top].sum() / max(errors.sum(), 1))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_lofar()
    Y = np.asarray(d.test_masks[..., 0], bool)
    imgs = [d.test_images[i, :, :, 0] for i in range(109)]
    R = {"patch": PATCH, "budgets": BUDGETS, "seeds": {}}
    for seed in (0, 1, 2):
        run = os.path.join(RUNS, f"hybrid_b08_nocw_seed{seed}")
        m = json.load(open(os.path.join(run, "eval_test", "metrics.json")))
        lo, hi = m["fixed_range"]; th = float(m["val_selected_threshold"])
        model = HybridRFINet(1, 2, base=m["base"], depth=m["depth"], dropout=m["dropout"])
        model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location="cpu"))
        model.to(dev).eval()
        P = predict(model, imgs, lo, hi, dev)
        E = (P > th) != Y                                  # error pixels vs the expert
        H = entropy(P)
        r = {"threshold": th, "error_px": int(E.sum()), "pixel": {}, "patch": {}}

        # pixel level
        h, e = H.ravel(), E.ravel()
        for b in BUDGETS:
            r["pixel"][str(b)] = share_in_top(h, e, b)

        # patch level: uncertainty vs random vs "most predicted RFI"
        pe = patch_sum(E.astype(np.float32))
        ph = patch_sum(H)
        prfi = patch_sum(P)
        rng = np.random.default_rng(seed)
        for b in BUDGETS:
            rand = np.mean([share_in_top(rng.random(len(pe)), pe, b) for _ in range(20)])
            r["patch"][str(b)] = {"uncertainty": share_in_top(ph, pe, b),
                                  "most_predicted_rfi": share_in_top(prfi, pe, b),
                                  "random": float(rand)}
        r["patch_spearman_uncertainty_vs_errors"] = float(spearmanr(ph, pe).correlation)
        r["patch_spearman_predicted_rfi_vs_errors"] = float(spearmanr(prfi, pe).correlation)

        # image level
        ie = E.reshape(109, -1).sum(1); ih = H.reshape(109, -1).mean(1)
        r["image_spearman_uncertainty_vs_errors"] = float(spearmanr(ih, ie).correlation)
        R["seeds"][seed] = r
        print(f"seed {seed}: threshold {th:.4f}, error px {E.sum():,}")
        for b in BUDGETS:
            q = r["patch"][str(b)]
            print(f"   top {int(b * 100):>2}% of 64x64 patches hold: uncertainty {100 * q['uncertainty']:5.1f}% | "
                  f"most predicted RFI {100 * q['most_predicted_rfi']:5.1f}% | random {100 * q['random']:5.1f}% "
                  f"of all errors   (top {int(b * 100)}% of PIXELS by uncertainty: {100 * r['pixel'][str(b)]:5.1f}%)")
        print(f"   Spearman, patches: uncertainty {r['patch_spearman_uncertainty_vs_errors']:.3f}, "
              f"predicted RFI {r['patch_spearman_predicted_rfi_vs_errors']:.3f} | "
              f"images: uncertainty {r['image_spearman_uncertainty_vs_errors']:.3f}")
        del model; torch.cuda.empty_cache()
    json.dump(R, open(OUT, "w"), indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
