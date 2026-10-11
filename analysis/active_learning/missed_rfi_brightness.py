"""How visible are the RFI pixels the hybrid MISSES? (bears on topic 10, restoration)

A restoration / inpainting residual can only expose RFI that stands out from
the clean background. Tiers exactly as analysis/ablation_faint_recall.py
(PART 18.5): each expert-RFI pixel is ranked against the CLEAN pixels of its own
image -- bright > clean p99, moderate p75-p99, faint p50-p75, invisible < p50.
Missed = expert RFI not flagged at the model's validation-selected threshold.
Runs on CPU so it does not disturb GPU work.

    ~/torch-env/bin/python analysis/active_learning/missed_rfi_brightness.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "hybrid_rfi_package"))
from hybrid_model import HybridRFINet  # noqa: E402
from lofar_data import load_lofar      # noqa: E402

NAMES = ["bright", "moderate", "faint", "invisible"]


def normalise_fixed(img, lo, hi):
    d = np.asarray(img, dtype=np.float64)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def main():
    torch.set_num_threads(4)
    d = load_lofar()
    out = {}
    for seed in (0, 1, 2):
        run = os.path.join(ROOT, "runs", "lofar", f"hybrid_b08_nocw_seed{seed}")
        m = json.load(open(os.path.join(run, "eval_test", "metrics.json")))
        lo, hi = m["fixed_range"]; th = float(m["val_selected_threshold"])
        model = HybridRFINet(1, 2, base=m["base"], depth=m["depth"], dropout=m["dropout"])
        model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location="cpu")); model.eval()
        tot = np.zeros(4); miss = np.zeros(4); hit = np.zeros(4)
        for i in range(109):
            img = normalise_fixed(d.test_images[i, :, :, 0], lo, hi)
            y = d.test_masks[i, :, :, 0].astype(bool)
            if not y.any():
                continue
            with torch.no_grad():
                p = F.softmax(model(torch.from_numpy(img)[None, None]), 1)[0, 1].numpy()
            pred = p > th
            clean = img[~y]
            c50, c75, c99 = (np.percentile(clean, q) for q in (50, 75, 99))
            v = img[y]
            t = np.full(v.shape, 3); t[v >= c50] = 2; t[v >= c75] = 1; t[v >= c99] = 0
            mi = ~pred[y]
            for k in range(4):
                tot[k] += (t == k).sum(); miss[k] += ((t == k) & mi).sum(); hit[k] += ((t == k) & ~mi).sum()
        out[seed] = {"all_rfi_share": (tot / tot.sum()).tolist(), "missed_share": (miss / miss.sum()).tolist(),
                     "recall_by_tier": (hit / tot).tolist(), "missed_px": int(miss.sum())}
        print(f"seed {seed}: missed {int(miss.sum()):,} px | " + " | ".join(
            f"{n}: {100 * miss[k] / miss.sum():.1f}% of misses (recall {hit[k] / tot[k]:.3f})" for k, n in enumerate(NAMES)), flush=True)
    s = np.array([out[k]["all_rfi_share"] for k in out]).mean(0)
    ms = np.array([out[k]["missed_share"] for k in out]).mean(0)
    print("\nshare of ALL expert RFI : " + " | ".join(f"{n} {100 * s[k]:.1f}%" for k, n in enumerate(NAMES)))
    print("share of MISSED RFI     : " + " | ".join(f"{n} {100 * ms[k]:.1f}%" for k, n in enumerate(NAMES)) + "   (mean of 3 models)")
    print(f"missed RFI that is NOT above the median clean pixel (invisible): {100 * ms[3]:.1f}%; "
          f"below the clean p75 (faint + invisible): {100 * (ms[2] + ms[3]):.1f}%")
    json.dump(out, open(os.path.join(ROOT, "analysis", "active_learning", "missed_rfi_brightness.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
