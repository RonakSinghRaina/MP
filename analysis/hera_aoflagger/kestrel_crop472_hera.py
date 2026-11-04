"""Kestrel (our settings) on HERA, scored on the centre 472x472 that tf_unet sees.

The training script saved max F1 and AUROC on this crop but not AUPRC. This reloads
each seed's best.pt, re-scores the 135 HERA test images with the training script's own
functions, checks the saved numbers are reproduced, and writes all crop metrics.
"""
import os, sys, json, statistics as st
import numpy as np, torch
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import lofar_hybrid as L

d = L.load_hera()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = {}
for s in (0, 1, 2):
    run = os.path.join(_ROOT, "runs", "hera", f"hybrid_b08_nocw_seed{s}")
    saved = json.load(open(os.path.join(run, "eval_test", "metrics.json")))
    lo, hi = saved["fixed_range"]
    m = L.HybridRFINet(1, 2, base=8, depth=4, dropout=0.2).to(dev)
    m.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=dev))
    yt, yp = L.predict_scores(m, d.test_images, d.test_masks, np.arange(d.test_images.shape[0]),
                              "fixed", lo, hi, dev)
    full = L.metrics_from(yt, yp, saved["threshold"])
    crop = L.metrics_from(yt, yp, saved["threshold"], crop=472)
    for k_new, k_old in [(full["max_f1"], "max_f1"), (crop["max_f1"], "max_f1_crop472"),
                         (crop["roc_auc"], "roc_auc_crop472")]:
        assert abs(k_new - saved[k_old]) < 1e-4, (s, k_old, k_new, saved[k_old])
    out[f"seed{s}"] = {k: crop[k] for k in ("max_f1", "pooled_f1", "roc_auc", "pr_auc", "positive_fraction")}
    print(s, {k: round(v, 4) for k, v in out[f"seed{s}"].items()}, flush=True)
for k in ("max_f1", "pooled_f1", "roc_auc", "pr_auc"):
    v = [out[f"seed{s}"][k] for s in (0, 1, 2)]
    out[k] = [st.mean(v), st.stdev(v)]
    print(f"{k}: {st.mean(v):.4f} +/- {st.stdev(v):.4f}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "kestrel_crop472_hera.json"), "w"), indent=2)
print("saved; reproduced the saved full and crop numbers to 1e-4")
