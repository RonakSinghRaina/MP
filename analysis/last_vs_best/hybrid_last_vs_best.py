"""Hybrid headline runs: test max F1 of the LAST-epoch model (Mesarcik et al.'s rule:
no validation, model after the final epoch) vs the BEST-on-validation model we report.
Reuses the training script's own split, calibration and evaluation code."""
import os, sys, json
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.path.insert(0, os.path.join(ROOT, "src", "hybrid_rfi_package")); sys.path.insert(0, ROOT)
import lofar_hybrid as H
from hybrid_model import HybridRFINet
d = H.load_lofar(); dev = torch.device("cuda"); out = {}
for seed in (0, 1, 2):
    run = os.path.join(ROOT, "runs", "lofar", f"hybrid_b08_nocw_seed{seed}")
    rng = np.random.default_rng(seed); idx = d.clean_train_idx.copy(); rng.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.1)); tr_idx = idx[n_val:]
    lo, hi = H.calibrate(d.train_images, tr_idx, "fixed", seed=seed)
    res = {}
    for tag, path in (("best", "best.pt"), ("last", "last.pt")):
        m = HybridRFINet(1, 2, base=8, depth=4, dropout=0.2).to(dev)
        st = torch.load(os.path.join(run, path), map_location=dev)
        m.load_state_dict(st["model"] if "model" in st else st); m.eval()
        yt, yp = H.predict_scores(m, d.test_images, d.test_masks, np.arange(109), "fixed", lo, hi, dev)
        res[tag] = H.metrics_from(yt, yp, 0.5)["max_f1"]
    ref = json.load(open(os.path.join(run, "eval_test", "metrics.json")))["max_f1"]
    print(f"seed {seed}: best {res['best']:.4f} (recorded {ref:.4f})  last {res['last']:.4f}  diff {res['last']-res['best']:+.4f}", flush=True)
    out[seed] = res
b = np.array([out[s]["best"] for s in out]); l = np.array([out[s]["last"] for s in out])
print(f"best {b.mean():.4f} +/- {b.std(ddof=1):.4f} | last {l.mean():.4f} +/- {l.std(ddof=1):.4f} | diff {np.mean(l-b):+.4f}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "hybrid_last_vs_best.json"), "w"), indent=2)
