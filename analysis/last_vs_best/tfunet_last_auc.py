"""tf_unet lr 1e-4 headline runs: ROC AUC and PR AUC of the LAST-epoch checkpoint (ckpt/), next to
the BEST-on-validation checkpoint (best/) as a check against the recorded metrics. No training.
Same split, calibration, provider and scoring as tfunet_last_vs_best.py. CPU is enough.
    --dataset hera reads runs/hera/tfunet_lr1e-4_seed*; the default is LOFAR."""
import argparse, os, sys, json
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import lofar_tfunet_baseline as B
tf1, unet = B.tf1, B.unet
KEYS = ("max_f1", "roc_auc", "pr_auc")
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["lofar", "hera"], default="lofar")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--run_root", default=None, help="folder holding the runs (default runs/<dataset>)")
ap.add_argument("--json", default=None, help="output file")
a = ap.parse_args()
run_root = a.run_root or os.path.join(ROOT, "runs", a.dataset)
if a.dataset == "hera":
    d, run_name = B.load_hera(), "tfunet_lr1e-4_seed{}"
    json_out = a.json or os.path.join(ROOT, "analysis", "hera_aoflagger", "tfunet_last_auc_hera.json")
else:
    d, run_name = B.load_lofar(), "lr1e-4_seed{}"
    json_out = a.json or os.path.join(os.path.dirname(__file__), "tfunet_last_auc.json")
n_test = d.test_images.shape[0]
out = {}
for seed in a.seeds:
    run = os.path.join(run_root, run_name.format(seed))
    rng = np.random.default_rng(seed); idx = d.clean_train_idx.copy(); rng.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.1)); tr_idx = idx[n_val:]
    lo, hi = B.calibrate(d.train_images, tr_idx, "fixed", seed=seed)
    res = {}
    for tag, sub in (("best", "best"), ("last", "ckpt")):
        tf1.reset_default_graph()
        net = unet.Unet(channels=1, n_class=2, layers=3, features_root=32, cost="cross_entropy",
                        cost_kwargs=dict(regularizer=0.001), summaries=False)
        p = B.LofarProvider(d.test_images, d.test_masks, np.arange(n_test), "fixed", lo, hi, shuffle=False)
        m = B.score(net, os.path.join(run, sub, "model.ckpt"), p, n_test)
        res[tag] = {k: float(m[k]) for k in KEYS}
        print(f"seed {seed} {tag}: " + "  ".join(f"{k} {res[tag][k]:.4f}" for k in KEYS), flush=True)
    res["recorded_best"] = {k: json.load(open(os.path.join(run, "eval_test", "metrics.json")))[k] for k in KEYS}
    out[seed] = res
for tag in ("best", "last"):
    for k in KEYS:
        v = np.array([out[s][tag][k] for s in out])
        print(f"{tag} {k}: {v.mean():.4f} +/- {(v.std(ddof=1) if len(v) > 1 else 0.0):.4f}", flush=True)
json.dump(out, open(json_out, "w"), indent=2)
print("DONE", flush=True)
