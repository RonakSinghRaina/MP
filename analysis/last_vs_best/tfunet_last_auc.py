"""tf_unet lr 1e-4 headline runs: ROC AUC and PR AUC of the LAST-epoch checkpoint (ckpt/), next to
the BEST-on-validation checkpoint (best/) as a check against the recorded metrics. No training.
Same split, calibration, provider and scoring as tfunet_last_vs_best.py. CPU is enough."""
import os, sys, json
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import lofar_tfunet_baseline as B
tf1, unet = B.tf1, B.unet
KEYS = ("max_f1", "roc_auc", "pr_auc")
d = B.load_lofar(); out = {}
for seed in (0, 1, 2):
    run = os.path.join(ROOT, "runs", "lofar", f"lr1e-4_seed{seed}")
    rng = np.random.default_rng(seed); idx = d.clean_train_idx.copy(); rng.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.1)); tr_idx = idx[n_val:]
    lo, hi = B.calibrate(d.train_images, tr_idx, "fixed", seed=seed)
    res = {}
    for tag, sub in (("best", "best"), ("last", "ckpt")):
        tf1.reset_default_graph()
        net = unet.Unet(channels=1, n_class=2, layers=3, features_root=32, cost="cross_entropy",
                        cost_kwargs=dict(regularizer=0.001), summaries=False)
        p = B.LofarProvider(d.test_images, d.test_masks, np.arange(109), "fixed", lo, hi, shuffle=False)
        m = B.score(net, os.path.join(run, sub, "model.ckpt"), p, 109)
        res[tag] = {k: float(m[k]) for k in KEYS}
        print(f"seed {seed} {tag}: " + "  ".join(f"{k} {res[tag][k]:.4f}" for k in KEYS), flush=True)
    res["recorded_best"] = {k: json.load(open(os.path.join(run, "eval_test", "metrics.json")))[k] for k in KEYS}
    out[seed] = res
for tag in ("best", "last"):
    for k in KEYS:
        a = np.array([out[s][tag][k] for s in out])
        print(f"{tag} {k}: {a.mean():.4f} +/- {a.std(ddof=1):.4f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "tfunet_last_auc.json"), "w"), indent=2)
print("DONE", flush=True)
