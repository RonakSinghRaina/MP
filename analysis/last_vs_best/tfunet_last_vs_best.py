"""tf_unet lr 1e-4 headline runs: test max F1 of the LAST-epoch checkpoint (ckpt/, epoch 150;
Mesarcik et al.'s rule) vs the BEST-on-validation checkpoint (best/) we report.
Reuses the training script's split, calibration, provider and scoring."""
import os, sys, json
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import lofar_tfunet_baseline as B
tf1, unet = B.tf1, B.unet
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
        res[tag] = B.score(net, os.path.join(run, sub, "model.ckpt"), p, 109)["max_f1"]
    ref = json.load(open(os.path.join(run, "eval_test", "metrics.json")))["max_f1"]
    print(f"seed {seed}: best {res['best']:.4f} (recorded {ref:.4f})  last {res['last']:.4f}  diff {res['last']-res['best']:+.4f}", flush=True)
    out[seed] = res
b = np.array([out[s]["best"] for s in out]); l = np.array([out[s]["last"] for s in out])
print(f"best {b.mean():.4f} +/- {b.std(ddof=1):.4f} | last {l.mean():.4f} +/- {l.std(ddof=1):.4f} | diff {np.mean(l-b):+.4f}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "tfunet_last_vs_best.json"), "w"), indent=2)
