"""Save tf_unet's test-set predictions (best checkpoint) for HERA and LOFAR, seeds 0-2.

Uses the training script's own data provider and score() function, so the predictions
are exactly the ones behind the reported numbers; each run's max F1 is checked against
its metrics.json. Output per run: eval_test/test_pred_crop472.npy, float32 (N, 472, 472),
the RFI probability on the centre 472x472 that tf_unet outputs.

    ~/tf-env/bin/python analysis/hera_vs_lofar/predict_tfunet.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import lofar_tfunet_baseline as B          # noqa: E402  (sets up TF and the authors' tf_unet)

RUNS = {"hera": "runs/hera/tfunet_lr1e-4_seed{}", "lofar": "runs/lofar/lr1e-4_seed{}"}

for ds, pattern in RUNS.items():
    d = B.load_hera() if ds == "hera" else B.load_lofar()
    n = d.test_images.shape[0]
    for s in (0, 1, 2):
        run = os.path.join(ROOT, pattern.format(s))
        out = os.path.join(run, "eval_test", "test_pred_crop472.npy")
        m0 = json.load(open(os.path.join(run, "eval_test", "metrics.json")))
        assert m0["layers"] == 3 and m0["features_root"] == 32 and m0["norm"] == "fixed"
        lo, hi = m0["fixed_range"]
        prov = B.LofarProvider(d.test_images, d.test_masks, np.arange(n), "fixed", lo, hi,
                               shuffle=False)
        B.tf1.reset_default_graph()
        net = B.unet.Unet(channels=1, n_class=2, layers=3, features_root=32,
                          cost="cross_entropy", cost_kwargs=dict(regularizer=0.001),
                          summaries=False)
        m = B.score(net, os.path.join(run, "best", "model.ckpt"), prov, n,
                    threshold=m0["threshold"])
        side = int(round((m["_yp"].size / n) ** 0.5))
        assert side == 472, side
        assert abs(m["max_f1"] - m0["max_f1"]) < 1e-4, (run, m["max_f1"], m0["max_f1"])
        np.save(out, m["_yp"].reshape(n, side, side).astype(np.float32))
        print("{} seed {}: max F1 {:.4f} (metrics.json {:.4f})  -> {}".format(
            ds, s, m["max_f1"], m0["max_f1"], os.path.relpath(out, ROOT)), flush=True)
print("done")
