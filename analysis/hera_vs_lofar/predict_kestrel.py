"""Save Kestrel's test-set predictions (best.pt) for HERA and LOFAR, seeds 0-2.

Uses the training script's own predict_scores() and metrics_from(), so the predictions are
exactly the ones behind the reported numbers; each run's max F1 (full image and 472 crop)
is checked against its metrics.json. Output per run: eval_test/test_pred_full512.npy,
float32 (N, 512, 512), the RFI probability for every pixel.

    ~/torch-env/bin/python analysis/hera_vs_lofar/predict_kestrel.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import lofar_hybrid as L                    # noqa: E402

RUNS = {"hera": "runs/hera/hybrid_b08_nocw_seed{}", "lofar": "runs/lofar/hybrid_b08_nocw_seed{}"}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for ds, pattern in RUNS.items():
    d = L.load_hera() if ds == "hera" else L.load_lofar()
    n = d.test_images.shape[0]
    for s in (0, 1, 2):
        run = os.path.join(ROOT, pattern.format(s))
        m0 = json.load(open(os.path.join(run, "eval_test", "metrics.json")))
        assert m0["base"] == 8 and m0["norm"] == "fixed" and m0.get("variant", "hybrid") == "hybrid"
        lo, hi = m0["fixed_range"]
        model = L.HybridRFINet(1, 2, base=8, depth=4, dropout=0.2).to(dev)
        model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=dev))
        yt, yp = L.predict_scores(model, d.test_images, d.test_masks, np.arange(n),
                                  "fixed", lo, hi, dev)
        full = L.metrics_from(yt, yp, m0["threshold"])
        crop = L.metrics_from(yt, yp, m0["threshold"], crop=472)
        assert abs(full["max_f1"] - m0["max_f1"]) < 1e-4, (run, full["max_f1"], m0["max_f1"])
        assert abs(crop["max_f1"] - m0["max_f1_crop472"]) < 1e-4
        out = os.path.join(run, "eval_test", "test_pred_full512.npy")
        np.save(out, yp.reshape(n, 512, 512).astype(np.float32))
        print("{} seed {}: max F1 {:.4f} full, {:.4f} crop (metrics.json {:.4f}, {:.4f})  -> {}".format(
            ds, s, full["max_f1"], crop["max_f1"], m0["max_f1"], m0["max_f1_crop472"],
            os.path.relpath(out, ROOT)), flush=True)
print("done")
