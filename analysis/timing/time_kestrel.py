"""Inference time of Kestrel (HybridRFINet, base 8) on one 512x512 spectrogram.

The reported LOFAR model (runs/lofar/hybrid_b08_nocw_seed0/best.pt) is run on the 109 LOFAR
test images, one image at a time, on the GPU. Each image is normalised beforehand (the same
fixed range as in training), so the timed part is: copy to the GPU, forward pass, softmax,
copy the 512x512 RFI probability map back. 10 warm-up passes are not timed. The script
refuses to run on battery or when another process is using the GPU, since either would make
the timing meaningless (CLAUDE.md section 1).

    ~/torch-env/bin/python analysis/timing/time_kestrel.py
Writes analysis/timing/kestrel_timing.json.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import lofar_hybrid as L                    # noqa: E402

RUN = os.path.join(ROOT, "runs/lofar/hybrid_b08_nocw_seed0")
OUT = os.path.join(os.path.dirname(__file__), "kestrel_timing.json")


def check_machine():
    ac = open([os.path.join("/sys/class/power_supply", p, "online")
               for p in os.listdir("/sys/class/power_supply") if p.startswith("A")][0]).read().strip()
    assert ac == "1", "on battery: the GPU is throttled, timing would be wrong"
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    assert not busy, "another process is using the GPU: " + busy


def main():
    check_machine()
    dev = torch.device("cuda")
    m0 = json.load(open(os.path.join(RUN, "eval_test", "metrics.json")))
    assert m0["base"] == 8 and m0["norm"] == "fixed" and m0.get("variant", "hybrid") == "hybrid"
    lo, hi = m0["fixed_range"]
    model = L.HybridRFINet(1, 2, base=8, depth=4, dropout=0.2).to(dev)
    model.load_state_dict(torch.load(os.path.join(RUN, "best.pt"), map_location=dev))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    d = L.load_lofar()
    n = d.test_images.shape[0]
    X = [L.normalise(d.test_images[i, :, :, 0], "fixed", lo, hi).astype(np.float32) for i in range(n)]
    times = []
    with torch.inference_mode():
        for k in range(10 + n):
            x = X[k % n]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            xt = torch.from_numpy(x)[None, None].to(dev)
            p = F.softmax(model(xt), dim=1)[0, 1].cpu().numpy()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            assert p.shape == (512, 512)
            if k >= 10:
                times.append(dt * 1000.0)
    t = np.array(times)
    res = dict(model="Kestrel (HybridRFINet base 8)", checkpoint=os.path.relpath(RUN, ROOT) + "/best.pt",
               parameters=int(n_params), images=int(n), image_size="512x512", batch=1,
               device=torch.cuda.get_device_name(0), torch=torch.__version__,
               ms_per_image_median=float(np.median(t)), ms_per_image_mean=float(t.mean()),
               ms_per_image_std=float(t.std(ddof=1)), ms_min=float(t.min()), ms_max=float(t.max()))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
