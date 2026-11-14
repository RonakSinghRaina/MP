"""Inference time of tf_unet (3 layers, 32 filters) on one 512x512 spectrogram.

The reported LOFAR model (runs/lofar/lr1e-4_seed0/best) is run on the 109 LOFAR test images,
one image at a time, on the GPU, through the same TensorFlow graph and data provider as the
training script. The images are prepared beforehand, so the timed part is one sess.run of
net.predicter: copy to the GPU, forward pass, softmax, copy the 472x472 RFI probability map
back. 10 warm-up passes are not timed. Refuses to run on battery or on a busy GPU.

    export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\\n' ':')$LD_LIBRARY_PATH"
    ~/tf-env/bin/python analysis/timing/time_tfunet.py
Writes analysis/timing/tfunet_timing.json.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import lofar_tfunet_baseline as B          # noqa: E402  (sets up TF and the authors' tf_unet)
import tensorflow as tf2                    # noqa: E402

RUN = os.path.join(ROOT, "runs/lofar/lr1e-4_seed0")
OUT = os.path.join(os.path.dirname(__file__), "tfunet_timing.json")


def check_machine():
    ac = open([os.path.join("/sys/class/power_supply", p, "online")
               for p in os.listdir("/sys/class/power_supply") if p.startswith("A")][0]).read().strip()
    assert ac == "1", "on battery: the GPU is throttled, timing would be wrong"
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    assert not busy, "another process is using the GPU: " + busy


def main():
    check_machine()
    tf = B.tf1
    gpus = tf2.config.list_physical_devices("GPU")
    assert gpus, "TensorFlow sees no GPU (is LD_LIBRARY_PATH set?)"
    m0 = json.load(open(os.path.join(RUN, "eval_test", "metrics.json")))
    assert m0["layers"] == 3 and m0["features_root"] == 32 and m0["norm"] == "fixed"
    lo, hi = m0["fixed_range"]
    d = B.load_lofar()
    n = d.test_images.shape[0]
    prov = B.LofarProvider(d.test_images, d.test_masks, np.arange(n), "fixed", lo, hi, shuffle=False)
    X = [prov(1)[0] for _ in range(n)]
    tf.reset_default_graph()
    net = B.unet.Unet(channels=1, n_class=2, layers=3, features_root=32,
                      cost="cross_entropy", cost_kwargs=dict(regularizer=0.001), summaries=False)
    n_params = int(sum(np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()))
    times = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        net.restore(sess, os.path.join(RUN, "best", "model.ckpt"))
        for k in range(10 + n):
            x = X[k % n]
            t0 = time.perf_counter()
            # only the input is fed: predicter does not depend on the label placeholder
            # (feeding a dummy 512x512x2 label, as the training script does, adds ~0.8 ms)
            pred = sess.run(net.predicter, feed_dict={net.x: x, net.keep_prob: 1.0})
            dt = time.perf_counter() - t0
            assert pred.shape[1:3] == (472, 472), pred.shape
            if k >= 10:
                times.append(dt * 1000.0)
    t = np.array(times)
    res = dict(model="tf_unet (3 layers, 32 filters)", checkpoint=os.path.relpath(RUN, ROOT) + "/best",
               parameters=n_params, images=int(n), image_size="512x512 in, 472x472 out", batch=1,
               device=gpus[0].name, tensorflow=tf2.__version__,
               ms_per_image_median=float(np.median(t)), ms_per_image_mean=float(t.mean()),
               ms_per_image_std=float(t.std(ddof=1)), ms_min=float(t.min()), ms_max=float(t.max()))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
