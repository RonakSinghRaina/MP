"""Reproduce the "U-Net" row of Mesarcik et al. (2022), Table 2, on our machine.

Their published LOFAR U-Net scores max F1 0.5876 +/- 0.0031; our tf_unet scores
0.5482 +/- 0.0139. This script runs THEIR U-Net through THEIR pipeline, taken
from their released code (github.com/mesarcik/RFI-NLN, commit 9e756de), so the
gap can be measured instead of argued about. Every choice below cites the line
it reproduces.

    --setup paper_code   exactly what their code does, including two things we
                         would not do ourselves:
                           * trains on all 7500 training images, which include
                             the 109 test images (data.py load_lofar: no
                             de-duplication);
                           * computes the normalisation clip bounds from the
                             TEST set's clean pixels, i.e. using the human test
                             labels (data.py lines 111-112).
    --setup clean        the same recipe with both removed: trains on our 6621
                         clean training images, clip bounds from those images,
                         threshold for pooled F1 chosen on our 735 validation
                         images (same split as experiments/lofar_hybrid.py).

Their pipeline, as implemented in the code (it differs from the paper's text,
which says clip [|mu-sigma|, mu+20 sigma] and lr 1e-4):
  * clip to [|mu - 3 sd|, mu + 95 sd] of clean pixels, natural log, then one
    global min-max                                       (data.py 111-120)
  * non-overlapping 32x32 patches                          (run_lofar.sh)
  * batch 1024, Adam() with Keras defaults (lr 1e-3)       (model_config.py,
                                                            architectures/unet.py)
  * loss written as `bce(x_hat, y)`                         (unet.py train_step)
    Keras losses take (y_true, y_pred), so this passes the PREDICTION as the
    target and the binary MASK as the prediction. Measured: it equals
    ~16.1 * mean|x_hat - y|, an L1 loss with a constant-size gradient, not
    binary cross-entropy. Reproduced as written by default; --correct_loss
    uses bce(y, x_hat).
  * 100 epochs, the FINAL model is evaluated (no checkpoint selection)
  * F1 = max over all thresholds on the test set, pooled over all pixels of the
    reassembled 512x512 images (segmentation_metrics.py get_metrics)

Known deviation: their tf.data shuffle buffer (25,000 patches over image-ordered
data) only mixes ~100 neighbouring images; this script shuffles all patches
every epoch.

Run (TensorFlow environment):
    export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\\n' ':')$LD_LIBRARY_PATH"
    ~/tf-env/bin/python experiments/mesarcik_repro/train_mesarcik_unet.py --setup paper_code --seed 0
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from lofar_data import load_lofar  # noqa: E402

P = 32                      # patch size (run_lofar.sh)
TILES = 512 // P            # 16 tiles per side, 256 per image


# ---------------------------------------------------------------------------
# Clip bound multipliers (k_lo, k_hi) -> [|mu - k_lo*sd|, mu + k_hi*sd].
#   july: commit 2b8e71b (12 Jul 2022) "LOFAR clipping bug resolved" onwards -- what
#         the released code does; TEST stats used for both train and test.
#   june: commit 753b0d3 (29 Jun 2022) "experiments complete" -- what produced the
#         published Table 2 and matches the paper's text; train clipped with TRAIN
#         stats, test with TEST stats.
CLIPS = {"july": (3, 95), "june": (1, 20)}


def clip_bounds(images, masks, indices, k=(3, 95), chunk=200):
    """mean/std of the CLEAN pixels of the given images -> their clip bounds."""
    s = ss = n = 0.0
    for i in range(0, len(indices), chunk):
        sel = np.sort(indices[i:i + chunk])
        x = np.asarray(images[sel, ..., 0], dtype=np.float64)
        m = np.asarray(masks[sel, ..., 0], dtype=bool)
        v = x[~m]
        s += v.sum(); ss += (v * v).sum(); n += v.size
    mu = s / n
    sd = np.sqrt(ss / n - mu * mu)
    return abs(mu - k[0] * sd), mu + k[1] * sd, mu, sd


def transform(x, lo, hi):
    """clip -> log -> min-max. Their global min-max of the clipped, logged set is
    exactly [log lo, log hi] whenever some pixel reaches each bound, which is
    checked by the caller."""
    x = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
    return ((np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))).astype(np.float32)


def prepare(images, indices, lo, hi, cache, chunk=100):
    """Normalise once and cache as float16 so training reads 2 bytes/pixel."""
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + ".part.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                    shape=(len(indices), 512, 512))
    raw_min, raw_max = np.inf, -np.inf
    for i in range(0, len(indices), chunk):
        sel = indices[i:i + chunk]
        x = np.asarray(images[np.sort(sel), ..., 0])
        order = np.argsort(np.argsort(sel))          # restore requested order
        x = x[order]
        raw_min = min(raw_min, float(x.min())); raw_max = max(raw_max, float(x.max()))
        out[i:i + len(sel)] = transform(x, lo, hi).astype(np.float16)
    out.flush(); del out
    if not (raw_min <= lo and raw_max >= hi):
        raise RuntimeError("clip bounds not reached; global min-max would differ "
                           f"from [log lo, log hi]: raw [{raw_min}, {raw_max}] vs [{lo}, {hi}]")
    os.replace(tmp, cache)
    return np.load(cache, mmap_mode="r")


def tiles_of(arr):
    """(N,512,512) -> view (N,16,16,32,32) of non-overlapping 32x32 tiles."""
    n = arr.shape[0]
    return arr.reshape(n, TILES, P, TILES, P).transpose(0, 1, 3, 2, 4)


def predict_images(model, X, bs=1024):
    """Predict full 512x512 images by tiling, exactly as their reconstruct()."""
    import tensorflow as tf
    out = np.empty(X.shape, dtype=np.float32)
    for i in range(X.shape[0]):
        t = tiles_of(np.asarray(X[i:i + 1], dtype=np.float32))[0].reshape(-1, P, P, 1)
        p = model(tf.convert_to_tensor(t), training=False).numpy()[..., 0]
        out[i] = p.reshape(TILES, TILES, P, P).transpose(0, 2, 1, 3).reshape(512, 512)
    return out


def metrics(y_true, y_score, threshold=None):
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    yt = y_true.ravel().astype(bool); ys = y_score.ravel()
    fpr, tpr, _ = roc_curve(yt, ys)
    pr, rc, th = precision_recall_curve(yt, ys)
    f1 = 2 * rc * pr / (rc + pr + 1e-12)
    k = int(np.argmax(f1))
    out = dict(max_f1=float(f1[k]),
               max_f1_threshold=float(th[min(k, len(th) - 1)]),
               roc_auc=float(auc(fpr, tpr)), pr_auc=float(auc(rc, pr)))
    if threshold is not None:
        p = ys > threshold
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        P_ = tp / max(tp + fp, 1); R_ = tp / max(tp + fn, 1)
        out.update(pooled_f1=2 * P_ * R_ / max(P_ + R_, 1e-12), pooled_precision=P_,
                   pooled_recall=R_, threshold=float(threshold))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", choices=["paper_code", "clean"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--limit_train", type=int, default=None, help="smoke tests only")
    ap.add_argument("--correct_loss", action="store_true",
                    help="use bce(y, x_hat) instead of their swapped bce(x_hat, y)")
    ap.add_argument("--clip", choices=["july", "june"], default="july",
                    help="july = released code (default); june = the version that "
                         "produced their published numbers (see CLIPS)")
    ap.add_argument("--output_dir", default=None)
    a = ap.parse_args()

    out = a.output_dir or os.path.join(_ROOT, "runs", "lofar",
                                       f"mesarcik_unet_{a.setup}"
                                       f"{'_bce' if a.correct_loss else ''}"
                                       f"{'_june' if a.clip == 'june' else ''}_seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    cache_dir = os.path.join(_ROOT, "data", "lofar", "mesarcik_cache")

    import tensorflow as tf
    from mesarcik_unet import UNET
    tf.keras.utils.set_random_seed(a.seed)
    gpus = tf.config.list_physical_devices("GPU")

    d = load_lofar()
    if a.setup == "paper_code":
        tr_idx = np.arange(d.train_images.shape[0])                 # all 7500
        val_idx = None
        k = CLIPS[a.clip]
        te_lo, te_hi, _, _ = clip_bounds(d.test_images, d.test_masks, np.arange(109), k)
        if a.clip == "june":     # train clipped with its own (train) statistics
            lo, hi, mu, sd = clip_bounds(d.train_images, d.train_masks, tr_idx, k)
        else:                    # july: test statistics for both
            lo, hi, mu, sd = clip_bounds(d.test_images, d.test_masks, np.arange(109), k)
    else:
        rng = np.random.default_rng(a.seed)                          # = lofar_hybrid.py split
        idx = d.clean_train_idx.copy(); rng.shuffle(idx)
        n_val = max(1, int(len(idx) * 0.1))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        assert not set(tr_idx.tolist()) & set(d.leak_train_idx.tolist())
        lo, hi, mu, sd = clip_bounds(d.train_images, d.train_masks, tr_idx, CLIPS[a.clip])
        te_lo, te_hi = lo, hi                                        # no test statistics
    if a.limit_train:
        tr_idx = tr_idx[:a.limit_train]

    tag = (f"{a.setup}_seed{a.seed}" if a.setup == "clean" else a.setup) + \
          ("_june" if a.clip == "june" else "")
    print("=" * 74)
    print(f"  Mesarcik et al. U-Net reproduction  |  setup = {a.setup}  |  seed {a.seed}")
    print("=" * 74)
    print(f"  GPU               : {[g.name for g in gpus] or 'NONE -- CPU only'}")
    print(f"  training images   : {len(tr_idx)}"
          + ("  (includes the 109 test images -- as in their code)" if a.setup == "paper_code" else ""))
    print(f"  clip ({a.clip:<4})       : train [{lo:.6g}, {hi:.6g}] (mu {mu:.6g}, sd {sd:.6g})"
          f"   test [{te_lo:.6g}, {te_hi:.6g}]")

    Xtr = np.asarray(prepare(d.train_images, tr_idx, lo, hi,
                             os.path.join(cache_dir, f"{tag}_train_{len(tr_idx)}.npy")))
    Ytr = np.asarray(d.train_masks[np.sort(tr_idx), ..., 0])[np.argsort(np.argsort(tr_idx))]
    Xte = transform(d.test_images[..., 0], te_lo, te_hi)
    Yte = np.asarray(d.test_masks[..., 0], dtype=bool)
    Ttr, Mtr = tiles_of(Xtr), tiles_of(Ytr)
    n_tiles = len(tr_idx) * TILES * TILES
    steps = int(np.ceil(n_tiles / a.batch))
    print(f"  patches per epoch : {n_tiles:,}  ->  {steps:,} steps x {a.epochs} epochs")

    model = UNET((P, P, 1))
    opt = tf.keras.optimizers.Adam()                              # Keras default lr 1e-3
    bce = tf.keras.losses.BinaryCrossentropy()
    print(f"  parameters        : {model.count_params():,}")
    print(f"  loss              : {'bce(y, x_hat) [corrected]' if a.correct_loss else 'bce(x_hat, y) [their code; effectively L1]'}")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            x_hat = model(x, training=True)
            loss = bce(y, x_hat) if a.correct_loss else bce(x_hat, y)   # see docstring
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    prog_path = os.path.join(out, "progress.json")
    wpath = os.path.join(out, "last.weights.h5")
    start_ep = 0
    if os.path.exists(prog_path) and os.path.exists(wpath):
        start_ep = json.load(open(prog_path))["epochs_completed"]
        model.load_weights(wpath)
        print(f"  resuming after epoch {start_ep} (optimizer state restarts)")

    rng = np.random.default_rng(a.seed)
    flat = np.arange(n_tiles)
    t0 = time.time()
    for ep in range(start_ep, a.epochs):
        rng.shuffle(flat)
        losses = []
        for s in range(steps):
            b = flat[s * a.batch:(s + 1) * a.batch]
            im, rem = np.divmod(b, TILES * TILES); ty, tx = np.divmod(rem, TILES)
            x = Ttr[im, ty, tx].astype(np.float32)[..., None]
            y = Mtr[im, ty, tx].astype(np.float32)[..., None]
            losses.append(float(train_step(tf.convert_to_tensor(x), tf.convert_to_tensor(y))))
        model.save_weights(wpath)
        json.dump({"epochs_completed": ep + 1, "loss": float(np.mean(losses))},
                  open(prog_path, "w"))
        el = time.time() - t0
        print(f"  epoch {ep + 1:>3}/{a.epochs}  loss {np.mean(losses):.5f}  "
              f"{el / 60:.1f} min  ETA {el / (ep + 1 - start_ep) * (a.epochs - ep - 1) / 60:.0f} min",
              flush=True)

    # ---- evaluation: final model, as in their code ---------------------------
    pred = predict_images(model, Xte)
    th = None
    if val_idx is not None:
        Xv = transform(d.train_images[np.sort(val_idx), ..., 0], lo, hi)
        Yv = np.asarray(d.train_masks[np.sort(val_idx), ..., 0], dtype=bool)
        th = metrics(Yv, predict_images(model, Xv))["max_f1_threshold"]
    m = metrics(Yte, pred, threshold=th)
    m.update(model="Mesarcik et al. U-Net (RFI-NLN models.UNET)", setup=a.setup,
             seed=a.seed, epochs=a.epochs, batch=a.batch, patch=P,
             parameters=int(model.count_params()), n_train=int(len(tr_idx)),
             loss="bce(y, x_hat) -- corrected" if a.correct_loss
                  else "bce(x_hat, y) -- as in their code (effectively L1)",
             clip_version=a.clip, clip_bounds_train=[lo, hi], clip_bounds_test=[te_lo, te_hi],
             published_max_f1="0.5876 +/- 0.0031 (Mesarcik et al. 2022, Table 2)")
    os.makedirs(os.path.join(out, "eval_test"), exist_ok=True)
    json.dump(m, open(os.path.join(out, "eval_test", "metrics.json"), "w"), indent=2)
    np.save(os.path.join(out, "eval_test", "test_pred.npy"), pred.astype(np.float16))
    print("=" * 74)
    print(f"  max F1 (their protocol) : {m['max_f1']:.4f}    published 0.5876 +/- 0.0031")
    if th is not None:
        print(f"  pooled F1 (val thresh)  : {m['pooled_f1']:.4f}")
    print(f"  ROC AUC / PR AUC        : {m['roc_auc']:.4f} / {m['pr_auc']:.4f}"
          f"    published AUROC 0.8017, AUPRC 0.5920")
    print("=" * 74)


if __name__ == "__main__":
    main()
