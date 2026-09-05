"""
The authors' unmodified tf_unet on the REAL LOFAR dataset.
Fixed-range normalisation, NO class weights -- i.e. run #4 of PART 1, the
configuration that scored F1 0.9317 on the synthetic data, applied to real
telescope data with human ground-truth labels.

WHAT THIS RUNS
--------------
  architecture : tf_unet (Akeret et al. 2017), untouched authors' code
                 layers=3, features_root=32, cross-entropy + 0.001 regulariser
  optimiser    : Adam @ 1e-3, batch 4          (same as PART 1)
  budget       : 175 steps/epoch x 60 epochs = 10,500 gradient steps,
                 IDENTICAL to what PART 1 gave the synthetic baseline
  class weights: OFF                            (PART 1 found these lethal here)
  normalisation: FIXED range, one global [lo,hi] for every image
  train on     : the 7356 CLEAN training images (lofar_clean_train_idx.npy)
  test on      : the 109 expert-labelled images (data[3], human ground truth)

WHY THE CLEAN SUBSET IS NOT OPTIONAL
------------------------------------
All 109 test images are byte-identical to 109 training images (PART 11.5).
Training on the full 7500 means testing on images the model has already seen.
This script therefore trains only on clean_train_idx, which excludes those 109
plus 35 fully-flagged dead baselines -- AND verifies it at startup. Before any
training happens, every train and val image is fingerprinted against all 109
test images and the run ABORTS if a single one matches. Costs ~2 s.

NORMALISATION -- READ THIS BEFORE INTERPRETING THE RESULT
---------------------------------------------------------
On the synthetic data, fixed-range beat per-image by +0.213 F1 (PART 1).
An earlier draft of PART 11.8 predicted the opposite would hold on LOFAR.
That prediction was RETRACTED on 2026-09-05 -- it was reasoned from the
global min/max rather than measured. Measured over 200 clean training images,
the fixed range calibrates to [283922, 4.776e6] and the images come out
healthy (per-image means 0.096-0.607, under 2% of pixels saturated), and it
separates RFI BETTER than the per-image recipe: 1.322 vs 0.718 clean-sigma.

So fixed range is a sound default here, not a straw man. Run the other arms
anyway -- whether PART 1's normalisation finding transfers from synthetic to
real data is exactly the open question.

Three arms are provided so the comparison is one flag away:
    --norm fixed      global linear clip+scale        (default; PART 1 recipe)
    --norm fixed_log  global log, then clip+scale     (fixed, but log-compressed)
    --norm per_image  Mesarcik et al. 2022 section 4.2 (clip 20 sigma, log, min-max)

METRICS
-------
Reports BOTH:
  pooled_f1  -- one confusion matrix over all 109 images, thresholded at 0.5.
                THIS is what the paper's Table 2 reports. Compare to
                AOFlagger 0.5698 and RFI-Net 0.5979.
  max_f1     -- oracle best threshold, the project's usual convention.
                Optimistic; do not compare it to the paper.
Mean-per-image F1 is deliberately NOT the headline: it runs ~0.10 lower than
pooled F1 on this data (PART 11.6) and is not what the paper quotes.

USAGE
-----
    export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
    ~/tf-env/bin/python experiments/lofar_tfunet_baseline.py

    # smoke test first (a few minutes):
    ~/tf-env/bin/python experiments/lofar_tfunet_baseline.py \
        --epochs 1 --limit_train 200 --output_dir lofar_runs/smoke

MATCHING PART 1's CONDITIONS
----------------------------
PART 1 trained on 700 synthetic images: 175 steps/epoch x 60 epochs = 10,500
gradient steps. LOFAR has 6620 training images, so a *full pass* would be 1655
steps/epoch -- 60 epochs of that is 99,300 steps, 9.5x more training than the
baseline it is being compared against. Matching the epoch NUMBER would not
match the CONDITIONS.

So --iters_per_epoch defaults to 175, holding the gradient-step budget
identical to PART 1 while drawing those batches at random from all 6620
images. Pass --iters_per_epoch 0 for full passes, but then say so explicitly
in the paper, because it is no longer a matched comparison.

HARDWARE
--------
Measured 2026-09-05 on the RTX 3060 6 GB, batch 4, features_root 32:
2.66 s/step ON BATTERY (GPU clamped to 210 MHz of 2100 MHz, "SW Power Cap
Active"). PLUG THE LAPTOP IN before starting -- PART 7 recorded 174 ms/step
on AC, which is ~15x faster and turns this run from ~8 h into ~30 min.
Step time scales with pixel count, not aspect ratio: 265x1024 and 512x512
measured within 1% of each other.

RESUMING
    Re-run the identical command. It reads progress.json and continues.
"""
import os, sys, json, time, shutil, logging, argparse

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
sys.modules["tensorflow"] = tf1

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "unet_rfi_package copy", "tf_unet"))
sys.path.insert(0, _ROOT)

from tf_unet import unet, util, image_util        # authors' unmodified code
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from lofar_data import load_lofar, preprocess


# ---------------------------------------------------------------------------
# Data provider: reads straight from the memory-mapped arrays, so nothing is
# duplicated on disk and RAM stays flat.
# ---------------------------------------------------------------------------
class LofarProvider(image_util.BaseDataProvider):
    channels, n_class = 1, 2

    def __init__(self, images, masks, indices, norm, lo=None, hi=None, shuffle=True):
        super(LofarProvider, self).__init__(None, None)
        self.images, self.masks = images, masks
        self.indices = np.asarray(indices)
        self.norm, self.lo, self.hi = norm, lo, hi
        self.shuffle, self.pos = shuffle, -1

    def _next_data(self):
        if self.shuffle:
            i = int(self.indices[np.random.randint(len(self.indices))])
        else:
            self.pos = (self.pos + 1) % len(self.indices)
            i = int(self.indices[self.pos])
        return (self.images[i, :, :, 0].astype(np.float32),
                self.masks[i, :, :, 0].astype(bool))

    def _process_data(self, data):
        if self.norm == "per_image":
            return preprocess(data)                       # clip 20 sigma, log, min-max
        d = data.astype(np.float64)
        if self.norm == "fixed_log":
            d = np.log(np.maximum(d, 1e-6))
        return np.clip((d - self.lo) / (self.hi - self.lo), 0.0, 1.0).astype(np.float32)


def assert_no_leakage(train_images, test_images, indices, label):
    """Hard guard: no image the model trains on may equal a test image.

    All 109 test spectrograms are byte-identical to 109 training spectrograms
    (PART 11.5), so a wrong index file, a regenerated clean_train_idx, or an
    edited split would silently contaminate the result. Relying on the index
    file being correct is an assumption; this is a check.

    Fast: a float64 sum is a near-unique fingerprint (1995 distinct values
    among 2000 images), so it prefilters, and only collisions are compared
    element-wise. ~2 s for the full training set.
    """
    t0 = time.time()
    test_fp = {}
    for t in range(test_images.shape[0]):
        test_fp.setdefault(float(test_images[t, :, :, 0].sum(dtype=np.float64)), []).append(t)

    leaked = []
    for i in np.asarray(indices):
        i = int(i)
        f = float(train_images[i, :, :, 0].sum(dtype=np.float64))
        for t in test_fp.get(f, ()):
            if np.array_equal(train_images[i], test_images[t]):
                leaked.append((i, t))
                break

    if leaked:
        raise SystemExit(
            "\nABORTED -- TEST SET LEAKAGE in the {} split.\n"
            "{} image(s) are byte-identical to test images, e.g. {}\n"
            "(train index -> test index). Training on these makes the reported\n"
            "score meaningless. Use lofar_analysis/lofar_clean_train_idx.npy,\n"
            "or regenerate it with lofar_analysis/deep_audit_stage4_labels.py.\n"
            .format(label, len(leaked), leaked[:5]))
    print("  leakage check     : {} clean, 0 of {} match any test image  ({:.1f}s)".format(
        label, len(indices), time.time() - t0), flush=True)


def calibrate(images, indices, norm, n=200, seed=0):
    """One global [lo, hi] for the whole dataset: mean 0.5/99.5 percentile."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(indices, size=min(n, len(indices)), replace=False)
    los, his = [], []
    for i in sel:
        a = images[int(i), :, :, 0].astype(np.float64)
        if norm == "fixed_log":
            a = np.log(np.maximum(a, 1e-6))
        los.append(np.percentile(a, 0.5))
        his.append(np.percentile(a, 99.5))
    return float(np.mean(los)), float(np.mean(his))


# ---------------------------------------------------------------------------
class EpochProgress(logging.Filter):
    """tf_unet restarts its epoch counter inside every chunk; rewrite to global."""
    import re as _re
    _pat = _re.compile(r"Epoch (\d+), Average loss: ([\d.eE+-]+), learning rate: ([\d.eE+-]+)")

    def __init__(self, total):
        super(EpochProgress, self).__init__()
        self.total, self.offset = total, 0
        self.t0, self.done = time.time(), 0

    def filter(self, record):
        m = self._pat.match(record.getMessage())
        if not m:
            return True
        done = self.offset + int(m.group(1)) + 1
        self.done += 1
        el = time.time() - self.t0
        eta = (el / self.done) * (self.total - done)
        b = int(24 * done / max(self.total, 1))
        record.msg = ("  [{}{}] Epoch {}/{}  |  loss {:.4f}  |  lr {:.5f}"
                      "  |  {:.1f}m elapsed  |  ETA {:.1f}m").format(
            "#" * b, "." * (24 - b), done, self.total,
            float(m.group(2)), float(m.group(3)), el / 60.0, eta / 60.0)
        record.args = ()
        return True


class _QuietRestore(logging.Filter):
    def filter(self, record):
        m = record.getMessage()
        return not (m.startswith("Restoring parameters") or m.startswith("Model restored"))


# ---------------------------------------------------------------------------
def best_threshold(yt, yp):
    """Threshold maximising F1 on THIS set. Used on validation, applied to test."""
    pr, rc, th = precision_recall_curve(yt, yp)
    f1 = 2 * pr * rc / (pr + rc + 1e-10)
    k = int(np.argmax(f1[:-1])) if len(th) else 0
    return float(th[k]) if len(th) else 0.5, float(f1[k])


def score(net, model_path, provider, n_images, quiet=True, threshold=0.5):
    """Restore the checkpoint ONCE, then predict every image."""
    yt, yp = [], []
    root = logging.getLogger()
    qf = _QuietRestore()
    for h in root.handlers:
        h.addFilter(qf)
    try:
        with tf1.Session() as sess:
            sess.run(tf1.global_variables_initializer())
            net.restore(sess, model_path)
            for k in range(n_images):
                x, y = provider(1)
                # zeros, not np.empty: uninitialised memory can hold values that
                # overflow the float32 cast and print a RuntimeWarning. The tensor
                # is a placeholder only -- predicter never reads it.
                y_dummy = np.zeros((x.shape[0], x.shape[1], x.shape[2], net.n_class))
                pred = sess.run(net.predicter,
                                feed_dict={net.x: x, net.y: y_dummy, net.keep_prob: 1.0})
                y = util.crop_to_shape(y, pred.shape)
                yt.append(y[..., 1].ravel().astype(bool))
                yp.append(pred[..., 1].ravel().astype(np.float32))
                if not quiet and (k + 1) % 25 == 0:
                    print("    evaluated {}/{}".format(k + 1, n_images), flush=True)
    finally:
        for h in root.handlers:
            h.removeFilter(qf)

    yt = np.concatenate(yt)
    yp = np.concatenate(yp)

    # pooled confusion matrix at the given threshold -- the paper's metric.
    # For the test set this threshold is SELECTED ON VALIDATION, never on test.
    p = yp > threshold
    tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
    tn = int((~p & ~yt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    pooled_f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    fpr, tpr, _ = roc_curve(yt, yp)
    pr_c, rc_c, _ = precision_recall_curve(yt, yp)
    f1_curve = 2 * pr_c * rc_c / (pr_c + rc_c + 1e-10)

    return dict(
        threshold=float(threshold),
        pooled_f1=float(pooled_f1), pooled_precision=float(prec), pooled_recall=float(rec),
        _yt=yt, _yp=yp,
        TP=tp, FP=fp, FN=fn, TN=tn,
        max_f1=float(np.max(f1_curve)),
        roc_auc=float(auc(fpr, tpr)), pr_auc=float(auc(rc_c, pr_c)),
        n_images=int(n_images),
        positive_fraction=float(yt.mean()),
        pred_min=float(yp.min()), pred_max=float(yp.max()), pred_std=float(yp.std()),
        collapsed=bool(np.allclose(yp, yp.flat[0], atol=1e-4)),
        reference=dict(aoflagger_f1=0.5698, rfinet_f1=0.5979,
                       sigma_clip_baseline_f1=0.4103),
    )


def load_progress(d):
    p = os.path.join(d, "progress.json")
    return json.load(open(p)) if os.path.exists(p) else \
        {"epochs_completed": 0, "best_f1": -1.0, "best_epoch": None}


def save_progress(d, prog):
    json.dump(prog, open(os.path.join(d, "progress.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm", choices=["fixed", "fixed_log", "per_image"], default="fixed",
                    help="fixed = PART 1 recipe (default). See module docstring.")
    ap.add_argument("--epochs", type=int, default=60,
                    help="60 = PART 1. Do not lower it without saying so in the paper.")
    ap.add_argument("--iters_per_epoch", type=int, default=175,
                    help="175 = PART 1 (700 synthetic images / batch 4). Keeping this "
                         "fixed means 60x175 = 10,500 gradient steps, the IDENTICAL "
                         "budget PART 1 gave the synthetic baseline. Pass 0 for a full "
                         "pass over the 6620 training images (1655 steps/epoch, i.e. "
                         "9.5x more training than PART 1 -- no longer a matched comparison).")
    ap.add_argument("--chunk", type=int, default=2, help="epochs per resumable chunk")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--features_root", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--limit_train", type=int, default=None,
                    help="use only this many training images (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default=None)
    a = ap.parse_args()

    out = a.output_dir or os.path.join(_ROOT, "lofar_runs", "tfunet_{}_ncw".format(a.norm))
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    d = load_lofar()
    rng = np.random.default_rng(a.seed)
    idx = d.clean_train_idx.copy()
    rng.shuffle(idx)
    if a.limit_train:
        idx = idx[: a.limit_train]
    n_val = max(1, int(len(idx) * a.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    print("=" * 74)
    print("  tf_unet on REAL LOFAR   |   norm = {}   |   class weights = OFF".format(a.norm))
    print("=" * 74)
    print("  train images      : {}".format(len(tr_idx)))
    print("  val images        : {}".format(len(val_idx)))
    print("  test images       : 109  (human expert labels)")
    print("  leaked images excluded from training: {}".format(len(d.leak_train_idx)))
    _it = a.iters_per_epoch if a.iters_per_epoch > 0 else max(1, len(tr_idx) // a.batch_size)
    print("  steps/epoch       : {}   x {} epochs = {:,} gradient steps".format(
        _it, a.epochs, _it * a.epochs))
    print("                      (PART 1 synthetic baseline got 175 x 60 = 10,500)")

    assert_no_leakage(d.train_images, d.test_images, tr_idx, "train")
    assert_no_leakage(d.train_images, d.test_images, val_idx, "val")

    lo = hi = None
    if a.norm in ("fixed", "fixed_log"):
        lo, hi = calibrate(d.train_images, tr_idx, a.norm, seed=a.seed)
        print("  fixed range       : [{:.6g}, {:.6g}]".format(lo, hi))
    print("=" * 74, flush=True)

    train_p = LofarProvider(d.train_images, d.train_masks, tr_idx, a.norm, lo, hi, shuffle=True)
    val_p = LofarProvider(d.train_images, d.train_masks, val_idx, a.norm, lo, hi, shuffle=False)
    test_p = LofarProvider(d.test_images, d.test_masks, np.arange(109), a.norm, lo, hi, shuffle=False)

    iters = a.iters_per_epoch if a.iters_per_epoch > 0 else max(1, len(tr_idx) // a.batch_size)
    prog = load_progress(out)
    best_dir = os.path.join(out, "best")
    ckpt_dir = os.path.join(out, "ckpt")

    prog_filter = EpochProgress(a.epochs)
    logging.getLogger().addFilter(prog_filter)
    for h in logging.getLogger().handlers:
        h.addFilter(prog_filter)

    while prog["epochs_completed"] < a.epochs:
        n_ep = min(a.chunk, a.epochs - prog["epochs_completed"])
        prog_filter.offset = prog["epochs_completed"]

        tf1.reset_default_graph()
        net = unet.Unet(channels=1, n_class=2, layers=a.layers,
                        features_root=a.features_root, cost="cross_entropy",
                        cost_kwargs=dict(regularizer=0.001), summaries=False)
        trainer = unet.Trainer(net, batch_size=a.batch_size, optimizer="adam",
                               opt_kwargs=dict(learning_rate=a.lr))
        restore = os.path.exists(os.path.join(ckpt_dir, "checkpoint"))
        trainer.train(train_p, ckpt_dir, training_iters=iters, epochs=n_ep,
                      dropout=0.75, display_step=max(1, iters // 4), restore=restore)
        prog["epochs_completed"] += n_ep

        tf1.reset_default_graph()
        net = unet.Unet(channels=1, n_class=2, layers=a.layers,
                        features_root=a.features_root, cost="cross_entropy",
                        cost_kwargs=dict(regularizer=0.001), summaries=False)
        val_p.pos = -1
        m = score(net, os.path.join(ckpt_dir, "model.ckpt"), val_p,
                  min(len(val_idx), 150))      # 150 = PART 1's validation set size
        m.pop("_yt", None); m.pop("_yp", None)
        print("  -> val pooled-F1 {:.4f}  max-F1 {:.4f}  ROC {:.4f}{}".format(
            m["pooled_f1"], m["max_f1"], m["roc_auc"],
            "   [COLLAPSED]" if m["collapsed"] else ""), flush=True)

        if m["max_f1"] > prog["best_f1"]:
            prog["best_f1"], prog["best_epoch"] = m["max_f1"], prog["epochs_completed"]
            if os.path.isdir(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)
            print("     new best checkpoint saved", flush=True)
        save_progress(out, prog)

    print("\nTraining done. Best val max-F1 {:.4f} @ epoch {}".format(
        prog["best_f1"], prog["best_epoch"]))
    print("Evaluating the BEST checkpoint on the 109 HUMAN-LABELLED test images ...",
          flush=True)

    tf1.reset_default_graph()
    net = unet.Unet(channels=1, n_class=2, layers=a.layers,
                    features_root=a.features_root, cost="cross_entropy",
                    cost_kwargs=dict(regularizer=0.001), summaries=False)

    # Select the operating threshold on VALIDATION, never on test.
    val_p.pos = -1
    vm = score(net, os.path.join(best_dir, "model.ckpt"), val_p, min(len(val_idx), 150))
    th, vf1 = best_threshold(vm.pop("_yt"), vm.pop("_yp"))
    print("  threshold selected on validation: {:.4f}  (val F1 there {:.4f})".format(th, vf1))

    tf1.reset_default_graph()
    net = unet.Unet(channels=1, n_class=2, layers=a.layers,
                    features_root=a.features_root, cost="cross_entropy",
                    cost_kwargs=dict(regularizer=0.001), summaries=False)
    test_p.pos = -1
    m = score(net, os.path.join(best_dir, "model.ckpt"), test_p, 109,
              quiet=False, threshold=th)
    m.pop("_yt", None); m.pop("_yp", None)
    m.update(val_selected_threshold=th, val_f1_at_threshold=vf1,
             norm=a.norm, fixed_range=[lo, hi], class_weights=False,
             layers=a.layers, features_root=a.features_root,
             batch_size=a.batch_size, epochs=prog["epochs_completed"],
             optimizer="adam", learning_rate=a.lr, seed=a.seed,
             n_train=len(tr_idx), n_val=len(val_idx),
             best_val_max_f1=prog["best_f1"], best_epoch=prog["best_epoch"],
             test_set="109 expert-labelled LOFAR baselines (data[3])")

    print("\n" + "=" * 74)
    print("  RESULT on real LOFAR data, human ground truth")
    print("=" * 74)
    print("  pooled F1      : {:.4f}   <-- HEADLINE. threshold {:.4f} chosen on val.".format(
        m["pooled_f1"], th))
    print("      precision  : {:.4f}".format(m["pooled_precision"]))
    print("      recall     : {:.4f}".format(m["pooled_recall"]))
    print("  max F1 (oracle): {:.4f}   (threshold picked ON TEST -- optimistic;".format(m["max_f1"]))
    print("                            report only to compare with PART 1's convention)")
    print("  ROC AUC        : {:.4f}".format(m["roc_auc"]))
    print("  PR  AUC        : {:.4f}".format(m["pr_auc"]))
    print("  collapsed      : {}   pred range [{:.3f}, {:.3f}]".format(
        m["collapsed"], m["pred_min"], m["pred_max"]))
    print("-" * 74)
    print("  benchmarks : sigma-clip 0.4103 | AOFlagger 0.5698 | RFI-Net 0.5979")
    print("=" * 74)

    os.makedirs(os.path.join(out, "eval_test"), exist_ok=True)
    dest = os.path.join(out, "eval_test", "metrics.json")
    json.dump(m, open(dest, "w"), indent=2)
    print("\nSaved: {}".format(dest))


if __name__ == "__main__":
    main()
