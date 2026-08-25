"""
The fix for the tf_unet baseline: FIXED-RANGE normalisation.
Resumable, with clear per-epoch progress and best-checkpoint tracking.

WHY THIS EXISTS
---------------
The project's data provider normalises every image by its OWN min and max:
    data -= data.min(); data /= data.max()
Across this dataset the per-image maximum varies by a factor of 8.2 (27 to 222)
depending on how bright that image's RFI happens to be. So identical physical
noise lands at completely different normalised values from one image to the next.

tf_unet has NO normalisation layers inside it, so it cannot absorb that
inconsistency and never learns. The hybrid model has GroupNorm, which is exactly
why the same normalisation does not hurt it.

The authors never hit this: their own scripts/rfi_launcher.py passes
a_min=30, a_max=210, clipping every image into one FIXED physical range first.

MEASURED (tf_unet, features_root=16, this dataset, CPU)
    project recipe, per-image min-max : verification error stuck at 81%,
                                        final ROC 0.582 / max F1 0.339
    authors' recipe, per-image min-max: collapsed to a constant 0.1655,
                                        ROC 0.530 / max F1 0.292
    fixed-range norm (this script)    : verification error 3.4% by epoch 13

USAGE
    source ~/tf-env/bin/activate
    python3 experiments/baseline_fixednorm.py \
        --dataset_dir "Synthetic Dataset 276x600" --features_root 32 --epochs 60

RESUMING
    Just run the exact same command again. It reads progress.json and carries on
    from the last completed chunk. Safe to close the laptop mid-run.
"""
import os, sys, re, glob, json, time, shutil, logging, argparse
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
sys.modules["tensorflow"] = tf1

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "unet_rfi_package copy", "tf_unet"))
from tf_unet import unet, util, image_util          # authors' unmodified code
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# ---------------------------------------------------------------------------
# Progress display. tf_unet logs "Epoch 0, Average loss: ..." and its counter
# restarts at 0 inside every chunk, so on its own you cannot tell how far along
# you are. This rewrites those lines into "Epoch 7/60" with elapsed time and ETA.
# ---------------------------------------------------------------------------
class EpochProgress(logging.Filter):
    _pat = re.compile(r"Epoch (\d+), Average loss: ([\d.eE+-]+), learning rate: ([\d.eE+-]+)")

    def __init__(self, total):
        super(EpochProgress, self).__init__()
        self.total, self.offset = total, 0
        self.t0, self.done_this_session = time.time(), 0

    def filter(self, record):
        m = self._pat.match(record.getMessage())
        if not m:
            return True
        done = self.offset + int(m.group(1)) + 1
        self.done_this_session += 1
        el = time.time() - self.t0
        eta = (el / self.done_this_session) * (self.total - done)
        bar_n = int(24 * done / max(self.total, 1))
        record.msg = ("  [{}{}] Epoch {}/{} done  |  loss {:.4f}  |  lr {:.5f}"
                      "  |  elapsed {:.1f}m  |  ETA {:.1f}m").format(
            "#" * bar_n, "." * (24 - bar_n), done, self.total,
            float(m.group(2)), float(m.group(3)), el / 60.0, eta / 60.0)
        record.args = ()
        return True


# ---------------------------------------------------------------------------
class FixedNormProvider(image_util.BaseDataProvider):
    """Identical to the project's provider EXCEPT for _process_data."""
    channels, n_class = 1, 2

    def __init__(self, img_dir, msk_dir, lo, hi, shuffle=True):
        super(FixedNormProvider, self).__init__(None, None)
        self.image_files = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(msk_dir, "*.npy")))
        assert self.image_files, "no .npy files found in " + img_dir
        self.lo, self.hi, self.shuffle, self.file_idx = lo, hi, shuffle, -1

    def _next_data(self):
        self.file_idx = (self.file_idx + 1) % len(self.image_files)
        i = np.random.randint(len(self.image_files)) if self.shuffle else self.file_idx
        return (np.load(self.image_files[i]).astype(np.float32),
                np.load(self.mask_files[i]).astype(bool))

    def _process_data(self, data):
        # THE FIX: one fixed range for every image, so the same physical value
        # always maps to the same normalised value.
        return np.clip((data - self.lo) / (self.hi - self.lo), 0.0, 1.0)


def calibrate(img_dir, n=40):
    """Global 0.5 / 99.5 percentile over a sample of training images."""
    fs = sorted(glob.glob(os.path.join(img_dir, "*.npy")))[:n]
    los = [np.percentile(np.load(f).astype(np.float64), 0.5) for f in fs]
    his = [np.percentile(np.load(f).astype(np.float64), 99.5) for f in fs]
    return float(np.mean(los)), float(np.mean(his))


class _QuietRestore(logging.Filter):
    """tf_unet's predict() restores the checkpoint from disk on EVERY call and logs
    two lines each time. Evaluating 150 images produced 300 log lines. Suppress them."""
    def filter(self, record):
        m = record.getMessage()
        return not (m.startswith("Restoring parameters") or m.startswith("Model restored"))


def score(net, model_path, img_dir, msk_dir, lo, hi, limit=None, quiet=True):
    """
    Evaluate with ONE session and ONE checkpoint restore.

    tf_unet's net.predict() opens a fresh session and reloads the checkpoint from
    disk for every single image (~2.5 s each). At 150 test images that is over six
    minutes of pure disk I/O per evaluation, repeated every chunk. This does the
    same computation through the same public attributes (net.predicter, net.restore)
    but restores once, which is ~50x faster.
    """
    p = FixedNormProvider(img_dir, msk_dir, lo, hi, shuffle=False)
    n = min(limit, len(p.image_files)) if limit else len(p.image_files)
    yt, yp = [], []
    root = logging.getLogger()
    qf = _QuietRestore()
    for h in root.handlers:
        h.addFilter(qf)
    try:
        with tf1.Session() as sess:
            sess.run(tf1.global_variables_initializer())
            net.restore(sess, model_path)                 # ONCE, not once per image
            for k in range(n):
                x, y = p(1)
                y_dummy = np.empty((x.shape[0], x.shape[1], x.shape[2], net.n_class))
                pred = sess.run(net.predicter,
                                feed_dict={net.x: x, net.y: y_dummy, net.keep_prob: 1.0})
                y = util.crop_to_shape(y, pred.shape)
                yt.append(y[..., 1].ravel().astype(bool))
                yp.append(pred[..., 1].ravel())
                if not quiet and (k + 1) % 50 == 0:
                    print("    evaluated {}/{}".format(k + 1, n), flush=True)
    finally:
        for h in root.handlers:
            h.removeFilter(qf)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    fpr, tpr, _ = roc_curve(yt, yp)
    pr, rc, _ = precision_recall_curve(yt, yp)
    f1 = 2 * pr * rc / (pr + rc + 1e-10)
    return dict(roc_auc=float(auc(fpr, tpr)), pr_auc=float(auc(rc, pr)),
                max_f1=float(np.max(f1)), n_images=n,
                pred_min=float(yp.min()), pred_max=float(yp.max()),
                pred_std=float(yp.std()),
                collapsed=bool(np.allclose(yp, yp.flat[0], atol=1e-4)))


def load_progress(d):
    p = os.path.join(d, "progress.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"epochs_completed": 0, "best_f1": -1.0, "best_epoch": None}


def save_progress(d, prog):
    json.dump(prog, open(os.path.join(d, "progress.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="tf_unet baseline with fixed-range normalisation (resumable)")
    p.add_argument("--dataset_dir", default=os.path.join(_HERE, "..", "Synthetic Dataset 276x600"))
    p.add_argument("--output_dir",  default=os.path.join(_HERE, "..", "unet_run_fixednorm"))
    p.add_argument("--features_root", type=int, default=32, help="32 = what the project used; 64 = the paper's")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4, help="raise if VRAM allows")
    p.add_argument("--epochs", type=int, default=60, help="22 is NOT enough -- it stays frozen ~15 epochs")
    p.add_argument("--epochs_per_chunk", type=int, default=5,
                   help="checkpoint + validate every N epochs. Smaller = lose less on a crash, "
                        "slightly more overhead.")
    p.add_argument("--n_val_images", type=int, default=50, help="val images used to pick the best checkpoint")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lo", type=float, default=None, help="default: auto-calibrated from the training set")
    p.add_argument("--hi", type=float, default=None)
    p.add_argument("--fresh", action="store_true", help="ignore progress.json and start over")
    a = p.parse_args()

    tr_img = os.path.join(a.dataset_dir, "train", "images")
    tr_msk = os.path.join(a.dataset_dir, "train", "masks")
    va_img = os.path.join(a.dataset_dir, "val", "images")
    va_msk = os.path.join(a.dataset_dir, "val", "masks")
    te_img = os.path.join(a.dataset_dir, "test", "images")
    te_msk = os.path.join(a.dataset_dir, "test", "masks")
    if not os.path.isdir(tr_img):
        raise SystemExit("ERROR: {} not found. Check --dataset_dir.".format(tr_img))
    if not os.path.isdir(va_img):
        raise SystemExit("ERROR: no val/ split at {}. Refusing to run -- without it the best "
                         "checkpoint would have to be chosen on the TEST set.".format(va_img))

    os.makedirs(a.output_dir, exist_ok=True)
    best_dir = os.path.join(a.output_dir, "best_checkpoint")
    if a.fresh:
        shutil.rmtree(a.output_dir, ignore_errors=True)
        os.makedirs(a.output_dir, exist_ok=True)
    prog = load_progress(a.output_dir)

    cfg_path = os.path.join(a.output_dir, "run_config.json")
    if os.path.exists(cfg_path) and not a.fresh:
        saved = json.load(open(cfg_path))
        lo, hi = saved["lo"], saved["hi"]          # keep the SAME normalisation on resume
    else:
        lo, hi = (a.lo, a.hi) if (a.lo is not None and a.hi is not None) else calibrate(tr_img)

    n_train = len(glob.glob(os.path.join(tr_img, "*.npy")))
    iters = max(1, -(-n_train // a.batch_size))     # ceil -> one full pass, matches the hybrid

    gpus = tf1.config.experimental.list_physical_devices("GPU")
    print("=" * 72)
    print(" tf_unet baseline + FIXED-RANGE normalisation")
    print("=" * 72)
    print("  GPU               : {}".format([g.name for g in gpus] if gpus else "NONE (will be slow)"))
    print("  normalisation     : fixed [{:.2f}, {:.2f}]   <-- the fix".format(lo, hi))
    print("  class weights     : OFF")
    print("  layers / features : {} / {}".format(a.layers, a.features_root))
    print("  batch / iters     : {} / {}  (one full pass over {} images)".format(
        a.batch_size, iters, n_train))
    print("  epochs            : {}   ({} gradient steps total)".format(a.epochs, iters * a.epochs))
    print("  checkpoint every  : {} epochs  -> safe to close the laptop".format(a.epochs_per_chunk))
    if prog["epochs_completed"]:
        print("  RESUMING from epoch {} (best val F1 {:.4f} @ epoch {})".format(
            prog["epochs_completed"], prog["best_f1"], prog["best_epoch"]))
    print("=" * 72, flush=True)

    if prog["epochs_completed"] >= a.epochs:
        print("Already completed {}/{} epochs. Raise --epochs to keep training."
              .format(prog["epochs_completed"], a.epochs))
        return

    json.dump(dict(vars(a), lo=lo, hi=hi, training_iters=iters, n_train=n_train),
              open(cfg_path, "w"), indent=2)

    logging.getLogger().setLevel(logging.INFO)
    prog_filter = EpochProgress(a.epochs)
    for h in logging.getLogger().handlers:
        h.addFilter(prog_filter)

    log_csv = os.path.join(a.output_dir, "training_log.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w") as f:
            f.write("epoch,val_roc_auc,val_pr_auc,val_max_f1,is_best,chunk_minutes\n")

    while prog["epochs_completed"] < a.epochs:
        chunk = min(a.epochs_per_chunk, a.epochs - prog["epochs_completed"])
        prog_filter.offset = prog["epochs_completed"]
        restore = prog["epochs_completed"] > 0

        tf1.reset_default_graph()
        np.random.seed(42 + prog["epochs_completed"])
        tf1.set_random_seed(42)
        net = unet.Unet(channels=1, n_class=2, layers=a.layers, features_root=a.features_root,
                        cost="cross_entropy", cost_kwargs=dict(regularizer=0.001), summaries=False)
        trainer = unet.Trainer(net, batch_size=a.batch_size, optimizer="adam",
                               opt_kwargs=dict(learning_rate=a.lr))
        prov = FixedNormProvider(tr_img, tr_msk, lo, hi, shuffle=True)

        t0 = time.time()
        mp = trainer.train(prov, a.output_dir, training_iters=iters, epochs=chunk,
                           dropout=0.5, display_step=10 ** 9, restore=restore,
                           prediction_path=os.path.join(a.output_dir, "pred"))
        mins = (time.time() - t0) / 60.0
        prog["epochs_completed"] += chunk

        v = score(net, mp, va_img, va_msk, lo, hi, limit=a.n_val_images)
        is_best = v["max_f1"] > prog["best_f1"]
        if is_best:
            prog["best_f1"], prog["best_epoch"] = v["max_f1"], prog["epochs_completed"]
            shutil.rmtree(best_dir, ignore_errors=True)
            shutil.copytree(a.output_dir, best_dir,
                            ignore=shutil.ignore_patterns("best_checkpoint", "pred", "eval_test"))
        save_progress(a.output_dir, prog)
        with open(log_csv, "a") as f:
            f.write("{},{:.4f},{:.4f},{:.4f},{},{:.1f}\n".format(
                prog["epochs_completed"], v["roc_auc"], v["pr_auc"], v["max_f1"],
                int(is_best), mins))

        print("  >> checkpoint saved at epoch {}/{}  |  val ROC {:.4f}  val maxF1 {:.4f}{}"
              .format(prog["epochs_completed"], a.epochs, v["roc_auc"], v["max_f1"],
                      "   <-- NEW BEST" if is_best else ""), flush=True)
        if v["collapsed"]:
            print("\n  !! COLLAPSE: the model outputs one constant value everywhere. Stopping.")
            print("     Try a lower --lr (currently {}).".format(a.lr))
            return

    print("\nTraining done. Best val max-F1 {:.4f} @ epoch {}".format(prog["best_f1"], prog["best_epoch"]))
    print("Evaluating the BEST checkpoint on the TEST split (once) ...", flush=True)

    tf1.reset_default_graph()
    net = unet.Unet(channels=1, n_class=2, layers=a.layers, features_root=a.features_root,
                    cost="cross_entropy", cost_kwargs=dict(regularizer=0.001), summaries=False)
    m = score(net, os.path.join(best_dir, "model.ckpt"), te_img, te_msk, lo, hi, quiet=False)
    m.update(norm=[lo, hi], features_root=a.features_root, layers=a.layers,
             batch_size=a.batch_size, epochs=prog["epochs_completed"],
             steps=iters * prog["epochs_completed"], class_weights=False,
             optimizer="adam", learning_rate=a.lr,
             best_val_f1=prog["best_f1"], best_epoch=prog["best_epoch"])

    print("\n" + "=" * 72)
    print("  ROC AUC : {:.4f}      (project's per-image norm gave 0.582)".format(m["roc_auc"]))
    print("  PR  AUC : {:.4f}".format(m["pr_auc"]))
    print("  max F1  : {:.4f}      (project's per-image norm gave 0.339)".format(m["max_f1"]))
    print("  collapsed: {}   pred range [{:.3f}, {:.3f}]".format(
        m["collapsed"], m["pred_min"], m["pred_max"]))
    print("=" * 72)
    os.makedirs(os.path.join(a.output_dir, "eval_test"), exist_ok=True)
    out = os.path.join(a.output_dir, "eval_test", "metrics.json")
    json.dump(m, open(out, "w"), indent=2)
    print("\nSaved: {}".format(out))


if __name__ == "__main__":
    main()
