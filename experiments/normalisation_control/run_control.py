"""
CONTROL EXPERIMENT: is normalisation really what broke the tf_unet baseline?

THE QUESTION
------------
The run that fixed the baseline (F1 0.39 -> 0.93) changed THREE things at once:
    1. per-image min-max normalisation  ->  fixed-range normalisation
    2. class weights ON                 ->  class weights OFF
    3. 22 epochs                        ->  60 epochs
So it does not, on its own, prove which one mattered.

THIS SCRIPT ISOLATES #1.
Run it twice. Everything is identical except the --norm flag:

    --norm fixed        the working setup  (expected: F1 around 0.93)
    --norm per_image    the ONLY change    (expected: F1 collapses back to ~0.3)

Class weights stay OFF and epochs stay 60 in BOTH runs. So if per_image fails
while fixed succeeds, normalisation is proven to be the cause by itself, and
neither the class weights nor the epoch count can be credited.

If per_image ALSO scores well, then epochs / class weights were doing more of
the work than believed, and the explanation in AUDIT_REPORT.md needs revising.
Either outcome is worth knowing and worth reporting.

WHY THE TWO NORMALISATIONS DIFFER
---------------------------------
per_image:  data -= data.min(); data /= data.max()
            Each image is scaled by ITS OWN brightest pixel. Measured on this
            dataset that brightest pixel ranges from 27 to 222 -- an 8.2x swing
            -- so the same physical noise lands at a different number in every
            image, and no single decision threshold can work across the set.

fixed:      np.clip((data - LO) / (HI - LO), 0, 1)
            One ruler for every image. The same physical value always maps to
            the same number. This is what the authors did: their own
            scripts/rfi_launcher.py passes a_min=30, a_max=210.

tf_unet has no normalisation layers inside it (no BatchNorm, no GroupNorm), so
it cannot compensate for an inconsistent input scale. The hybrid's GroupNorm can,
which is why the same loader never hurt the hybrid.

USAGE
    source ~/tf-env/bin/activate
    cd "/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project"

    python3 experiments/normalisation_control/run_control.py --norm fixed
    python3 experiments/normalisation_control/run_control.py --norm per_image

Each writes to its own output folder. Nothing existing is touched.
Both are resumable: rerun the same command to continue after a stop.
"""
import os, sys, re, glob, json, time, shutil, logging, argparse
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
sys.modules["tensorflow"] = tf1

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "unet_rfi_package copy", "tf_unet"))
from tf_unet import unet, util, image_util          # authors' unmodified code
from sklearn.metrics import roc_curve, auc, precision_recall_curve


class EpochProgress(logging.Filter):
    """tf_unet's epoch counter restarts at 0 in every chunk. Rewrite it as N/total."""
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
        b = int(24 * done / max(self.total, 1))
        record.msg = ("  [{}{}] Epoch {}/{} done  |  loss {:.4f}  |  lr {:.5f}"
                      "  |  elapsed {:.1f}m  |  ETA {:.1f}m").format(
            "#" * b, "." * (24 - b), done, self.total,
            float(m.group(2)), float(m.group(3)), el / 60.0, eta / 60.0)
        record.args = ()
        return True


class _QuietRestore(logging.Filter):
    def filter(self, record):
        m = record.getMessage()
        return not (m.startswith("Restoring parameters") or m.startswith("Model restored"))


class Provider(image_util.BaseDataProvider):
    """
    One provider, two normalisation modes. THE ONLY THING THAT VARIES BETWEEN
    THE TWO RUNS OF THIS EXPERIMENT.
    """
    channels, n_class = 1, 2

    def __init__(self, img_dir, msk_dir, mode, lo, hi, shuffle=True):
        super(Provider, self).__init__(None, None)
        self.image_files = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(msk_dir, "*.npy")))
        assert self.image_files, "no .npy files in " + img_dir
        self.mode, self.lo, self.hi = mode, lo, hi
        self.shuffle, self.file_idx = shuffle, -1

    def _next_data(self):
        self.file_idx = (self.file_idx + 1) % len(self.image_files)
        i = np.random.randint(len(self.image_files)) if self.shuffle else self.file_idx
        return (np.load(self.image_files[i]).astype(np.float32),
                np.load(self.mask_files[i]).astype(bool))

    def _process_data(self, data):
        if self.mode == "fixed":
            # ONE ruler for every image.
            return np.clip((data - self.lo) / (self.hi - self.lo), 0.0, 1.0)
        # per_image: each image scaled by its own min and max -- the original
        # behaviour. Note there is no np.fabs() here, matching the project's
        # existing override (fabs would turn negative noise dips into bright
        # pixels the mask still calls clean).
        data = data - np.amin(data)
        mx = np.amax(data)
        return data / mx if mx != 0 else data


def calibrate(img_dir, n=40):
    fs = sorted(glob.glob(os.path.join(img_dir, "*.npy")))[:n]
    los = [np.percentile(np.load(f).astype(np.float64), 0.5) for f in fs]
    his = [np.percentile(np.load(f).astype(np.float64), 99.5) for f in fs]
    return float(np.mean(los)), float(np.mean(his))


def score(net, model_path, img_dir, msk_dir, mode, lo, hi, limit=None, quiet=True):
    """One session, one checkpoint restore -- ~50x faster than calling predict() per image."""
    p = Provider(img_dir, msk_dir, mode, lo, hi, shuffle=False)
    n = min(limit, len(p.image_files)) if limit else len(p.image_files)
    yt, yp = [], []
    root, qf = logging.getLogger(), _QuietRestore()
    for h in root.handlers:
        h.addFilter(qf)
    try:
        with tf1.Session() as sess:
            sess.run(tf1.global_variables_initializer())
            net.restore(sess, model_path)
            for k in range(n):
                x, y = p(1)
                yd = np.empty((x.shape[0], x.shape[1], x.shape[2], net.n_class))
                pred = sess.run(net.predicter,
                                feed_dict={net.x: x, net.y: yd, net.keep_prob: 1.0})
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


def main():
    p = argparse.ArgumentParser(description="Control experiment: isolate the effect of normalisation")
    p.add_argument("--norm", choices=["fixed", "per_image"], required=True,
                   help="THE VARIABLE UNDER TEST. Everything else is held constant.")
    p.add_argument("--dataset_dir", default=os.path.join(_ROOT, "Synthetic Dataset 276x600"))
    p.add_argument("--output_dir", default=None, help="default: <root>/unet_run_control_<norm>")
    p.add_argument("--features_root", type=int, default=32)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--class_weights", choices=["on", "off"], default="off",
                   help="SECOND variable under test. The original failing run had these ON "
                        "([0.58, 3.45]); every fixed run so far had them OFF.")
    p.add_argument("--epochs_per_chunk", type=int, default=5)
    p.add_argument("--n_val_images", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()
    if a.output_dir is None:
        a.output_dir = os.path.join(_ROOT, "unet_run_control_" + a.norm +
                                    ("_cw" if a.class_weights == "on" else ""))

    tr_i = os.path.join(a.dataset_dir, "train", "images")
    tr_m = os.path.join(a.dataset_dir, "train", "masks")
    va_i = os.path.join(a.dataset_dir, "val", "images")
    va_m = os.path.join(a.dataset_dir, "val", "masks")
    te_i = os.path.join(a.dataset_dir, "test", "images")
    te_m = os.path.join(a.dataset_dir, "test", "masks")
    for d in (tr_i, va_i, te_i):
        if not os.path.isdir(d):
            raise SystemExit("ERROR: {} not found.".format(d))

    if a.fresh:
        shutil.rmtree(a.output_dir, ignore_errors=True)
    os.makedirs(a.output_dir, exist_ok=True)
    best_dir = os.path.join(a.output_dir, "best_checkpoint")
    pfile = os.path.join(a.output_dir, "progress.json")
    prog = json.load(open(pfile)) if os.path.exists(pfile) else \
        {"epochs_completed": 0, "best_f1": -1.0, "best_epoch": None}

    cfg = os.path.join(a.output_dir, "run_config.json")
    if os.path.exists(cfg) and not a.fresh:
        s = json.load(open(cfg)); lo, hi = s["lo"], s["hi"]
    else:
        lo, hi = calibrate(tr_i)

    n_train = len(glob.glob(os.path.join(tr_i, "*.npy")))
    iters = max(1, -(-n_train // a.batch_size))
    gpus = tf1.config.experimental.list_physical_devices("GPU")

    cost_kwargs = dict(regularizer=0.001)
    cw = None
    if a.class_weights == "on":
        meta = os.path.join(a.dataset_dir, "train", "metadata.jsonl")
        mean_rfi = float(np.mean([json.loads(l)["rfi_fraction"] for l in open(meta)]))
        cw = [0.5 / (1 - mean_rfi), 0.5 / mean_rfi]
        cost_kwargs["class_weights"] = cw

    print("=" * 74)
    print(" CONTROL RUN  --  norm={}  class_weights={}".format(a.norm.upper(), a.class_weights.upper()))
    print("=" * 74)
    print("  GPU              : {}".format([g.name for g in gpus] if gpus else "NONE (slow)"))
    if a.norm == "fixed":
        print("  normalisation    : FIXED range [{:.2f}, {:.2f}]  (one ruler for every image)".format(lo, hi))
    else:
        print("  normalisation    : PER-IMAGE min-max  (each image scaled by its own brightest pixel)")
    print("  class weights    : {}{}".format(
        a.class_weights.upper(),
        "  [{:.3f}, {:.3f}]".format(cw[0], cw[1]) if cw else ""))
    print("  ---- held constant ----")
    print("  epochs           : {}".format(a.epochs))
    print("  layers/features  : {} / {}".format(a.layers, a.features_root))
    print("  batch / iters    : {} / {}   ({} steps total)".format(a.batch_size, iters, iters * a.epochs))
    print("  optimiser        : adam @ {}".format(a.lr))
    print("  output           : {}".format(a.output_dir))
    if prog["epochs_completed"]:
        print("  RESUMING from epoch {} (best val F1 {:.4f})".format(
            prog["epochs_completed"], prog["best_f1"]))
    print("=" * 74, flush=True)

    if prog["epochs_completed"] >= a.epochs:
        print("Already at {}/{} epochs; re-evaluating only.".format(prog["epochs_completed"], a.epochs))
    else:
        json.dump(dict(vars(a), lo=lo, hi=hi, training_iters=iters), open(cfg, "w"), indent=2)
        logging.getLogger().setLevel(logging.INFO)
        pf = EpochProgress(a.epochs)
        for h in logging.getLogger().handlers:
            h.addFilter(pf)
        csv = os.path.join(a.output_dir, "training_log.csv")
        if not os.path.exists(csv):
            open(csv, "w").write("epoch,val_roc_auc,val_pr_auc,val_max_f1,is_best,chunk_minutes\n")

        while prog["epochs_completed"] < a.epochs:
            chunk = min(a.epochs_per_chunk, a.epochs - prog["epochs_completed"])
            pf.offset = prog["epochs_completed"]
            restore = prog["epochs_completed"] > 0
            tf1.reset_default_graph()
            np.random.seed(42 + prog["epochs_completed"]); tf1.set_random_seed(42)
            net = unet.Unet(channels=1, n_class=2, layers=a.layers, features_root=a.features_root,
                            cost="cross_entropy", cost_kwargs=cost_kwargs, summaries=False)
            tr = unet.Trainer(net, batch_size=a.batch_size, optimizer="adam",
                              opt_kwargs=dict(learning_rate=a.lr))
            prov = Provider(tr_i, tr_m, a.norm, lo, hi, shuffle=True)
            t0 = time.time()
            mp = tr.train(prov, a.output_dir, training_iters=iters, epochs=chunk, dropout=0.5,
                          display_step=10 ** 9, restore=restore,
                          prediction_path=os.path.join(a.output_dir, "pred"))
            mins = (time.time() - t0) / 60.0
            prog["epochs_completed"] += chunk
            v = score(net, mp, va_i, va_m, a.norm, lo, hi, limit=a.n_val_images)
            is_best = v["max_f1"] > prog["best_f1"]
            if is_best:
                prog["best_f1"], prog["best_epoch"] = v["max_f1"], prog["epochs_completed"]
                shutil.rmtree(best_dir, ignore_errors=True)
                shutil.copytree(a.output_dir, best_dir,
                                ignore=shutil.ignore_patterns("best_checkpoint", "pred", "eval_test"))
            json.dump(prog, open(pfile, "w"), indent=2)
            open(csv, "a").write("{},{:.4f},{:.4f},{:.4f},{},{:.1f}\n".format(
                prog["epochs_completed"], v["roc_auc"], v["pr_auc"], v["max_f1"], int(is_best), mins))
            print("  >> epoch {}/{} checkpointed  |  val ROC {:.4f}  val maxF1 {:.4f}{}".format(
                prog["epochs_completed"], a.epochs, v["roc_auc"], v["max_f1"],
                "   <-- NEW BEST" if is_best else ""), flush=True)
            if v["collapsed"]:
                print("\n  !! COLLAPSE: constant output everywhere. This IS a result -- record it.")
                break

    print("\nEvaluating BEST checkpoint on TEST (once) ...", flush=True)
    tf1.reset_default_graph()
    net = unet.Unet(channels=1, n_class=2, layers=a.layers, features_root=a.features_root,
                    cost="cross_entropy", cost_kwargs=cost_kwargs, summaries=False)
    ck = os.path.join(best_dir, "model.ckpt") if os.path.isdir(best_dir) \
        else os.path.join(a.output_dir, "model.ckpt")
    m = score(net, ck, te_i, te_m, a.norm, lo, hi, quiet=False)
    m.update(norm_mode=a.norm, norm_range=[lo, hi], class_weights=a.class_weights,
             epochs=prog["epochs_completed"], features_root=a.features_root,
             layers=a.layers, batch_size=a.batch_size, learning_rate=a.lr,
             best_val_f1=prog["best_f1"], best_epoch=prog["best_epoch"])
    print("\n" + "=" * 74)
    print("  --norm {}   --class_weights {}".format(a.norm.upper(), a.class_weights.upper()))
    print("  ROC AUC : {:.4f}".format(m["roc_auc"]))
    print("  PR  AUC : {:.4f}".format(m["pr_auc"]))
    print("  max F1  : {:.4f}".format(m["max_f1"]))
    print("  collapsed: {}   pred range [{:.3f}, {:.3f}]  std {:.4f}".format(
        m["collapsed"], m["pred_min"], m["pred_max"], m["pred_std"]))
    print("=" * 74)
    os.makedirs(os.path.join(a.output_dir, "eval_test"), exist_ok=True)
    out = os.path.join(a.output_dir, "eval_test", "metrics.json")
    json.dump(m, open(out, "w"), indent=2)
    print("\nSaved: {}".format(out))


if __name__ == "__main__":
    main()
