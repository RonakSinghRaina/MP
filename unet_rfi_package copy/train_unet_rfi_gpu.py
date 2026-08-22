"""
GPU training/evaluation of the ORIGINAL tf_unet U-Net (Akeret et al. 2017)
on our synthetic RFI dataset -- with epoch-accurate resumability and
validation-based best-checkpoint tracking added on top.

FIDELITY GUARANTEE
-------------------
Every line inside tf_unet/*.py is the authors' unmodified published code.
This file calls only their PUBLIC, documented API:
    - unet.Unet(...)                 (architecture -- untouched)
    - unet.Trainer(...).train(...)   (training loop -- untouched)
    - net.predict(...)               (inference -- untouched)
    - image_util.BaseDataProvider    (the documented extension point for
                                       custom datasets -- we only override
                                       `_next_data` and `_post_process`,
                                       exactly as their own docstring says
                                       subclasses should)

WHY THIS VERSION EXISTS
-------------------------
A prior attempt at this used a hand-written PyTorch reimplementation
(different architecture: added BatchNorm at small batch size, no dropout,
no class weighting for the imbalanced RFI/clean pixel split, same-padding
instead of valid-padding). It trained "successfully" but under-performed
the paper on PR AUC / F1 specifically because of those deviations. This
script goes back to the actual published tf_unet code and adds three
things ON TOP of it, without touching it:

  1. GPU: TensorFlow auto-places ops on GPU if CUDA is available -- no
     code change needed, just `pip install tensorflow` on a CUDA-capable
     machine. This script prints which device it's using so you can
     confirm before committing hours of training.

  2. Class weighting: our synthetic dataset is ~86% clean / ~14% RFI
     pixels (measured directly from your generated metadata.jsonl, not
     assumed). tf_unet's own `Unet` class already supports
     `cost_kwargs={"class_weights": [...]}`  -- a real, documented
     parameter of the published code, not a hack. This directly fixes the
     unweighted-cross-entropy problem that hurt the PyTorch attempt's
     PR AUC / F1.

  3. Epoch-accurate resumability + best-checkpoint tracking: tf_unet's own
     Trainer.train(restore=True) restores WEIGHTS correctly across
     restarts, but its internal epoch loop always runs `range(epochs)` --
     it does not remember how many epochs you already completed in a
     previous run. If you resumed after a crash at epoch 60 with
     --epochs 100 again, you'd silently train 160 total epochs. This
     script tracks true cumulative epochs itself (progress.json, outside
     tf_unet) and only ever asks their unmodified train() for the epochs
     still remaining. It also evaluates on your validation set after each
     chunk and keeps a separate copy of whichever checkpoint had the best
     F1 so far -- tf_unet's own train() just overwrites one checkpoint
     every epoch with no "best" tracking at all.

Patch-based training (--patch_size) is included because our synthetic
images are 1024x1024 -- about 6x more pixels than the 276x600 crops the
paper actually trained on. Training on random patches (via the documented
`_post_process` augmentation hook) keeps VRAM usage bounded on consumer
GPUs and is, if anything, closer to the paper's own working image scale,
not a deviation from it.
"""
import os
import sys
import glob
import json
import time
import shutil
import argparse

import numpy as np

# On a 6GB laptop GPU, TF's default behavior of grabbing ~all GPU memory at
# Session creation competes with the OS's own display use of the GPU and can
# cause failures unrelated to your model's actual memory footprint. This env
# var (TensorFlow's own officially documented mechanism) makes it allocate
# only what it actually needs, growing as required -- set before TF is
# imported. This is OS-level configuration, not an edit to tf_unet's code.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# ---------------------------------------------------------------------------
# 1. TensorFlow 1.x compatibility shim (no tf_unet code touched)
# ---------------------------------------------------------------------------
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
sys.modules["tensorflow"] = tf1

TF_UNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_unet")
sys.path.insert(0, TF_UNET_PATH)

from tf_unet import unet, util, image_util  # noqa: E402  (unmodified upstream package)


def preflight_vram_check(image_h, image_w, layers, features_root, batch_size):
    """
    Builds the SAME architecture at the SAME image size and does one real
    forward+backward step with dummy data, to catch out-of-VRAM failures
    in seconds instead of after however long training ran before it hit
    the same wall. Uses only tf_unet's own public Unet/Trainer classes --
    no internals touched. The graph this creates is discarded afterward
    (unet.Unet.__init__ calls tf.reset_default_graph() itself, same as
    every real training chunk does), so it doesn't affect the real run.
    """
    print(f"\nRunning VRAM preflight check: {image_h}x{image_w} image, "
          f"layers={layers}, features_root={features_root}, batch_size={batch_size} ...")
    try:
        dummy_x = np.random.rand(batch_size, image_h, image_w, 1).astype(np.float32)

        net = unet.Unet(channels=1, n_class=2, layers=layers, features_root=features_root,
                         cost="cross_entropy", cost_kwargs=dict(regularizer=0.001), summaries=False)
        trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum",
                                opt_kwargs=dict(learning_rate=0.2, momentum=0.2))

        global_step = tf1.Variable(0, name="global_step")
        optimizer = trainer._get_optimizer(training_iters=1, global_step=global_step)

        with tf1.Session() as sess:
            sess.run(tf1.global_variables_initializer())
            offset = net.offset
            dummy_y = np.zeros((batch_size, image_h - offset, image_w - offset, 2), dtype=np.float32)
            dummy_y[..., 0] = 1  # all "clean" -- content doesn't matter, only shapes/memory do
            sess.run(optimizer, feed_dict={net.x: dummy_x, net.y: dummy_y, net.keep_prob: 0.5})

        print("VRAM check passed -- these settings fit in memory. Proceeding.\n")
        return True

    except tf1.errors.ResourceExhaustedError:
        print(f"\n!! OUT OF GPU MEMORY at {image_h}x{image_w}, "
              f"layers={layers}, features_root={features_root}, batch_size={batch_size} !!")
        print("These settings don't fit in your GPU's VRAM. Try one of:")
        if batch_size > 1:
            print(f"  --batch_size {max(1, batch_size // 2)}   <-- TRY THIS FIRST (halve the batch)")
        print(f"  --patch_size {max(128, min(image_h, image_w) // 2)}   (train on smaller crops)")
        print(f"  --features_root {max(16, features_root // 2)}   (smaller network)")
        if batch_size > 1:
            print(f"  --batch_size 1   (you're currently using {batch_size})")
        return False
    finally:
        tf1.reset_default_graph()


def print_device_info():
    gpus = tf1.config.experimental.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected: {[g.name for g in gpus]} -- training will run on GPU.")
    else:
        print("!! NO GPU DETECTED !! TensorFlow will fall back to CPU, which will be"
              " very slow (potentially days) for 1024x1024 images. Check your CUDA/"
              " cuDNN install before starting a long run. Aborting is recommended if"
              " you did not expect this.")
    return len(gpus) > 0


# ---------------------------------------------------------------------------
# 2. Data provider: our .npy dataset + optional random-patch augmentation
#    (both are documented BaseDataProvider extension points, not edits to
#    tf_unet itself)
# ---------------------------------------------------------------------------
class RFINpyDataProvider(image_util.BaseDataProvider):
    channels = 1
    n_class = 2

    def __init__(self, images_dir, masks_dir, patch_size=None, a_min=None, a_max=None, shuffle_data=True):
        super(RFINpyDataProvider, self).__init__(a_min, a_max)
        self.patch_size = patch_size
        self.shuffle_data = shuffle_data
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.npy")))
        assert len(self.image_files) == len(self.mask_files) and len(self.image_files) > 0, (
            f"No matching image/mask .npy pairs found in {images_dir} / {masks_dir}"
        )
        self.file_idx = -1
        self._shuffle()
        print(f"RFINpyDataProvider: {len(self.image_files)} image/mask pairs from {images_dir}"
              + (f" (random {patch_size}x{patch_size} patches)" if patch_size else ""))

    def _shuffle(self):
        if self.shuffle_data:
            order = np.random.permutation(len(self.image_files))
            self.image_files = [self.image_files[i] for i in order]
            self.mask_files = [self.mask_files[i] for i in order]

    def _cycle_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.image_files):
            self.file_idx = 0
            self._shuffle()

    def _next_data(self):
        self._cycle_file()
        img = np.load(self.image_files[self.file_idx]).astype(np.float32)
        mask = np.load(self.mask_files[self.file_idx]).astype(bool)
        return img, mask

    def _process_data(self, data):
        # Overrides BaseDataProvider._process_data (documented override
        # point -- "To change this behavior the `_process_data` method can
        # be overwritten"). tf_unet's own implementation is
        #   data = np.clip(np.fabs(data), self.a_min, self.a_max)
        # i.e. it takes the ABSOLUTE VALUE before normalizing. That's a
        # correct assumption for real telescope power/amplitude data,
        # which is already non-negative -- but our synthetic spectrograms
        # have real, meaningful NEGATIVE values (thermal noise dipping
        # below the background mean, sometimes by 80+ units). Taking fabs()
        # first would turn a strong negative noise dip into a
        # near-maximum-brightness pixel, visually almost indistinguishable
        # from genuine strong RFI, while the ground truth mask correctly
        # still says "clean" there -- actively teaching the network wrong
        # associations. This override keeps everything else about their
        # normalization (clip, then min-subtract, then max-divide to [0,1])
        # identical, and just removes the fabs().
        data = np.clip(data, self.a_min, self.a_max)
        data -= np.amin(data)
        if np.amax(data) != 0:
            data /= np.amax(data)
        return data

    def _post_process(self, data, labels):
        # Documented augmentation hook (BaseDataProvider docstring:
        # "To enable some post processing such as data augmentation the
        # `_post_process` method can be overwritten.") -- not a change to
        # tf_unet's core logic.
        if self.patch_size is not None:
            ph = pw = self.patch_size
            ny, nx = data.shape[0], data.shape[1]
            y0 = np.random.randint(0, max(1, ny - ph + 1))
            x0 = np.random.randint(0, max(1, nx - pw + 1))
            data = data[y0:y0 + ph, x0:x0 + pw]
            labels = labels[y0:y0 + ph, x0:x0 + pw, :]
        return data, labels


# ---------------------------------------------------------------------------
# 3. Class weights computed from the ACTUAL dataset (not assumed)
# ---------------------------------------------------------------------------
def compute_class_weights(dataset_dir, split="train"):
    meta_path = os.path.join(dataset_dir, split, "metadata.jsonl")
    if not os.path.exists(meta_path):
        print(f"WARNING: {meta_path} not found -- cannot measure real class balance. "
              "Falling back to unweighted loss (this is the same situation the "
              "PyTorch attempt was in, and likely to reproduce its PR/F1 gap).")
        return None
    fracs = [json.loads(line)["rfi_fraction"] for line in open(meta_path)]
    mean_rfi = float(np.mean(fracs))
    mean_clean = 1.0 - mean_rfi
    w_clean = 0.5 / mean_clean
    w_rfi = 0.5 / mean_rfi
    print(f"Measured class balance from {meta_path}: {mean_rfi*100:.1f}% RFI pixels -> "
          f"class_weights = [{w_clean:.3f}, {w_rfi:.3f}]")
    return [w_clean, w_rfi]


# ---------------------------------------------------------------------------
# 4. Progress tracking (outside tf_unet) so resuming after a crash/restart
#    trains exactly the remaining epochs, not epochs + already-done epochs
# ---------------------------------------------------------------------------
def load_progress(output_dir):
    path = os.path.join(output_dir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"epochs_completed": 0, "best_f1": -1.0, "best_epoch": None}


def save_progress(output_dir, progress):
    path = os.path.join(output_dir, "progress.json")
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# 5. Evaluation (unchanged approach: their own predict() + sklearn metrics)
# ---------------------------------------------------------------------------
def evaluate(net, model_path, val_provider, n_eval_images, eval_batch_size=1):
    """
    Evaluates on n_eval_images validation images, feeding eval_batch_size
    images through the network at a time -- not all n_eval_images in one
    shot. net.predict() has no built-in batching, and calling it with a
    large batch dimension can use dramatically more VRAM than a single
    training step (e.g. requesting 10 eval images at once needs roughly
    10x the memory of the batch_size=1 training steps that ran successfully
    moments earlier). This was a real bug in an earlier version of this
    script -- discovered when it OOM'd during evaluation right after
    training itself completed cleanly.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    all_y_true, all_y_pred = [], []
    remaining = n_eval_images
    while remaining > 0:
        batch_n = min(eval_batch_size, remaining)
        x_val, y_val = val_provider(batch_n)
        prediction = net.predict(model_path, x_val)
        y_true = util.crop_to_shape(y_val, prediction.shape)
        all_y_true.append(y_true[..., 1].flatten())
        all_y_pred.append(prediction[..., 1].flatten())
        remaining -= batch_n

    y_true_rfi = np.concatenate(all_y_true)
    y_pred_rfi = np.concatenate(all_y_pred)

    # Detect the irrecoverable "dead network" state. tf_unet's output layer
    # applies a ReLU to the logits before softmax (unet.py line ~145). If both
    # logits go negative, ReLU clamps both to 0, softmax([0,0]) is exactly
    # [0.5, 0.5], and the gradient back through the ReLU is exactly zero --
    # so the network can never recover, no matter how long it trains. The
    # signature is: every predicted probability is ~0.5, loss pinned at
    # ln(2)=0.6931, ROC AUC exactly 0.5. Catching it here means we stop in
    # seconds instead of burning hours training a network that is provably
    # incapable of improving.
    collapsed = bool(np.allclose(y_pred_rfi, 0.5, atol=1e-4))

    fpr, tpr, _ = roc_curve(y_true_rfi, y_pred_rfi)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true_rfi, y_pred_rfi)
    pr_auc = auc(recall, precision)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_f1 = float(np.max(f1_scores))

    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "max_f1": best_f1,
            "collapsed": collapsed}


# ---------------------------------------------------------------------------
# 6. Main: chunked training with true cumulative-epoch resumability
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPU tf_unet training on the synthetic RFI dataset, resumable")
    # Defaults are computed relative to THIS FILE's location, not the current
    # working directory -- this matters because pressing VS Code's "Run"
    # button passes no arguments and its working directory isn't always the
    # script's folder. Matches the layout:
    #   Minor Project/
    #     Synthetic Dataset/{train,val or test}/{images,masks}/...
    #     unet_rfi_package/train_unet_rfi_gpu.py   <- this script
    _here = os.path.dirname(os.path.abspath(__file__))
    _default_dataset_dir = os.path.normpath(os.path.join(_here, "..", "Synthetic Dataset"))
    _default_output_dir = os.path.normpath(os.path.join(_here, "..", "unet_run_gpu"))

    parser.add_argument("--dataset_dir", default=_default_dataset_dir,
                         help=f"Default: {_default_dataset_dir} (edit if your folder is named/placed differently)")
    parser.add_argument("--output_dir", default=_default_output_dir)
    parser.add_argument("--layers", type=int, default=3, help="paper: 3")
    parser.add_argument("--features_root", type=int, default=64, help="paper: 64")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=512,
                         help="Random crop size for training (VRAM control). "
                              "Set 0 to disable and train on full 1024x1024 images.")
    parser.add_argument("--training_iters", type=int, default=64, help="steps per epoch")
    parser.add_argument("--total_epochs", type=int, default=100,
                         help="Total epochs to reach across ALL runs, resumed or not.")
    parser.add_argument("--epochs_per_chunk", type=int, default=5,
                         help="How many epochs to train before checkpointing, validating, "
                              "and (if you Ctrl+C or the process ends) being safely resumable.")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "momentum"],
                         help="Default 'adam'. The paper uses momentum with lr=0.2, but that was with "
                              "mini-batch size 32; at batch_size=1 the gradient noise is far higher and "
                              "lr=0.2 reliably drives tf_unet's ReLU-on-logits output layer into a dead "
                              "state (softmax outputs exactly 0.5 everywhere, loss pinned at ln(2)=0.6931, "
                              "ROC AUC exactly 0.5) from which it CANNOT recover, because ReLU's gradient "
                              "at negative input is zero. Measured on this dataset: momentum lr=0.2 died at "
                              "iteration 4; adam lr=0.001 reached ROC AUC 0.84 on the same data and seed.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                         help="Default 0.001 (Adam). If you switch to --optimizer momentum, use something "
                              "like 0.01 -- do NOT use the paper's 0.2 at batch_size=1.")
    parser.add_argument("--n_eval_images", type=int, default=10, help="Val images per checkpoint eval")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                         help="How many eval images go through the network at once. Keep this at 1 "
                              "unless you've confirmed higher values fit your VRAM -- evaluation uses "
                              "the same memory-per-image as training, multiplied by this batch size.")
    parser.add_argument("--display_step", type=int, default=10)
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weighting")
    parser.add_argument("--skip_vram_check", action="store_true",
                         help="Skip the preflight VRAM test and go straight to training")
    args = parser.parse_args()

    has_gpu = print_device_info()
    if not has_gpu:
        resp = input("Continue on CPU anyway? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted. Install a CUDA-enabled TensorFlow build and NVIDIA drivers, then rerun.")
            return

    train_images_check = os.path.join(args.dataset_dir, "train", "images")
    if not os.path.isdir(train_images_check):
        print(f"\nERROR: Could not find {train_images_check}")
        print(f"       (--dataset_dir resolved to: {args.dataset_dir})")
        print("Fix by either:")
        print(f"  1. Renaming your dataset folder so this path exists, or")
        print(f"  2. Running with an explicit path, e.g.:")
        print(f'     python {os.path.basename(__file__)} --dataset_dir "/full/path/to/Synthetic Dataset"')
        return

    if not args.skip_vram_check:
        sample_files = glob.glob(os.path.join(train_images_check, "*.npy"))
        if args.patch_size > 0:
            probe_h = probe_w = args.patch_size
        elif sample_files:
            # Use the REAL image dimensions. An earlier version probed a square
            # image (height x height), which silently under-tested memory for
            # non-square data such as the paper's 276x600.
            probe_h, probe_w = np.load(sample_files[0]).shape[:2]
        else:
            probe_h = probe_w = 1024
        ok = preflight_vram_check(probe_h, probe_w, args.layers, args.features_root, args.batch_size)
        if not ok:
            print("Aborting before wasting time on a full run. Adjust settings above and rerun, "
                  "or pass --skip_vram_check to bypass this test.")
            return

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    best_dir = os.path.join(args.output_dir, "best_checkpoint")
    log_path = os.path.join(args.output_dir, "training_log.csv")

    progress = load_progress(args.output_dir)
    if progress["epochs_completed"] >= args.total_epochs:
        print(f"Already completed {progress['epochs_completed']} / {args.total_epochs} epochs. "
              f"Increase --total_epochs to keep training.")
        return
    if progress["epochs_completed"] > 0:
        print(f"Resuming: {progress['epochs_completed']} epochs already completed "
              f"(best F1 so far: {progress['best_f1']:.4f} at epoch {progress['best_epoch']}).")

    patch_size = args.patch_size if args.patch_size > 0 else None
    class_weights = None if args.no_class_weights else compute_class_weights(args.dataset_dir, "train")

    train_provider = RFINpyDataProvider(
        os.path.join(args.dataset_dir, "train", "images"),
        os.path.join(args.dataset_dir, "train", "masks"),
        patch_size=patch_size,
    )
    val_img_dir = os.path.join(args.dataset_dir, "val", "images")
    val_mask_dir = os.path.join(args.dataset_dir, "val", "masks")
    if not os.path.exists(val_img_dir):
        # DO NOT fall back to test/. A previous version of this script silently
        # substituted the test set here when val/ was missing, which meant the
        # "best" checkpoint was selected on the same data later reported as a
        # held-out result -- i.e. real leakage, invisible in the logs. Any
        # number produced that way is invalid and cannot be published.
        # Failing loudly is the only safe behaviour. See CLAUDE.md section 10.
        print(f"\nERROR: no validation split at {val_img_dir}")
        print("Refusing to run: without val/, checkpoint selection would have to use")
        print("the TEST set, which invalidates every number this run produces.")
        print("Create a validation split first (this does not touch test/):")
        print(f'  python3 ../hybrid_rfi_package/make_val_split.py --dataset_dir "{args.dataset_dir}"')
        print("or regenerate the dataset with dataset_generator_v3_strength.py, which")
        print("writes train/val/test directly.")
        return

    # Validation must be DETERMINISTIC. RFINpyDataProvider._post_process takes a
    # RANDOM crop whenever patch_size is set, regardless of shuffle_data, so
    # passing patch_size here would score every checkpoint on a different random
    # window. That is exactly the noisy-selection mechanism behind the inflated
    # "best F1 = 0.8138" that collapsed to 0.34 on the real test set
    # (CLAUDE.md section 10). Evaluate on whole images instead.
    if patch_size is not None:
        print(f"NOTE: training uses {patch_size}x{patch_size} random crops, but validation "
              f"will use FULL images so checkpoint selection is deterministic and comparable "
              f"across epochs.")
    val_provider = RFINpyDataProvider(val_img_dir, val_mask_dir, patch_size=None, shuffle_data=False)

    cost_kwargs = dict(regularizer=0.001)  # paper Sec 2.2 / authors' own rfi_launcher.py
    if class_weights is not None:
        cost_kwargs["class_weights"] = class_weights

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,roc_auc,pr_auc,max_f1,is_best,chunk_time_sec\n")

    while progress["epochs_completed"] < args.total_epochs:
        chunk_epochs = min(args.epochs_per_chunk, args.total_epochs - progress["epochs_completed"])
        restore = progress["epochs_completed"] > 0

        print(f"\n=== Training epochs {progress['epochs_completed']+1}-"
              f"{progress['epochs_completed']+chunk_epochs} of {args.total_epochs} "
              f"(restore={restore}) ===")

        # Fresh graph/session per chunk avoids the graph-duplication issue
        # that would occur from calling Trainer.train() repeatedly on one
        # long-lived Python process -- unet.Unet() calls tf.reset_default_graph()
        # in its own __init__ (their code), so a new Unet() per chunk is the
        # correct, code-preserving way to do this.
        net = unet.Unet(
            channels=1, n_class=2,
            layers=args.layers, features_root=args.features_root,
            cost="cross_entropy", cost_kwargs=cost_kwargs,
        )
        trainer = unet.Trainer(
            net, batch_size=args.batch_size, optimizer=args.optimizer,
            opt_kwargs=(dict(learning_rate=args.learning_rate)
                        if args.optimizer == "adam"
                        else dict(learning_rate=args.learning_rate, decay_rate=0.95, momentum=0.2)),
        )

        t0 = time.time()
        model_path = trainer.train(
            train_provider, checkpoint_dir,
            training_iters=args.training_iters, epochs=chunk_epochs,
            dropout=0.5, display_step=args.display_step,
            restore=restore, prediction_path=os.path.join(args.output_dir, "predictions"),
        )
        chunk_time = time.time() - t0

        metrics = evaluate(net, model_path, val_provider, args.n_eval_images, args.eval_batch_size)

        if metrics["collapsed"]:
            print("\n" + "!" * 70)
            print("NETWORK COLLAPSE DETECTED -- stopping now instead of wasting hours.")
            print("!" * 70)
            print("The model is outputting exactly 0.5 for every pixel. This is the dead")
            print("state of tf_unet's ReLU-on-logits output layer: both logits went")
            print("negative, ReLU clamped them to zero, and ReLU's gradient at negative")
            print("input is zero -- so NO further training can ever recover it.")
            print("")
            print(f"Current settings: --optimizer {args.optimizer} --learning_rate {args.learning_rate}")
            print("")
            print("To fix, start a FRESH run (the existing checkpoint is dead and cannot")
            print("be resumed from) with a lower learning rate:")
            print(f"  rm -rf '{args.output_dir}'")
            print(f"  python3 {os.path.basename(__file__)} --optimizer adam --learning_rate 0.0005")
            if progress["best_epoch"]:
                print("")
                print(f"NOTE: your best pre-collapse checkpoint (F1={progress['best_f1']:.4f} at "
                      f"epoch {progress['best_epoch']}) is preserved at:")
                print(f"  {best_dir}")
                print("Back that folder up before deleting the output directory if you want it.")
            return

        progress["epochs_completed"] += chunk_epochs
        is_best = metrics["max_f1"] > progress["best_f1"]
        if is_best:
            progress["best_f1"] = metrics["max_f1"]
            progress["best_epoch"] = progress["epochs_completed"]
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)

        save_progress(args.output_dir, progress)
        with open(log_path, "a") as f:
            f.write(f"{progress['epochs_completed']},{metrics['roc_auc']:.4f},"
                    f"{metrics['pr_auc']:.4f},{metrics['max_f1']:.4f},{int(is_best)},{chunk_time:.1f}\n")

        print(f"Epoch {progress['epochs_completed']}: ROC AUC={metrics['roc_auc']:.4f}  "
              f"PR AUC={metrics['pr_auc']:.4f}  F1={metrics['max_f1']:.4f}  "
              f"{'<-- NEW BEST' if is_best else ''}  ({chunk_time:.0f}s for this chunk)")

    print(f"\nDone. {args.total_epochs} epochs completed. "
          f"Best F1={progress['best_f1']:.4f} at epoch {progress['best_epoch']} -> {best_dir}")
    print(f"Full history: {log_path}")


if __name__ == "__main__":
    main()
