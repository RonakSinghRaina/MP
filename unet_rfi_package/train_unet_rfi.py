"""
Trains and evaluates the original tf_unet U-Net (Akeret et al. 2017,
"Radio frequency interference mitigation using deep convolutional neural
networks") on our synthetic GMRT-style RFI dataset.

WHAT THIS SCRIPT DOES vs. WHAT IT DOESN'T DO
----------------------------------------------
- The tf_unet package (unet.py, layers.py, image_util.py, util.py) is used
  completely UNMODIFIED, cloned verbatim from
  https://github.com/jakeret/tf_unet. Not a single line inside that package
  is edited.
- The only new code here is (a) a data provider that reads OUR .npy dataset
  instead of the .tif images tf_unet's own ImageDataProvider expects, and
  (b) a thin training/evaluation harness. This is exactly the extension
  mechanism the package's own docstring describes: "Subclasses have to
  overwrite the `_next_data` method that load the next data and label
  array." ImageDataProvider itself is built the same way, just for .tif
  files.
- Model architecture, loss function, optimizer, weight init, dropout,
  and training loop are all the authors' original code, untouched.

WHY THE COMPAT SHIM AT THE TOP
--------------------------------
tf_unet was written for TensorFlow 1.x (tf.placeholder, tf.Session, etc.).
True TensorFlow 1.x is not installable on modern Python, so we run the
UNMODIFIED code through TensorFlow 2's official `tf.compat.v1` shim with v2
behavior disabled. This is the standard, code-preserving way to run legacy
TF1 code on TF2 -- it changes zero lines inside tf_unet itself.

WHY NOT tf_unet's SimpleDataProvider
--------------------------------------
tf_unet ships a `SimpleDataProvider` whose docstring says to pass
already-one-hot labels of shape [n, X, Y, classes]. But the base class's
`_process_labels` it inherits unconditionally re-one-hots a *raw 2D
boolean* mask -- this is a real bug in the published package for the
n_class=2 case (confirmed by reproducing the crash). Using
`SimpleDataProvider` as documented throws a broadcast error. Instead we use
the officially documented, intended extension point: subclass
`BaseDataProvider` and override only `_next_data`, exactly like
`ImageDataProvider` does internally.
"""

import os
import sys
import glob
import json
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# 1. TensorFlow 1.x compatibility shim (no tf_unet code touched)
# ---------------------------------------------------------------------------
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
sys.modules["tensorflow"] = tf1  # so `import tensorflow as tf` inside tf_unet gets the v1 shim

TF_UNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_unet")
sys.path.insert(0, TF_UNET_PATH)

from tf_unet import unet, util, image_util  # noqa: E402  (unmodified upstream package)


# ---------------------------------------------------------------------------
# 2. Data provider for our synthetic .npy dataset
#    (This is the ONLY new code that touches the data pipeline. It follows
#    the exact same shape as tf_unet's own ImageDataProvider._next_data.)
# ---------------------------------------------------------------------------
class RFINpyDataProvider(image_util.BaseDataProvider):
    """
    Feeds our synthetic RFI spectrogram/mask .npy pairs into the unmodified
    tf_unet training loop. Overrides only `_next_data`, the documented
    extension point of `BaseDataProvider`.
    """
    channels = 1
    n_class = 2

    def __init__(self, images_dir, masks_dir, a_min=None, a_max=None, shuffle_data=True):
        super(RFINpyDataProvider, self).__init__(a_min, a_max)
        self.shuffle_data = shuffle_data
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.npy")))
        assert len(self.image_files) == len(self.mask_files) and len(self.image_files) > 0, (
            f"No matching image/mask .npy pairs found in {images_dir} / {masks_dir}"
        )
        self.file_idx = -1
        self._shuffle()
        print(f"RFINpyDataProvider: {len(self.image_files)} image/mask pairs from {images_dir}")

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
        mask = np.load(self.mask_files[self.file_idx]).astype(bool)  # raw 2D boolean, as BaseDataProvider expects
        return img, mask


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
def train(args):
    train_provider = RFINpyDataProvider(
        os.path.join(args.dataset_dir, "train", "images"),
        os.path.join(args.dataset_dir, "train", "masks"),
    )
    val_img_dir = os.path.join(args.dataset_dir, "val", "images")
    val_mask_dir = os.path.join(args.dataset_dir, "val", "masks")
    if not os.path.exists(val_img_dir):
        val_img_dir = os.path.join(args.dataset_dir, "test", "images")
        val_mask_dir = os.path.join(args.dataset_dir, "test", "masks")

    val_provider = RFINpyDataProvider(
        val_img_dir,
        val_mask_dir,
        shuffle_data=False,
    )

    # Network configuration: 3 layers, 64 features -- the configuration the
    # paper (Akeret et al. 2017, Section 3) reports as the best
    # performance/cost trade-off (their "red solid line" in Fig. 3).
    # NOTE: the authors' own scripts/rfi_launcher.py in the repo defaults to
    # layers=5 instead -- the paper's prose recommends 3, their shipped
    # script defaults to 5. We follow the paper's stated conclusion (3) but
    # expose --layers so this is trivial to switch.
    # cost_kwargs regularizer=0.001 matches both the paper text ("L2
    # regularizer of strength equal to 10^-3", Sec 2.2) and their own
    # rfi_launcher.py script -- this was missing from an earlier draft of
    # this harness and has been added to match the paper exactly.
    net = unet.Unet(
        channels=1,
        n_class=2,
        layers=args.layers,
        features_root=args.features_root,
        cost="cross_entropy",
        cost_kwargs=dict(regularizer=0.001),
    )

    trainer = unet.Trainer(
        net,
        batch_size=args.batch_size,
        optimizer="momentum",
        opt_kwargs=dict(learning_rate=0.2, decay_rate=0.95, momentum=0.2),  # paper's Sec. 2.2 settings
    )

    output_path = os.path.join(args.output_dir, "checkpoints")
    model_path = trainer.train(
        train_provider,
        output_path,
        training_iters=args.training_iters,
        epochs=args.epochs,
        dropout=0.5,  # paper's Sec. 2.2 dropout rate
        display_step=args.display_step,
        restore=args.restore,
        prediction_path=os.path.join(args.output_dir, "predictions"),
    )
    print(f"Trained model checkpoint: {model_path}")
    return net, model_path, val_provider


# ---------------------------------------------------------------------------
# 4. Evaluation: ROC / PR / F1, matching the paper's reported metrics
# ---------------------------------------------------------------------------
def evaluate(net, model_path, test_provider, output_dir, n_eval_images):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

    x_test, y_test = test_provider(n_eval_images)
    prediction = net.predict(model_path, x_test)

    y_true = util.crop_to_shape(y_test, prediction.shape)
    y_true_rfi = y_true[..., 1].flatten()
    y_pred_rfi = prediction[..., 1].flatten()

    fpr, tpr, _ = roc_curve(y_true_rfi, y_pred_rfi)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true_rfi, y_pred_rfi)
    pr_auc = auc(recall, precision)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_f1 = float(np.max(f1_scores))

    results = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "max_f1": best_f1,
        "n_eval_images": n_eval_images,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation on synthetic test set:")
    print(f"  ROC AUC : {roc_auc:.4f}   (paper's simulated-data result: ~0.96)")
    print(f"  PR  AUC : {pr_auc:.4f}   (paper's simulated-data result: ~0.92)")
    print(f"  Max F1  : {best_f1:.4f}  (paper's simulated-data result: ~0.85)")

    # Save a visual comparison panel using tf_unet's own util.combine_img_prediction
    img = util.combine_img_prediction(x_test, y_test, prediction)
    util.save_image(img, os.path.join(output_dir, "prediction_panel.jpg"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Train tf_unet (Akeret et al. 2017) on the synthetic RFI dataset")
    parser.add_argument("--dataset_dir", default="./Synthetic Dataset",
                         help="Root of the dataset (must contain train/, val/, test/ subfolders)")
    parser.add_argument("--output_dir", default="./unet_run")
    parser.add_argument("--layers", type=int, default=3, help="U-Net depth (paper: 3)")
    parser.add_argument("--features_root", type=int, default=64, help="Features in first layer (paper: 64)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--training_iters", type=int, default=32, help="Steps per epoch (paper: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Paper: 100")
    parser.add_argument("--display_step", type=int, default=5)
    parser.add_argument("--n_eval_images", type=int, default=20)
    parser.add_argument("--restore", action="store_true", default=True, help="Restore from existing checkpoint if available (default: True)")
    args = parser.parse_args()

    net, model_path, val_provider = train(args)
    evaluate(net, model_path, val_provider, os.path.join(args.output_dir, "eval"), args.n_eval_images)


if __name__ == "__main__":
    main()
