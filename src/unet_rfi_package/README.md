# Applying Akeret et al. (2017) tf_unet to the Synthetic RFI Dataset

## What's in this package

```
tf_unet/               <- the ORIGINAL package, cloned verbatim from
                           https://github.com/jakeret/tf_unet (commit 0dcdf2f,
                           2020-05-05). Not a single line inside tf_unet/*.py
                           has been edited.
train_unet_rfi.py      <- the only new code: applies tf_unet to our dataset
README.md              <- this file
```

## What was changed, and what wasn't

**Untouched (100% the authors' original code):**
- `tf_unet/unet.py` — the U-Net architecture, the `Trainer` class, the
  cross-entropy/dice cost functions, the momentum optimizer setup
- `tf_unet/layers.py` — conv/deconv/pooling/crop-and-concat ops, weight init
- `tf_unet/util.py` — image cropping, prediction visualization helpers
- `tf_unet/image_util.py` — `BaseDataProvider`, `ImageDataProvider` (the
  built-in providers for `.tif` files)

**New, in `train_unet_rfi.py`:**
1. **`RFINpyDataProvider`** — a data loader for our `.npy` spectrogram/mask
   pairs. This is *not* a change to their algorithm — it's the extension
   point their own package documents: `BaseDataProvider`'s docstring says
   *"Subclasses have to overwrite the `_next_data` method that loads the
   next data and label array."* `RFINpyDataProvider` does exactly that and
   nothing else; everything downstream (normalization, one-hot conversion,
   batching) is their unmodified code.
2. A thin `train()` / `evaluate()` harness that wires their `Unet` and
   `Trainer` classes to our dataset folders and reports ROC AUC / PR AUC /
   max F1 the same way the paper does (Section 3, Fig. 3).
3. A TensorFlow-1-to-2 compatibility shim (`tf.compat.v1` + `disable_v2_behavior()`).
   True TensorFlow 1.x isn't installable on modern Python, so this is the
   standard way to run *unmodified* legacy TF1 code on a current TF
   install — it doesn't touch a single line inside `tf_unet/`.

## One bug found (and worked around, not silently papered over)

`tf_unet`'s own `SimpleDataProvider` — the class its docs say to use for
numpy-array datasets — is actually broken for binary segmentation
(`n_class=2`): its docstring says to pass pre-one-hot labels, but the
`_process_labels` method it inherits assumes a raw 2D boolean mask and
tries to one-hot it again, throwing a shape error. This is a real,
previously-reported issue in the published package. We didn't patch it —
we simply used the officially documented alternative extension point
(`BaseDataProvider._next_data`) instead, so no upstream code needed
changing.

## Network configuration

Following the paper's own conclusion (Section 3): **3 layers, 64 features
in the first layer** — the configuration the authors report as the best
balance of prediction performance and compute cost. Optimizer, learning
rate (0.2, momentum-based, exponential decay), and dropout (0.5) all match
Section 2.2 of the paper.

## Running it

```bash
pip install tensorflow scikit-learn numpy pillow matplotlib

python train_unet_rfi.py \
    --dataset_dir /path/to/6_Simulated_Dataset \
    --output_dir  /path/to/output \
    --layers 3 --features_root 64 \
    --batch_size 1 --training_iters 32 --epochs 100
```

This mirrors the paper's own training regime (100 epochs, 32 iterations/epoch,
batch size effectively 1 image per step as in the original TOD experiments).
On CPU this will be slow for 1024x1024 images — a GPU (even a modest one) is
strongly recommended, matching the paper's own use of an NVIDIA K20.

Output:
- `output_dir/checkpoints/` — trained model checkpoint
- `output_dir/predictions/` — per-epoch prediction preview images (tf_unet's own diagnostic output)
- `output_dir/eval/eval_metrics.json` — ROC AUC, PR AUC, max F1 on held-out data
- `output_dir/eval/prediction_panel.jpg` — input / ground truth / prediction comparison image

## Verified working

This script was smoke-tested end-to-end against real files from the
generated `6_Simulated_Dataset/` (small crops, 1 epoch, tiny network) to
confirm the full pipeline — data loading, training, checkpoint save/restore,
and evaluation — runs correctly before handing it over. Full-scale training
was not run in this environment (no GPU, and 1024x1024 x 700 images x 100
epochs is multi-hour+ compute), so accuracy numbers here are illustrative
only; use the eval numbers from your own real training run.
