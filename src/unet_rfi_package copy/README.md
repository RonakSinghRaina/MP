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

## Two scripts in this package

- **`train_unet_rfi.py`** — the original CPU-friendly harness (from earlier in this project).
- **`train_unet_rfi_gpu.py`** — **use this one.** Same unmodified `tf_unet` package underneath, but adds:
  1. A GPU check that warns you *before* a long CPU run starts
  2. Class weighting (`[0.58, 3.65]`-ish, computed from your actual `metadata.jsonl`, not guessed) to fix the RFI/clean pixel imbalance — this uses `tf_unet`'s own documented `cost_kwargs={"class_weights": ...}` parameter, not a code change
  3. Optional patch-based training (`--patch_size`, default 512) to keep VRAM usage bounded on a consumer GPU, since our 1024×1024 images are ~6x larger than the paper's 276×600 training crops
  4. **Epoch-accurate resumability.** tf_unet's own `Trainer.train(restore=True)` correctly restores weights across restarts, but its internal epoch loop always runs `range(epochs)` from zero — it has no memory of epochs already completed in a previous process. A naive resume-after-crash with `--epochs 100` again would silently train 100 *more* epochs on top of whatever you'd already done. This script tracks true cumulative epochs itself in `progress.json` (outside `tf_unet`, so still zero edits to their code) and only ever asks the unmodified `train()` for the epochs still remaining.
  5. **Best-checkpoint tracking.** `tf_unet`'s `train()` just overwrites one checkpoint every epoch — no concept of "best." This script evaluates on your validation set after each chunk of epochs and keeps a separate `best_checkpoint/` copy, so a late epoch that overfits doesn't silently replace an earlier, better one.

Verified end-to-end, including simulating a true crash-and-restart across two separate process invocations (not just one script skipping past a restore flag) — confirmed it resumes at the correct epoch and preserves the correct best-F1 record.

### Why this exists: a note on the PyTorch attempt

If you also have a PyTorch version of this floating around: it was a full
architectural reimplementation (added BatchNorm at batch size 1, no
dropout, no class weighting, different padding/normalization), not a
"small tweak" of the TF code. It trained successfully and got a reasonable
ROC AUC (~0.95) but a much weaker PR AUC/F1 (~0.75/0.72 vs. the paper's
~0.92/0.85) — consistent with unweighted cross-entropy on an ~86/14
imbalanced pixel classification task, which barely dents ranking-based
ROC AUC but hurts precision/recall-based metrics a lot. This GPU script
fixes that by using tf_unet's real, published architecture with the class
weighting it already supports.

## Running the GPU version

```bash
pip install tensorflow scikit-learn numpy pillow matplotlib
# Requires an NVIDIA GPU with CUDA/cuDNN set up; TensorFlow auto-detects it.

python train_unet_rfi_gpu.py \
    --dataset_dir /path/to/6_Simulated_Dataset \
    --output_dir  /path/to/output \
    --layers 3 --features_root 64 \
    --patch_size 512 \
    --total_epochs 100 --epochs_per_chunk 5
```

If your laptop restarts or you Ctrl+C, just re-run the exact same command
— it reads `progress.json` in `output_dir` and continues from the correct
epoch automatically, without asking you to remember how far you got.

Output:
- `output_dir/checkpoints/` — most recent checkpoint (may not be the best)
- `output_dir/best_checkpoint/` — checkpoint with the best validation F1 seen so far
- `output_dir/training_log.csv` — per-chunk ROC AUC / PR AUC / F1 / timing history
- `output_dir/progress.json` — cumulative epoch count + best-F1 record (don't delete this if you want to resume correctly)


## Running the original (CPU-friendly) script

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

## Note for 6GB GPUs (e.g. RTX 3060 Laptop)

`layers=3, features_root=64` at full 1024x1024 (`--patch_size 0`) is a lot
of activation memory for backprop on a 6GB card -- it may not fit. The
script now runs an actual preflight test (one real forward+backward pass
at your target settings) before committing to a long run, and fails fast
with concrete suggestions if it doesn't fit, instead of finding out an
hour in. It also sets `TF_FORCE_GPU_ALLOW_GROWTH=true` automatically so
TensorFlow doesn't grab your whole 6GB at startup and compete with your
laptop's own display.

If the preflight check fails, in order of preference (least to most
impact on fidelity to the paper's config):
1. Lower `--patch_size` (e.g. 384 or 256) -- keeps the same network, trains on smaller crops
2. Lower `--features_root` (e.g. 32) -- a genuinely smaller network
3. `--skip_vram_check` only if you just want to bypass the test itself (doesn't change memory usage)

## IMPORTANT: why the optimizer default deviates from the paper

The paper (Sec 2.2) uses momentum with learning rate 0.2. **Do not use that
at batch_size=1.** The paper used mini-batch size 32; at batch size 1 the
per-step gradient noise is far higher, and lr=0.2 reliably drives the network
into an irrecoverable dead state.

The mechanism is specific and worth understanding: `tf_unet/unet.py` line ~145
applies `tf.nn.relu()` to the final logits before the softmax. If both logits
go negative, ReLU clamps both to exactly 0, `softmax([0,0])` is exactly
`[0.5, 0.5]`, and **ReLU's gradient at negative input is exactly zero** -- so
no gradient flows back and the network can never recover, however long you
train. The signature is unmistakable: loss pinned at exactly 0.6931 (= ln 2),
ROC AUC exactly 0.5000, every predicted probability 0.5.

Measured directly on this dataset (same seed, same data order, 250 iterations):

| optimizer            | outcome        | ROC AUC | PR AUC | F1     |
|----------------------|----------------|---------|--------|--------|
| momentum lr=0.2      | DEAD at iter 4 | 0.5000  | 0.5484 | 0.1764 |
| momentum lr=0.01     | alive          | 0.5539  | 0.1250 | 0.2163 |
| **adam lr=0.001**    | **alive**      | **0.8447** | **0.6509** | **0.6506** |

So the default is now `--optimizer adam --learning_rate 0.001`. The model
architecture, loss function, dropout, L2 regularization, and layer/feature
counts all remain exactly the paper's. Only the optimizer changed, and only
because the paper's own setting is incompatible with the batch size a 6GB
consumer GPU can hold.

The script also now auto-detects this collapse after every chunk and aborts
immediately with instructions, rather than letting you burn hours training a
network that is provably incapable of improving.
