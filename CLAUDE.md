# Project Context: RFI Detection — Baseline vs. Hybrid U-Net

This file is read automatically by Claude Code at the start of every session
in this folder. It exists so a fresh session does not repeat mistakes that
already cost real debugging time. Read `RFI_Project_Model_Comparison.md` in
this same folder for full model results and file locations — this file is
about *environment, hardware, and precautions*, not results.

---

## 1. Hardware — hard constraints, not guidelines

- **GPU:** NVIDIA RTX 3060 Laptop, **6 GB VRAM**, 130W TGP.
- **Effective usable VRAM is ~3.5 GB, not 6 GB.** Windows' own display
  compositor shares the same physical GPU. TensorFlow sessions have
  repeatedly shown a hard allocator limit around 3.5 GiB in practice.
  Always assume ~3.5 GB when reasoning about whether a batch size/image
  size/model width will fit — do not use the 6 GB spec sheet number.
- GPU has been observed running at **82°C under sustained load**. Not
  immediately dangerous (laptop GPUs throttle themselves near 87-90°C), but
  worth mentioning if a long run seems slower than expected — may be
  thermal throttling. Recommend a hard flat surface with airflow for
  multi-hour runs.
- **Do not let Windows sleep during long training runs.** Locking the
  screen (Win+L) is safe and does not interrupt anything. Sleep does.
  Check Settings > System > Power & battery > Sleep is set to Never while
  plugged in before starting any run expected to take over ~30 minutes.

## 2. OS / environment — why it's built this way

- Windows 11, with **WSL2 (Ubuntu)** as the actual working environment for
  all GPU training.
- **Why WSL at all:** TensorFlow dropped native Windows GPU support after
  v2.10. Native Windows TensorFlow silently falls back to CPU regardless of
  drivers. `tensorflow-directml-plugin` (the old workaround) is discontinued
  and only supports Python ≤3.10 — not a viable option. WSL2 + a current
  NVIDIA driver is the only remaining path to real TensorFlow GPU
  acceleration on this machine.
- PyTorch, by contrast, DOES support native Windows GPU — but for
  consistency (and because the TensorFlow baseline has no choice), the
  PyTorch/hybrid work also runs inside WSL, in the same environment style.
- Project files live on the **Windows filesystem**, accessed from WSL via
  `/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project/...` — this is
  fine for reading/running scripts. It is NOT fine for creating Python
  virtual environments (see below).

## 3. Two separate Python virtual environments — do not merge them

| Env | Purpose | Location | Key package |
|---|---|---|---|
| `tf-env` | TensorFlow / the authors' `tf_unet` baseline | `~/tf-env` | `nvidia-cudnn-cu12` **9.24.x** |
| `torch-env` | PyTorch / the hybrid model | `~/torch-env` | `nvidia-cudnn-cu12` **9.1.x** |

**These two environments were originally the same one, and it broke.**
`pip install torch` silently downgraded the cuDNN version TensorFlow needed,
producing a `CuDNN version mismatch` crash mid-project. TensorFlow and
PyTorch on this exact setup **require different, incompatible cuDNN
versions** and cannot coexist reliably in one environment. Always confirm
which env is active (`which python3`, or check the venv name in the
prompt) before installing anything, and never `pip install` a
framework-specific package into the wrong env.

**Virtual environments must be created inside the Linux filesystem
(`~/`), never on `/mnt/c/...`.** `python -m venv` relies on symlinks;
Windows drives mounted into WSL (`drvfs`) do not support them reliably,
causing a silent broken venv (activation script exists but package
installs fail with confusing "externally-managed-environment" errors).

Reading/running project files from `/mnt/c/...` is fine — only *venv
creation* must happen on native Linux filesystem.

## 4. GPU detection gotchas

- `pip install tensorflow` alone does **not** bundle CUDA/cuDNN. Use
  `pip install "tensorflow[and-cuda]"`.
- Even with that, TensorFlow may still fail to find the libraries at
  runtime with `Cannot dlopen some GPU libraries`. Fix: set
  `LD_LIBRARY_PATH` to include every `site-packages/nvidia/*/lib` folder.
  This needs to be permanent (added to `~/.bashrc`), not just exported
  once per session.
- Always set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow imports.
  Without it, TF grabs ~all GPU memory at session start, competing with
  Windows' own display use of the same card.
- `nvidia-smi` run from **Windows PowerShell** often shows
  `No running processes found` even while a WSL process is actively using
  the GPU at 100%. This is a known WSL/Windows reporting gap, not evidence
  of a hang. Trust the `GPU-Util %` and `Memory-Usage` numbers over the
  process list when checking from Windows.
- If WSL's GPU passthrough seems broken after any driver update, run (from
  Windows PowerShell, not WSL): `wsl --update` then `wsl --shutdown`,
  then reopen WSL.

## 5. VRAM preflight checks — a mistake pattern that recurred twice

A "does this fit in memory" preflight check was added to both the
TensorFlow and PyTorch training scripts, and **both had the same bug
independently**: they tested a *square* guessed image size instead of the
*real* dataset image shape (e.g. tested `276x276` when the real images are
`276x600`, 2.17x smaller than reality — the check passed, then training
still OOM'd for real). When writing or reviewing any memory preflight
check: **always probe using the actual image shape read from a real file
in the dataset, never an assumed/square size**, and pass the *true* batch
size, not 1.

## 6. Batch size / architecture width — what actually fits

At `276x600` (the paper-matched dataset), on this GPU's real ~3.5 GB:

- TensorFlow `tf_unet` baseline: `layers=3, features_root=64, batch_size=8`
  does **not** fit. `batch_size=4, features_root=32` does.
- PyTorch hybrid model: `batch_size=8` at full model width fits
  comfortably (uses GroupNorm + more efficient ops, different memory
  profile than tf_unet's valid-padding conv stack).

If asked to increase either further, treat it as untested and recommend
running the VRAM preflight check first, not assuming it will fit.

## 7. Do not use the paper's original learning rate at low batch size

The authors' paper (Akeret et al. 2017) specifies momentum optimizer,
`lr=0.2`, trained at batch size 32. **This setting reliably kills the
network when batch size is forced down** (as it is here, due to VRAM
limits) — `tf_unet`'s output layer applies ReLU directly to the logits;
once both logits go negative, ReLU zeroes them, softmax outputs exactly
`[0.5, 0.5]`, and gradient becomes exactly zero — an unrecoverable dead
state. Measured directly: this happened at iteration 4 with the paper's
lr=0.2 at batch_size=1. Both training scripts in this project now default
to `optimizer=adam, lr=0.001` for this reason, with automatic collapse
detection (checks for `loss ≈ ln(2) = 0.6931` and `ROC AUC ≈ 0.5000`, which
is the exact signature of this dead state).

## 8. tf_unet's "epoch" is not a full pass over the data

`tf_unet.Trainer.train(..., epochs=N, training_iters=M)` runs exactly
`M` gradient steps per "epoch", not one pass over the training set. For
any fair comparison against the PyTorch hybrid (where one epoch genuinely
is one full pass), `training_iters` must be explicitly set to
`(number of training images) / batch_size`. This was missed once already
and would have unfairly under-trained the baseline by ~26%. See
`run_fair_comparison.py`, which computes this automatically — prefer using
it over calling `train_unet_rfi_gpu.py` directly for any baseline-vs-hybrid
comparison work.

## 9. Evaluation must be batched one image at a time

Running `net.predict()` / a forward pass on many evaluation images at once
(e.g. 10-20 in a single batch) can OOM even when training itself fit fine,
because evaluation batches don't get the same memory-saving treatment
training steps do. All evaluation code in this project processes one image
at a time and concatenates results before computing metrics. Preserve this
pattern in any new evaluation code.

## 10. Checkpoints — which one is "the model," and a real leakage bug already found and fixed

- Always use `best_checkpoint/` (TensorFlow) or `best.pt` (PyTorch), never
  the plain `checkpoints/`/`last.pt` — those are just the most recent
  epoch, not the best one.
- **A checkpoint's architecture (layers, features_root / base, depth) must
  match exactly to restore it.** Attempting to restore a `features_root=32`
  checkpoint into a `features_root=64` graph fails with a shape-mismatch
  error. If unsure what a checkpoint was trained with, inspect it directly
  rather than assuming:
  ```python
  import tensorflow.compat.v1 as tf1
  tf1.disable_v2_behavior()
  for name, shape in tf1.train.list_variables('path/to/model.ckpt'):
      print(name, shape)
  ```
- **Model selection (picking the "best" checkpoint) must use the
  VALIDATION set, never the test set, and must use enough validation
  images to be reliable.** An early run selected checkpoints using only 10
  random validation patches; because per-image RFI content varies wildly
  (0-60% of pixels), taking the *max* F1 across ~20 such noisy checks
  produced a fake "best" score (0.81) that collapsed to the real value
  (0.34) when evaluated properly on the full test set — a classic
  winner's-curse artifact. All current scripts use the full validation
  set (150 images) for this reason. Do not reduce this to save time
  without flagging the risk.
- **The test set is touched exactly once**, after training is fully done,
  to report a final number. It is never used to pick a checkpoint or tune
  a threshold. If a dataset is missing a separate `val/` split, create one
  from `train/` (see `make_val_split.py`) rather than letting training
  silently fall back to using `test/` for validation — this happened once
  already and was a real (if ultimately non-catastrophic) leakage risk.

## 11. Two datasets exist — know which is which

- `Synthetic Dataset/` — original, 1024×1024. Model 1 (baseline) was
  trained on this.
- `Synthetic Dataset 276x600/` — matches the paper's own image dimensions.
  RFI morphology parameters (blob size, line width, band width, etc.) were
  deliberately rescaled proportionally when this was generated — using the
  1024px-tuned parameters unscaled at this resolution would produce
  physically wrong RFI (oversized or literally unable to fit in the
  smaller frequency axis). Models 2 and 3 both use this dataset and are
  the ones that are fairly comparable to each other. Model 1 (different
  dataset) is a "before" reference point only, not a controlled
  comparison.

## 12. Full command reference

```bash
# TensorFlow baseline work
source ~/tf-env/bin/activate
cd "/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project/unet_rfi_package copy"
python3 run_fair_comparison.py --batch_size 4          # matched comparison run
python3 evaluate_test_set.py --checkpoint_dir "../unet_run_faircompare/best_checkpoint" \
    --dataset_dir "../Synthetic Dataset 276x600" --patch_size 0 --features_root 32

# PyTorch hybrid work
source ~/torch-env/bin/activate
cd "/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project/hybrid_rfi_package"
python3 train_hybrid.py --dataset_dir "../Synthetic Dataset 276x600" \
    --output_dir "../hybrid_run_paperdim" --patch_size 0 --batch_size 8 --n_val_images 150
python3 evaluate_hybrid_test.py --dataset_dir "../Synthetic Dataset 276x600" \
    --output_dir "../hybrid_run_paperdim"
```

Both training scripts are resumable — rerunning the exact same command
after an interruption (laptop restart, Ctrl+C) continues from the last
completed chunk via `progress.json`. Do not change batch_size / features /
architecture flags between a stopped run and its resume — this will not
error cleanly and produces the kind of checkpoint-mismatch confusion
described in §10.

## 13. Current status (update this section as work continues)

See `RFI_Project_Model_Comparison.md` for full numbers. As of the last
session: all three models (baseline-old, baseline-fair-comparison, hybrid)
have completed training and final test-set evaluation. The hybrid
(F1=0.98) substantially outperforms the baseline under matched conditions
(F1=0.39), and this gap has been attributed to architecture rather than
training setup after controlling for dataset, batch composition, optimizer,
and epoch count.
