# Hybrid RFI Model (separate from the Akeret baseline)

This folder is **completely independent** of `unet_rfi_package`. It shares
nothing and overwrites nothing, so your already-trained baseline stays intact
for your lab presentation.

- `unet_rfi_package/`  -> the Akeret et al. baseline (contains the authors'
  original `tf_unet` code). **Leave it alone.**
- `hybrid_rfi_package/` -> this folder. Pure PyTorch. Does not use `tf_unet`.

Both read the SAME `Synthetic Dataset/` folder, so put this folder beside
`unet_rfi_package` inside `Minor Project/`.

## Files

| File | Purpose |
|---|---|
| `hybrid_model.py` | HybridRFINet architecture (residual + multiscale strip conv + ECA) |
| `train_hybrid.py` | Training with val-only model selection |
| `make_val_split.py` | Creates `val/` by moving images out of `train/` (test untouched) |
| `evaluate_hybrid_test.py` | Final test-set evaluation, run ONCE |
| `dataset_generator_v3_strength.py` | Generator that also saves RFI strength maps (optional, for weak/medium/strong analysis later) |

## How to run

> **Do NOT install torch into `~/tf-env`.** An earlier version of this README
> said `source ~/tf-env/bin/activate` then `pip install torch`. That is the exact
> action `CLAUDE.md` §3 documents as having broken this project once: installing
> torch silently downgrades the cuDNN build TensorFlow needs, and the `tf_unet`
> baseline then dies with a `CuDNN version mismatch`. The two frameworks require
> incompatible cuDNN versions and cannot share one environment. Use
> `~/torch-env` for everything in this folder.

```bash
source ~/torch-env/bin/activate          # NOT tf-env -- see the warning above
cd "Minor Project/hybrid_rfi_package"
pip install -r ../requirements-torch.txt

python3 train_hybrid.py \
    --dataset_dir "../Synthetic Dataset 276x600" \
    --output_dir "../hybrid_run_paperdim" \
    --patch_size 0 --batch_size 8 --n_val_images 150 \
    --deterministic --early_stop_patience 3

# ONCE, at the very end. --patch_size 0 is now the default, but pass it
# explicitly so the command is self-documenting.
python3 evaluate_hybrid_test.py \
    --dataset_dir "../Synthetic Dataset 276x600" \
    --output_dir "../hybrid_run_paperdim" \
    --patch_size 0 --per_image_csv --strength_report
```

Rerunning the same `train_hybrid.py` command resumes from `progress.json` after
an interruption.

## `make_val_split.py` is NOT needed for the 276x600 dataset

`dataset_generator_v3_strength.py` already writes `train/` (700), `val/` (150)
and `test/` (150) directly, and stamps a `split` field into each
`metadata.jsonl` record. On that dataset `make_val_split.py` prints
*"val/ already exists -- nothing to do"* and changes nothing.

Only run it on a dataset that has `train/` and `test/` but no `val/`. If you do,
it moves 150 images out of `train/`, so training then sees **550** images, not
700 — and any baseline retrained afterwards is no longer comparable to one
trained on 700. **State in the write-up which of the two situations applies**;
`RFI_Project_Model_Comparison.md` currently says 700 while this file used to
imply 550, and nothing in the repository resolves it.

## Preserved from the baseline debugging work

Min-max normalization WITHOUT `fabs()` (the baseline's own normalization took
absolute values first, turning strong negative noise dips into
maximum-brightness pixels the mask still called clean), Adam @ 1e-3 (not the
paper's 0.2, which killed training at iteration 4), measured class weights,
resumability across restarts, best-checkpoint tracking, VRAM preflight,
one-image-at-a-time evaluation, collapse detection, gradient clipping, and
GroupNorm rather than BatchNorm.

> Note: the small-batch arguments above (`batch_size=1`, 512 px patches) come
> from the original 1024×1024 work. The reported 276×600 result was trained at
> `--batch_size 8 --patch_size 0`. GroupNorm is still the right choice at batch
> 8, but do not repeat "BatchNorm is meaningless at batch_size=1" as the
> justification for the run that was actually published.
