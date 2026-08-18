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

```bash
source ~/tf-env/bin/activate
cd "Minor Project/hybrid_rfi_package"
pip install torch

python3 make_val_split.py      # once. moves 150 train images -> val/
python3 train_hybrid.py        # rerun the same command to resume after a restart
python3 evaluate_hybrid_test.py   # ONCE, at the very end
```

## IMPORTANT: `make_val_split.py` changes the shared dataset

It moves 150 images from `train/` into a new `val/` folder. Your already
trained baseline model is NOT affected (it is already trained). But if you
ever RE-train the baseline afterwards, it will then be training on 550
images instead of 700, so its numbers would no longer be directly comparable
to your existing baseline result. Note the split in your report.

## Preserved from the baseline debugging work

batch_size=1 and 512px patches (6 GB VRAM), min-max normalization WITHOUT
`fabs()`, Adam @ 1e-3 (not the paper's 0.2, which killed training at
iteration 4), measured class weights, resumability across laptop restarts,
best-checkpoint tracking, VRAM preflight, one-image-at-a-time evaluation,
collapse detection. Plus gradient clipping and GroupNorm (BatchNorm is
meaningless at batch_size=1).
