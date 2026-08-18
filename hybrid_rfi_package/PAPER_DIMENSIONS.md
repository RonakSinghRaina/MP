# Matching the Paper's Image Dimensions (276 x 600)

## Why

Akeret et al. (2017) Sec. 2.2 trained on images of **276 x 600** pixels with a
**mini-batch size of 32**, on an NVIDIA Kepler K20 (5 GB).

Your RTX 3060 Laptop has **6 GB** -- MORE than their K20 -- so hardware was
never the limitation. The limitation was that our images were 1024x1024
(1,048,576 px), **6.3x larger** than theirs (165,600 px). That forced
batch_size=1, which caused noisy gradients, training instability, and killed
the network at the paper's learning rate.

Regenerating at 276x600 removes that confound and allows a much fairer
reproduction.

## IMPORTANT: RFI morphology is now scaled with image size

The generator's pixel-valued parameters were tuned for 1024x1024. Used
unchanged at 276x600 they would produce physically wrong RFI -- e.g. a
40-channel blob is 3.9% of a 1024-channel band but 14.5% of a 276-channel
band, and a 300-channel broadband block cannot even fit in 276 channels.

Seven parameters now scale proportionally with image dimensions: band-edge
rolloff, narrowband line width, wideband burst duration, broadband block
width, persistent band width, blob size, and scattered cluster size.
Parameters expressed in MHz (the bandpass gain curve) are already
resolution-independent and are deliberately NOT scaled.

Verified: RFI pixel fraction stays comparable across resolutions
(12.4% at 276x600 vs 14.5% at 1024x1024).

## Regenerate the dataset

```bash
python3 dataset_generator_v3_strength.py \
    --n_images 1000 --n_freq 276 --n_time 600 \
    --output_dir "../Synthetic Dataset 276x600" --seed 42
```

~1.3 GB instead of ~8.4 GB, and much faster to train on.

## Retrain the BASELINE with the paper's original settings

Now that images are small enough for a real batch size, the paper's own
optimizer settings become viable. No code change needed -- just flags:

```bash
cd "../unet_rfi_package copy"
python3 train_unet_rfi_gpu.py \
    --dataset_dir "../Synthetic Dataset 276x600" \
    --output_dir "../unet_run_paperdim" \
    --patch_size 0 --batch_size 32 \
    --optimizer momentum --learning_rate 0.2 \
    --layers 3 --features_root 64 \
    --total_epochs 100
```

`--patch_size 0` means no cropping: train on whole 276x600 images, exactly
like the paper.

**Caveat, stated honestly:** in a short 15-iteration test at 276x600, the
paper's lr=0.2 survived where it had died at iteration 4 on 512x512 patches.
That test used a smaller network (16 vs 64 features) and few iterations, so
it is SUGGESTIVE, NOT CONCLUSIVE. Collapse detection remains enabled -- if
the loss pins at exactly 0.6931 and ROC AUC at 0.5000, the run aborts and
tells you. If that happens, fall back to:
`--optimizer adam --learning_rate 0.001`

If batch 32 does not fit in VRAM, the preflight check will say so; try 16 or 8.

## Retrain the HYBRID on the same data

```bash
cd ../hybrid_rfi_package
python3 make_val_split.py --dataset_dir "../Synthetic Dataset 276x600"
python3 train_hybrid.py \
    --dataset_dir "../Synthetic Dataset 276x600" \
    --output_dir "../hybrid_run_paperdim" \
    --patch_size 0 --batch_size 8 \
    --n_val_images 150
```

Then once, at the very end:

```bash
python3 evaluate_hybrid_test.py \
    --dataset_dir "../Synthetic Dataset 276x600" \
    --output_dir "../hybrid_run_paperdim" --patch_size 0
```

## Keep your old results

Generate into a NEW folder (`Synthetic Dataset 276x600`) and use NEW output
directories. Your existing 1024x1024 dataset, trained baseline, and results
stay untouched for comparison in your report.
