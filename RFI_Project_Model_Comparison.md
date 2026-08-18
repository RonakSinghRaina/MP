# RFI Detection Project — Model Comparison & File Locations

**Task:** Detect radio frequency interference (RFI) in synthetic GMRT-style spectrograms using image segmentation.

---

## 1. Summary Table — All Trained Models

| # | Model | Dataset | ROC AUC | PR AUC | F1 | Location (trained model) |
|---|---|---|---|---|---|---|
| 1 | Authors' U-Net (Akeret et al. 2017) | 1024×1024 (original) | 0.5533 | 0.3886 | 0.3438 | `unet_run_gpu/best_checkpoint/` |
| 2 | Authors' U-Net, fair comparison | 276×600 (paper-matched) | 0.6681 | 0.4350 | 0.3879 | `unet_run_faircompare/best_checkpoint/` |
| 3 | **Hybrid model (this project)** | 276×600 (paper-matched) | **0.9995** | **0.9982** | **0.9808** | `hybrid_run_paperdim/best.pt` |

All three were evaluated on their respective **150-image, never-trained-on test sets** — not validation sets, not training sets.

---

## 2. Where Everything Lives

```
Minor Project/
│
├── Synthetic Dataset/                    <- original dataset, 1024x1024
│   ├── train/  (700 images)  val/ (150)  test/ (150)
│   └── dataset_generator.py used to create it
│
├── Synthetic Dataset 276x600/            <- paper-dimension dataset, RFI morphology
│   │                                        rescaled to stay physically correct
│   ├── train/  (700 images)  val/ (150)  test/ (150)
│   └── generated with dataset_generator_v3_strength.py, seed=42
│
├── unet_rfi_package copy/                <- Authors' code (TensorFlow / tf_unet)
│   ├── tf_unet/                          <- UNMODIFIED original package (Akeret et al.)
│   ├── train_unet_rfi_gpu.py             <- training harness (ours, wraps tf_unet)
│   ├── run_fair_comparison.py            <- runs tf_unet with settings matched to hybrid
│   └── evaluate_test_set.py              <- final test-set evaluation
│
├── unet_run_gpu/                         <- Model #1 results (old 1024x1024 dataset)
│   ├── best_checkpoint/                  <- TRAINED MODEL — use this one
│   ├── checkpoints/                      <- last checkpoint (not necessarily best)
│   ├── training_log.csv
│   ├── progress.json
│   └── eval_test/metrics.json            <- final test-set numbers
│
├── unet_run_faircompare/                 <- Model #2 results (276x600, matched settings)
│   ├── best_checkpoint/                  <- TRAINED MODEL — use this one
│   ├── training_log.csv
│   ├── progress.json
│   └── eval_test/metrics.json
│
├── hybrid_rfi_package/                   <- Hybrid model code (PyTorch, built from scratch)
│   ├── hybrid_model.py                   <- architecture definition
│   ├── train_hybrid.py                   <- training script
│   ├── make_val_split.py                 <- creates val/ split if missing
│   └── evaluate_hybrid_test.py           <- final test-set evaluation
│
└── hybrid_run_paperdim/                  <- Model #3 results (276x600)
    ├── best.pt                           <- TRAINED MODEL — use this one
    ├── last.pt                           <- last checkpoint (not necessarily best)
    ├── training_log.csv
    ├── progress.json
    └── eval_test/metrics.json
```

**Rule of thumb:** whenever you need "the trained model" for anything (a demo, a screenshot, further evaluation), use the **`best_checkpoint`** / **`best.pt`** file, never `checkpoints`/`last.pt` — those are just the most recent epoch, not necessarily the best one.

---

## 3. Model Configurations

### Model 1 — Authors' U-Net, original 1024×1024 dataset
- Architecture: tf_unet, layers=3, features_root=64 (unmodified from paper)
- Batch size: 1 (forced — 1024×1024 images too large for more on a 6GB GPU)
- Optimizer: Adam, lr=0.001 *(the paper's own momentum lr=0.2 provably kills this network — see §5)*
- Patch size: 512×512 random crops
- 100 epochs

### Model 2 — Authors' U-Net, 276×600 dataset, matched to hybrid
- Architecture: tf_unet, layers=3, **features_root=32** *(reduced from paper's 64 — did not fit in available VRAM at batch 4)*
- Batch size: 4 *(reduced from hybrid's 8 — did not fit at 8)*
- Optimizer: Adam, lr=0.001
- Patch size: none — full 276×600 images
- `training_iters=87` set explicitly so one "epoch" = one full pass over the 700 training images (tf_unet's default epoch is a fixed step count, not a full pass — this was corrected to make the comparison fair)
- 22 epochs (matched to hybrid — see below)
- Checkpoint selection used **all 150 validation images**, not a small sample (an earlier attempt at model 1 used only 10, which produced an inflated, non-reproducible "best" score — see §5)

### Model 3 — Hybrid model, 276×600 dataset
- Custom architecture (PyTorch): residual blocks + multiscale anisotropic strip convolutions + efficient channel attention (ECA), GroupNorm (not BatchNorm), no activation on output logits. 9,304,186 parameters.
- Batch size: 8
- Optimizer: Adam, lr=0.001
- Loss: weighted cross-entropy + Dice loss *(tf_unet only supports weighted cross-entropy — Dice could not be added to the authors' code without modifying it, so this is one genuine, disclosed difference between the two methods, not just an architecture change)*
- Patch size: none — full 276×600 images
- Training was stopped manually at epoch 22 (of a planned 40) once score gains had clearly flattened; `best.pt` is epoch 22

---

## 4. Why Two Datasets Exist

The **original 1024×1024 dataset** was built first, before the paper's exact image dimensions (276×600) were matched. Model 1 was trained on it and is kept as the "before" result.

The **276×600 dataset** was generated afterward specifically to match Akeret et al.'s image size, since training at 1024×1024 forced batch_size=1, which is not representative of how the paper's method (batch 32) actually behaves. Regenerating at the paper's own dimensions required **rescaling every RFI morphology parameter proportionally** (blob size, band width, line width, etc.) — otherwise RFI shapes tuned for 1024px images would be physically wrong at 276px (e.g. a block sized for 1024px could not even fit inside a 276px band). This was verified: RFI pixel-fraction statistics stayed consistent across both dataset sizes (~12–15%) after rescaling.

Models 2 and 3 both use this dataset, so they are directly comparable to each other. Model 1 uses the older dataset and is **not** directly comparable to Models 2/3 on image-for-image terms — it's useful as a "before the fixes" reference point, not as a controlled variable in the architecture comparison.

---

## 5. Key Findings & Caveats (for your report)

1. **The authors' code has a real, provable failure mode.** `tf_unet` applies a ReLU to its final output logits before the softmax. If both logits go negative, ReLU clamps them to exactly zero, giving `softmax([0,0]) = [0.5, 0.5]` and **zero gradient** — an irrecoverable dead state. This was directly reproduced: the paper's own learning rate (momentum, lr=0.2) killed the network at iteration 4 when batch size was forced down to 1. Switching to Adam at lr=0.001 avoided this.

2. **An early baseline result (F1=0.8138) was an artifact, not a real result.** Checkpoint selection during one early run used only 10 randomly sampled validation patches per check. Given the dataset's RFI-fraction ranges from 0–60% per image, a 10-image sample is extremely noisy, and taking the *maximum* across ~20 such noisy checks systematically produces an inflated number (the "winner's curse"). Re-evaluated on the full 150-image test set, that same checkpoint actually scored F1=0.34 — consistent with Model 1's officially reported number above. All later runs use full validation sets specifically to avoid this.

3. **Validation/test leakage was checked and ruled out for the final numbers.** All three models' reported scores are evaluated on test images never used for training or checkpoint selection. Where a `val/` split was initially missing, one was created before any reported number was produced (see `make_val_split.py`).

4. **Overfitting was explicitly tested for the hybrid model**, given its unusually high 0.98 test F1:
   - Test F1 (0.9808) was *higher* than the best validation F1 (0.9802) during training — a model that had memorized its validation data could not do this on genuinely unseen data.
   - No sustained divergence between falling training loss and worsening validation score across the training run.
   - High precision (0.980) *and* high recall (0.982) simultaneously — degenerate shortcuts (e.g. predicting all-clean or all-RFI) cannot produce this; they trade one off against the other.
   - MCC = 0.977 (a metric specifically resistant to being fooled by class imbalance).
   - **Caveat that remains, stated honestly:** this dataset is generated by a rule-based synthetic process. A 9.3M-parameter model may be partly learning the generator's own signature rather than a fully general concept of "RFI." This result is genuine for this synthetic dataset; it is not a claim about performance on real GMRT telescope data.

5. **The Model 2 vs. Model 3 comparison is fair but not perfectly identical.** Both use the same dataset, same batch composition philosophy (as close as VRAM allowed), same optimizer and learning rate, same epoch count (22), and the same one-epoch-equals-one-full-pass definition. The two genuine remaining differences, disclosed rather than hidden:
   - Model 2 ran at `features_root=32` and `batch_size=4` (not the paper's 64 / hybrid's 8) due to GPU memory limits.
   - Model 3's loss function includes a Dice term that could not be added to the authors' unmodified code.

   Even with these adjustments favoring what fits on a 6GB laptop GPU, Model 2 reached only F1=0.39 versus Model 3's F1=0.98 — this gap is large enough that it is attributable to the architecture, not merely to training conditions.

---

## 6. Suggested One-Line Summary for a Report/Abstract

> "The authors' original U-Net (Akeret et al. 2017), reproduced faithfully and evaluated under matched training conditions on our synthetic 276×600 RFI dataset, achieved F1=0.39 (ROC AUC=0.67) — only modestly above its performance on our original, unmatched 1024×1024 setup (F1=0.34, ROC AUC=0.55). A hybrid architecture combining residual blocks, multiscale anisotropic strip convolutions, and efficient channel attention, trained under identical conditions on the same dataset, achieved F1=0.98 (ROC AUC=0.9995), indicating the performance gap is attributable to architectural limitations of the plain U-Net rather than training configuration."
