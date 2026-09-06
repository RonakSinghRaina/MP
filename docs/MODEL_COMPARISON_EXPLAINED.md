wThe Three Models, Explained

Written 2026-08-24. Companion to `AUDIT_REPORT.md`. This file answers one
question: **all three models were trained and tested on exactly the same data, so
why did two of them score so differently?**

---

## First, the thing that must be clear

**All three models were trained on the same dataset and tested on the same test
split.** There is no difference in data between them.

|                    | Value                                                              |
| ------------------ | ------------------------------------------------------------------ |
| Dataset            | `Synthetic Dataset 276x600`, generated with `--seed 42`        |
| Train / val / test | 700 / 150 / 150 images                                             |
| Test split         | the same 150 images for all three, 24,840,000 pixels, 15.0808% RFI |

So the score differences are **entirely** down to code and settings. Not data.

---

## The scoreboard

| # | Model                                   | Where it lives            |          ROC AUC |               F1 |
| - | --------------------------------------- | ------------------------- | ---------------: | ---------------: |
| 1 | Authors'`tf_unet`, as first run       | `unet_run_faircompare/` |           0.6681 |           0.3879 |
| 2 | Authors'`tf_unet` + normalisation fix | `unet_run_fixednorm/`   | **0.9908** | **0.9317** |
| 3 | Hybrid (ours)                           | `hybrid_run_paperdim/`  | **0.9995** | **0.9808** |

Models 1 and 2 run **the same architecture, from the same unmodified `tf_unet`
package**. The only differences are in how data was fed to it and how long it
trained. That is the whole story of the 0.39 → 0.93 jump.

> **Metric caveat:** models 1 and 2 report *oracle max-F1* (the best F1 over all
> thresholds, chosen on the test set). Model 3 reports F1 at a threshold fixed on
> validation beforehand. The oracle metric **flatters models 1 and 2**. A properly
> matched comparison would push their numbers slightly down, so the true hybrid
> margin is a little larger than 0.9808 − 0.9317 = 0.049.

---

## Why the first run failed

`tf_unet`'s data loader normalises each image like this:

```python
data = np.clip(np.fabs(data), self.a_min, self.a_max)
data -= np.amin(data)
data /= np.amax(data)
```

The authors' own RFI script supplies **`a_min=30, a_max=210`** — every image gets
clipped into one fixed physical range before scaling.

This project's `RFINpyDataProvider` passes **`a_min=None, a_max=None`**. Those
become `-inf` and `+inf`, so the clip does nothing, and each image is scaled
purely by **its own** minimum and maximum.

**The code was inherited. The two numbers that make the code work were not.**

Why that is fatal here: measured across this dataset, the per-image maximum ranges
from **27 to 222 — an 8.2× swing** — because some generated images contain bright
RFI and others contain almost none. After per-image scaling, identical physical
noise ends up at wildly different values from one image to the next. The network
can never learn "this brightness means RFI", because brightness means something
different in every image.

`tf_unet` is a 2015 design with **no normalisation layers inside it** (no
BatchNorm, no GroupNorm), so it has no way to compensate. It sat frozen at 81%
verification error for 15 straight epochs.

The hybrid model uses **GroupNorm** in every block, which re-standardises the data
inside the network at every layer. It absorbs the inconsistent input scaling
automatically — which is why the *same* normalisation never hurt it.

---

## Every difference between run 1 and run 2

Same architecture (`layers=3`, `features_root=32`), same dataset, same optimiser
(Adam @ 1e-3), same batch size (4), same dropout (0.5). Only these changed:

| # | Change                              | Run 1               | Run 2                         | Did it matter?                                                                |
| - | ----------------------------------- | ------------------- | ----------------------------- | ----------------------------------------------------------------------------- |
| 1 | **Normalisation**             | per-image min-max   | fixed range`[-9.69, 47.40]` | **Decisive.** Verification error fell 81% → 4% in a single epoch       |
| 2 | **Class weights**             | ON,`[0.58, 3.45]` | OFF                           | Probably harmful,**not isolated** — see honesty note                   |
| 3 | **Epochs**                    | 22                  | 60                            | **Significant.** Run 1 stayed frozen for ~15 epochs, so 22 barely began |
| 4 | **Best-checkpoint selection** | last epoch          | best of 12 validation checks  | Minor but correct — run 2's best was epoch 55, not 60                        |

### Honesty note

The final run changed **three things at once**. The evidence that normalisation is
the dominant cause is strong but not airtight:

- Verification error collapsed from 81% to 4% **within one epoch** — far too fast
  to be explained by "more epochs".
- Run 1 was frozen at *exactly* 81.1% for 15 epochs, the signature of a network
  receiving no usable signal, not one training slowly.
- The authors' own recipe (momentum lr=0.2, 100 epochs, no class weights) was
  tested separately on this data **with per-image normalisation** and collapsed to
  a constant output — ROC 0.530, F1 0.292. So more epochs and no class weights
  alone do **not** fix it.

A run isolating class weights was started and lost to a machine timeout. To be
fully rigorous, rerun `baseline_fixednorm.py` with per-image normalisation and
60 epochs; if it still fails, normalisation is proven to be the cause on its own.

---

## Changes made earlier, to get the authors' code running at all

These predate the fix above and are separate from it.

| Change                                                    | Why                                                                                                                                                                | Did it matter?                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| New`.npy` data provider replacing the authors' HDF5 one | Their loader reads Bleien`.h5` files; ours are `.npy`                                                                                                          | **Required.** No effect on quality — it is the documented extension point |
| TensorFlow 1 → 2 compatibility shim                      | TF1 will not install on modern Python                                                                                                                              | **Required.** No effect on the maths                                       |
| Removed`np.fabs()` from normalisation                   | `fabs` turns a strong *negative* noise dip into a maximum-brightness pixel that the mask still calls clean — actively teaching the network wrong associations | **Good change, keep it.** Correct for this data                            |
| Adam @ 1e-3 instead of momentum @ 0.2                     | The paper's learning rate killed training at iteration 4 on this data                                                                                              | **Good change.** Momentum 0.2 was verified to collapse the network here    |
| `features_root` 32 instead of 64                        | 6 GB VRAM, split with Windows                                                                                                                                      | Small negative. The paper's Fig. 3 shows 16 vs 64 is worth ≈0.02 ROC AUC        |
| `batch_size` 4 instead of 1                             | Fits, and reduces gradient noise                                                                                                                                   | Neutral to slightly positive                                                     |
| Added class weights                                       | The dataset is 85/15 imbalanced                                                                                                                                    | Questionable — see note above                                                   |
| Full images instead of 600-wide random slices             | The images are already 276×600                                                                                                                                    | Neutral                                                                          |
| **Dropped `a_min=30, a_max=210`**                 | Not noticed                                                                                                                                                        | **This was the fatal one**                                                 |

**None of the VRAM-driven workarounds caused the failure.** That was checked
directly: the authors' own settings, unmodified, also fail on this dataset.

---

## Why the authors did not hit this

Two reasons, and it is worth being precise about which is verified:

**Verified — they clipped to a fixed range.** Their `scripts/rfi_launcher.py`
passes `a_min=30, a_max=210`. That is in their published code and can be read
today. It makes every image share one scale.

**Inferred — their data was more uniform to begin with.** Their input was a single
continuous 24-hour recording from one telescope on one night, sliced into
600-column windows. The overall brightness of such a stream is broadly stable.
This dataset is 700 **independently generated** images, each with independently
random RFI brightness. This is a reasonable inference from how both datasets are
constructed, but their data was never published, so it cannot be checked directly.

Either way, the fix does not depend on which reason dominates: supplying a fixed
range recovers the model.

---

## A bookkeeping problem worth fixing

`unet_run_gpu/eval_test/metrics.json` contains:

```json
{"roc_auc": 0.6681, "max_f1": 0.3879,
 "checkpoint_dir": "../unet_run_faircompare/best_checkpoint"}
```

Those are the **`unet_run_faircompare` (Model 2) numbers, saved into the
`unet_run_gpu` (Model 1) folder.** It happened because
`evaluate_test_set.py` was run with `--checkpoint_dir` pointing at faircompare but
without `--output_dir`, so the output defaulted to `unet_run_gpu`.

**Consequence:** Model 1's original test metrics (ROC 0.5533 / F1 0.3438, quoted in
`RFI_Project_Model_Comparison.md`) were overwritten and **no longer exist on
disk.** `unet_run_faircompare/` has no `eval_test/` folder at all.

Re-run Model 1's evaluation with an explicit `--output_dir` before relying on that
row in any table.

---

## Two smaller things the run logs reveal

**The first run was still improving when it was stopped.**
`unet_run_faircompare/training_log.csv` ends:

```
18,0.5729,0.5020,0.2927,0,138.9
20,0.5860,0.3241,0.3068,1,139.4
22,0.5996,0.3483,0.3213,1,143.8     <- flagged as best, i.e. still climbing
```

Validation F1 rose at every one of the last three checks. Stopping at 22 epochs to
"match the hybrid" cut it off mid-climb.

**The 1024×1024 run was unstable.** `unet_run_gpu/training_log.csv`:

```
90 : roc 0.8611  f1 0.7776
95 : roc 0.5072  f1 0.6237
100: roc 0.5766  f1 0.2586     <- collapsed at the end
```

It reached F1 0.78 at epoch 90 and then fell apart. Another symptom of a network
with no internal normalisation being fed inconsistently scaled inputs.

---

## Where every trained model lives

| Model                                  | Folder                    | The weights file                 | Metrics                     |
| -------------------------------------- | ------------------------- | -------------------------------- | --------------------------- |
| Authors', original run (1024×1024)    | `unet_run_gpu/`         | `best_checkpoint/model.ckpt.*` | ⚠ overwritten — see above |
| Authors', first 276×600 run           | `unet_run_faircompare/` | `best_checkpoint/model.ckpt.*` | ⚠ none on disk             |
| **Authors' + normalisation fix** | `unet_run_fixednorm/`   | `best_checkpoint/model.ckpt.*` | `eval_test/metrics.json`  |
| **Hybrid**                       | `hybrid_run_paperdim/`  | `best.pt`                      | `eval_test/metrics.json`  |

TensorFlow checkpoints are three files together (`.index`, `.meta`,
`.data-00000-of-00001`) sharing the `model.ckpt` prefix. Keep all three.

Always use `best_checkpoint/` or `best.pt`, never `checkpoints/` or `last.pt` —
the latter are just wherever training happened to stop.

## Re-testing any of them later

```bash
# --- hybrid (torch-env) ---
source ~/torch-env/bin/activate
python3 hybrid_rfi_package/evaluate_hybrid_test.py \
    --dataset_dir "Synthetic Dataset 276x600" \
    --output_dir "hybrid_run_paperdim" \
    --patch_size 0 --per_image_csv --strength_report

# --- authors' + fix (tf-env) ---
source ~/tf-env/bin/activate
python3 experiments/baseline_fixednorm.py \
    --dataset_dir "Synthetic Dataset 276x600" --features_root 32 --epochs 60
#   already at 60 epochs, so it skips training and re-evaluates

# --- authors', original (tf-env) --- NOTE the explicit --output_dir
cd "unet_rfi_package copy"
python3 evaluate_test_set.py \
    --checkpoint_dir "../unet_run_faircompare/best_checkpoint" \
    --output_dir "../unet_run_faircompare" \
    --dataset_dir "../Synthetic Dataset 276x600" \
    --patch_size 0 --features_root 32
```
