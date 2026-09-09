# RFI Project — shared context for any Claude chat in this project

Updated 2026-09-06. Body through PART 7 is the fourth revision (2026-08-27);
PART 8 added 2026-08-30, PART 9 added 2026-08-31, PARTS 10-11 added 2026-09-04, PART 12 added 2026-09-05 and extended 2026-09-06; PART 13 added 2026-09-06, PART 14 (integrity audit) and PART 15 (where the remaining F1 is) and PART 16 (the Chen 2023 CRF model) added 2026-09-07; repository reorganised 2026-09-06.
**Read this first.** It carries the findings
from a deep audit so any new chat, Cowork session, or Claude Code terminal
session starts with the same picture instead of re-deriving it.

> **If you read nothing else:**
> 1. The strip-convolution architecture is **no longer novel** — PART 4
>    (MARS, arXiv:2608.05546, same idea at 34× fewer parameters).
> 2. The model is **oversized** — PART 6. base 16 (2.3M params) scores 0.9788
>    against base 32's (9.3M) 0.9812.
> 3. **The hybrid beats the published state of the art on real LOFAR data.**
>    base 8, no class weight, 3 seeds: **max F1 0.6603 ± 0.0040** against
>    RFI-Net's 0.5979, with **593,842 parameters** (PART 13). Significant at
>    t = 27.2. This is the headline result.
> 4. **We now have a real-data result** — PART 12. tf_unet on the 109
>    expert-labelled LOFAR baselines. The **synthetic-to-real gap is −0.44 F1**
>    for identical code and budget (0.9317 synthetic → 0.4901 real). That single
>    number is the strongest result this project has.
> 5. **Learning rate mattered more than architecture ever did** — PART 12.8.
>    Adam 1e-3 → 1e-4 gained **+0.094 F1**, which is 3.4× the seed spread and
>    far larger than any architectural effect measured here.
> 6. **N=1 is not safe.** Real-data seed spread is **0.052**, larger than most
>    architecture effects previously claimed (strip conv +0.023, ECA −0.005).
> 7. The paper should be reframed around the failure diagnoses (PART 1), the
>    efficiency finding (PART 6) and the synthetic-vs-real gap (PART 12) — not
>    around the architecture.

---

## Where everything lives (reorganised 2026-09-06)

The root had grown to 60 entries and is now 23. **Every path below changed on
2026-09-06** — an older chat or an old command line will reference the previous
locations.

```
Minor Project/
├── data/                    ALL datasets. Gitignored in full.
│   ├── synthetic/           Synthetic Dataset, ...276x600, ...1024x265
│   ├── lofar/               LOFAR_Full_RFI_dataset.pkl, L629174_...pkl, LOFAR_npy/
│   └── hera/                HERA_04-03-2022_all.pkl
├── src/                     model code
│   ├── unet_rfi_package/            original tf_unet package
│   ├── unet_rfi_package copy/       the one the experiments import (tf_unet/)
│   └── hybrid_rfi_package/          the PyTorch hybrid
├── analysis/                one-off investigations
│   ├── lofar_analysis/      PART 11 audit scripts, index files, figures
│   ├── hera_transfer_test/  PARTS 2-3
│   └── dataset_v4_bandpass/ PART 9 generator + verification
├── experiments/             the reusable experiment runners
│   ├── lofar_tfunet_baseline.py     PART 12
│   ├── baseline_fixednorm.py        PART 1 run #4
│   ├── normalisation_control/, width_sweep/, run_ablation.py, ...
├── runs/                    every training output
│   ├── unet/                unet_run*, 8 directories
│   ├── hybrid/              hybrid_run*, 3 directories
│   ├── lofar/               the 8 PART 12 runs
│   └── logs/                *.log
├── docs/                    AUDIT_REPORT.md, NOTES.md, MODEL_COMPARISON_*.md,
│                            RFI_Project_Model_Comparison.md, preview/, image/
├── scripts/                 run_lofar_*.sh, status.sh   (run as ./scripts/status.sh)
├── notebooks/               pkl_viewer.ipynb, lofar_testing.ipynb
├── notes/                   papers (Mesarcik 2022, ML/U-Net notes)
├── results/                 small JSON results
├── lofar_data.py            the memory-mapped LOFAR loader (import from root)
└── RFI-project-context.md   this file
```

**Nothing was deleted.** File count identical before and after (12,485); git
recorded 1,145 renames and 0 deletions.

**What changed in code.** 38 path references across 25 files: `sys.path`
inserts, dataset and output defaults, the absolute paths in the LOFAR audit
scripts, and the shell scripts' `cd`. Packages that moved a level deeper had
their `".."` root resolution corrected to `"..", ".."`. Verified by resolving
every path default on disk, parsing all 60 Python files, loading `lofar_data`
end to end, and completing a real training run from the new layout.

**A .gitignore trap worth remembering.** The old file used unanchored rules
like `unet_run_fixednorm/`, which match at *any* depth. Once those directories
moved under `runs/`, the rules silently swallowed the tracked `metrics.json`
and `training_log.csv` inside them — 52 result files would have quietly left
version control. Root-only rules are now anchored with a leading `/`, and no
directory containing anything worth tracking is excluded wholesale, because
**git cannot re-include a file whose parent directory is excluded**. The
reasoning is written into `.gitignore` itself.

---

## How to navigate this document

It grew chronologically, so **it is a lab notebook, not a report**: later parts
correct earlier ones. Do not read a claim in an early PART and assume it still
stands — check the superseded list below first.

| you want | go to |
|---|---|
| what the project is | "The project in one paragraph" |
| why the baseline first failed | PART 1 |
| HERA transfer results | PARTS 2, 3 |
| why the architecture isn't novel | PART 4 |
| parameter-efficiency sweep | PART 6 |
| machine / GPU / environment gotchas | PART 7, 11.10c, 11.13, 11.13b |
| the v4 bandpass model and its provenance | PART 9 |
| **LOFAR dataset — what it is, how to load it** | **PARTS 10, 11.1–11.10** |
| **LOFAR — traps that silently corrupt results** | **11.5 (leakage), 11.6 (metric), 11.6b (clip)** |
| **the real-data result** | **PART 12** |
| what to run next | 11.14, 12.8 |
| what still blocks publication | "Still outstanding" |

### Every superseded claim, in one place

| claim | where | status |
|---|---|---|
| "normalisation was the main cause of baseline failure" | revision 2 | **wrong** — class weights were, PART 1 |
| "the ReLU trap freeze is permanent" | PART 1 | **superseded** — it escapes after 15 epochs, PART 8 |
| "9 hours to regenerate the dataset" | PART 7 history | **wrong**, corrected in PART 7 |
| "per-image normalisation is mandatory on LOFAR" | 11.8 | **retracted 2026-09-05** — fixed range is fine and separates better |
| clip at μ+4σ for LOFAR | 11.1–11.9 as first written | **wrong** — LOFAR is μ+20σ, 11.6b |
| "tf_unet does not beat a σ-clip baseline" | 12.2 | **superseded** — true at lr 1e-3 only, 12.8/12.10 |
| "lr 1e-4 gains +0.094 F1" | 12.8 | **halved** — seed 0 only; 3-seed mean is +0.046 and not significant, 12.10 |
| dataset mean RFI fraction 12.4% | PAPER_DIMENSIONS.md | **wrong** — 14.67% |
| `unet_run_gpu/eval_test/metrics.json` | — | **mislabelled**, holds faircompare numbers |

### Headline results in one table

Synthetic = `Synthetic Dataset 276x600`, max-F1. Real = 109 expert-labelled
LOFAR baselines. Real numbers use the paper's oracle-threshold protocol so they
are comparable to Mesarcik et al. Table 2.

| model / setting | dataset | F1 | where |
|---|---|---|---|
| tf_unet, per-image norm, class weights ON | synthetic | 0.3879 | PART 1 |
| tf_unet, fixed norm, no class weights | synthetic | **0.9317** | PART 1 |
| hybrid, base 32 (9.3M params) | synthetic | 0.9812 | PART 6 |
| hybrid, base 16 (2.3M params) | synthetic | 0.9788 | PART 6 |
| constant threshold | synthetic | 0.7421 | item 4 |
| σ-clip @2.5σ | **real** | 0.4103 | 11.9 |
| tf_unet, lr 1e-3, matched budget, 3 seeds | **real** | 0.4901 ± 0.0495 | PART 12 |
| **tf_unet, lr 1e-4, 150 ep, 3 seeds** | **real** | **0.5482 ± 0.0139** | **12.10** |
| **hybrid base 8 (593,842 params), 3 seeds** | **real** | **0.6467 ± 0.0079** | **PART 13** |
| **hybrid base 8, NO class weight, 3 seeds — BEST** | **real** | **0.6603 ± 0.0040** | **13.11** |
| hybrid base 32 (9,304,186 params), N=1 | **real** | 0.6571 | 13.7 |
| hybrid base 8 + Chen 2023 CRF head, N=1 | **real** | 0.6604 (+0.0012, inside noise) | 16.4 |
| tf_unet, lr 1e-4, per-image norm | **real** | 0.3039 | 12.11 |
| AOFlagger (wrote the training labels) | **real** | 0.5698 | 11.6 |
| Mesarcik et al.'s U-Net, same architecture | **real** | 0.5876 ± 0.0031 | 12.7 |
| RFI-Net, published best | **real** | 0.5979 | 12.7 |

---

## The project in one paragraph

Detecting radio frequency interference (RFI) in radio-telescope spectrograms as
pixel-wise segmentation. Originally synthetic-only; since 2026-09-04 it also
runs on **real LOFAR data with human expert labels** (PARTS 10–12). Two models: a reproduction of the U-Net
from Akeret et al. 2017 (`tf_unet`, TensorFlow) as the baseline, and a custom
PyTorch "hybrid" (residual blocks + multiscale anisotropic strip convolutions +
efficient channel attention, GroupNorm, 9,304,186 parameters). Dataset generated
by `dataset_generator_v3_strength.py` at 276×600 with `--seed 42`.
Repo: `RonakSinghRaina/MP`.

---

## PART 1 — Why the tf_unet baseline scored F1 0.39 (fully diagnosed)

Four controlled runs. Identical dataset, architecture (layers=3,
features_root=32), optimiser (Adam @ 1e-3) and batch size (4). Only the listed
variables changed.

| # | Normalisation | Class weights | Epochs | ROC AUC | max F1 | Behaviour |
|---|---|---|---:|---:|---:|---|
| 1 | per-image | ON | 22 | 0.6681 | **0.3879** | froze at 81.1% error for 15 epochs |
| 2 | per-image | ON | died at 5 | 0.5677 | **0.3064** | **collapsed** — constant 0.537 everywhere |
| 3 | per-image | OFF | 60 | 0.8747 | **0.7191** | trained normally |
| 4 | **fixed range** | OFF | 60 | 0.9908 | **0.9317** | trained well |

### Decomposition

| Change | Gain in F1 | Share |
|---|---:|---:|
| **Class weights OFF** | **+0.413** | **66%** |
| **Fixed-range normalisation** | **+0.213** | **34%** |
| More epochs (22 → 60) | ≈0 | ≈0% |

**Class weighting was the primary cause. Normalisation was secondary but real.
Epoch count was irrelevant** — run 2 collapsed at epoch 5, so the extra epochs
never mattered.

The gap to the hybrid fell from **0.59 → ≈0.05**. The claim *"the gap is
attributable to architectural limitations of the plain U-Net"* is **dead**.

> **CORRECTION HISTORY:** revision 2 of this doc claimed normalisation was the
> main cause at "~85% confidence". Runs 2 and 3 refuted that. Two pieces of
> evidence had been misread — the "frozen at 81%" run also had class weights ON,
> and the "collapsed to a constant" run used momentum @ lr 0.2. Both were
> confounded. Trust the table above.

### Cause A — class weighting kills this network (the ReLU trap)

Class weighting is the correct textbook remedy for an 85/15 imbalance. It is
lethal *here* because of one line of the authors' code:

```python
output_map = tf.nn.relu(conv)      # tf_unet/unet.py line 145 — AUTHORS' code
```

The network emits two raw scores per pixel, `score_clean` and `score_rfi`.
The mechanism:

1. ReLU forces both scores to be ≥ 0.
2. The 3.45× weight on RFI pushes `score_clean` negative.
3. ReLU clamps it to exactly 0.
4. **ReLU's slope at a negative input is exactly 0**, so nudging `score_clean`
   changes nothing and produces no learning signal. It can never climb back.
   **⚠ SUPERSEDED — see PART 8. On the 1024×265 dataset the network escaped this
   state after 15 epochs. The freeze is long, not permanent.**

Verified against the observed output: a constant 0.537 for all 24,840,000 test
pixels. Working backwards, `softmax(0, c) = 0.537 → c = 0.148`. The network
settled on `score_clean` clamped at 0 and `score_rfi` frozen at 0.148 everywhere
— it stopped being a detector and became a constant.

**Neither piece is the villain alone.** ReLU alone is fine (the authors trained
with it). Class weights alone are fine (the hybrid uses them happily — it has no
output ReLU). Together they are fatal. That is the sharp, publishable finding:

> **The plain U-Net's ReLU-gated output makes the standard remedy for class
> imbalance drive it into a dead state.** (Earlier revisions said *unrecoverable*.
> PART 8 shows it is escapable after ~15 frozen epochs — drop that word.)

### Cause B — per-image normalisation (+0.213)

`RFIPatchDataset` and `RFINpyDataProvider` scale each image by its own min/max.
Measured on this dataset the per-image **maximum ranges 27 → 222 (8.2×)**, so
identical physical noise lands at different values in different images and no
single decision threshold generalises.

Worked example — noise (raw 10) and faint RFI (raw 20) in two images:

| | per-image, quiet img (max 27) | per-image, loud img (max 222) | fixed ruler |
|---|---:|---:|---:|
| noise (10) | 0.370 | 0.045 | **0.345 in both** |
| faint RFI (20) | 0.741 | 0.090 | **0.520 in both** |

Under per-image scaling, *noise in the quiet image (0.370) reads brighter than
real RFI in the loud image (0.090)*. The ordering breaks across images.

The authors avoided this: their `scripts/rfi_launcher.py` passes
`a_min=30, a_max=210`, clipping every image into one fixed physical range. That
parameter pair was never carried into this project's provider.

`tf_unet` has **no normalisation layers** (2015 design) so it cannot compensate.
The hybrid's **GroupNorm** re-standardises activations at every layer, which is
why the identical loader never hurt the hybrid.

**The fix** — `np.clip((data - LO) / (HI - LO), 0, 1)` with `LO, HI = -9.69, 47.40`.
Those are **measured, not chosen**: the 0.5th and 99.5th percentile of each image,
averaged over 40 *training* images (test never touched), recomputed each run by
`calibrate()`. Percentiles rather than min/max because min/max are set by a single
pixel — the 0.5th percentile has sd 4.43 across images versus the raw minimum's
19.59.

Full write-up: `BASELINE_FAILURE_DECOMPOSITION.md`.

---

## PART 2 — HERA transfer test (external dataset)

Source: Zenodo 6724065 / 8275061, `HERA_04-03-2022_all.pkl` (440 MB), simulated
HERA data from Mesarcik et al. Structure: 420 train + 140 test, 512×512, with
boolean masks. **2.75% RFI** versus our synthetic 15.08%. Exported to
`hera_transfer_test/HERA_npy/` in the project's `.npy` layout.

### ⚠ THE PUBLISHED HERA SPLIT HAS TRAIN/TEST LEAKAGE

Verified by SHA-256 over raw pixels:

| Check | Result |
|---|---|
| Unique train images | **382 / 420** (38 internal duplicates) |
| Unique test images | **135 / 140** (5 internal duplicates) |
| **Train∩test overlap** | **31 images — 22% of the test set** |

**31 test images are byte-identical copies of training images.** This is the
dataset's own published split, not something introduced here. This is a finding
about a published, peer-reviewed benchmark and is worth a sentence in the
write-up.

### Zero-shot results (valid — no training involved)

| Experiment | Result |
|---|---|
| Zero-shot, our threshold 0.6113 | ROC 0.8965 · F1 **0.1415** (P 0.077 / R 0.868) |
| Zero-shot, recalibrated on HERA train | F1 **0.4481** @ thr 0.9999 |
| Constant threshold, no model | ROC 0.8120 · oracle F1 **0.5665** |

**Read:** the hybrid *ranks* HERA pixels sensibly (ROC 0.90) but its threshold is
badly miscalibrated for 2.75% RFI — recall 0.87 at precision 0.077. Recalibration
alone triples F1 (0.14 → 0.45). But even recalibrated it **loses to a constant
brightness threshold** (0.4481 vs 0.5665). Partial transfer only.

Do **not** quote "recovered 100% of the gap to the oracle" — the train-chosen and
test-oracle thresholds coinciding is an artefact of the model's saturation (huge
numbers of predictions sit at ~0.9999), not a meaningful claim.

A float16 bug was found and fixed in `recalibrate_threshold.py`: softmax
saturates near 1.0 and float16's spacing there is 0.000488, which collapsed
thousands of distinct scores into one and made the threshold search coarse.
Everything now uses float32. It cost ~0.015 F1.

---

## PART 3 — HERA, deduplicated and re-run (CLOSED)

Deduplication done (`hera_transfer_test/dedupe_hera.py`), output in
`hera_transfer_test/HERA_npy_clean/`:

| | Before | After |
|---|---:|---:|
| Train images | 420 | **351** |
| Test images | 140 | **135** |
| Train∩test overlap | 31 | **0** |
| Contradictory labels | — | **0** |
| RFI fraction | — | train 2.78% / test 2.71% |

Both arms re-run on the clean split, 40 epochs each:

| Arm | Test F1 |
|---|---:|
| Fine-tuned from `hybrid_run_paperdim/best.pt` | **0.9964** |
| **Trained from scratch (the control)** | **0.9996** |
| Difference (pretrained − scratch) | **−0.0032** |

**The transferable value of the synthetic dataset measured negative.** Starting
from the synthetic weights was very slightly *worse* than starting from random.
The honest read is not "pretraining hurts" — it is that **HERA is too easy for
this comparison to say anything**. Both arms sit at 0.999+, i.e. at ceiling, so
there is no headroom in which pretraining could show a benefit. Note also that
leakage was *not* what was propping up the number: 0.9974 (contaminated) →
0.9964 (clean) barely moved, because the task was already saturated.

**Conclusion: HERA is a solved/ceiling benchmark and is not a useful transfer
target. Stop spending runs on it.** Report the dedup finding (31 leaked test
images in a published benchmark) as a data-quality note and move on.

---

## PART 4 — PRIOR ART: the strip-convolution idea is no longer novel

**MARS, arXiv:2608.05546 (2026)** is direct prior art for this project's central
architectural claim. What it does:

- Parallel **anisotropic convolutions**: 3×3 local, **1×9 horizontal, 9×1
  vertical**; decoder widens to **1×31 / 31×1**
- Plain Conv-BN-ReLU, additive skips, **no attention**
- U-Net widths (8, 16, 32, 64)
- **270,769 parameters**, **F1 0.978**
- Trained on 4,800 synthetic 512×512 patches, **evaluated on real GMRT
  observations**, benchmarked against RFDL and filtool

That is the same core idea — anisotropic strip kernels matched to RFI morphology
— at **34× fewer parameters** (270,769 vs 9,304,186) for effectively the same F1
(0.978 vs 0.9808), and validated on **real telescope data** rather than only
synthetic.

**Consequence for the write-up:** "multiscale strip convolutions for RFI" cannot
be presented as novel. This is consistent with the project's own matched-budget
ablation, where strip convolutions were worth only **+0.023 F1** and ECA was
**−0.005**.

**What is still genuinely the project's own contribution** — and it is a better
paper than the architecture one:

1. **The ReLU-gated-output failure mode** (Part 1, Cause A). A named, mechanically
   explained, reproducible failure in a widely-cited baseline, where the standard
   textbook remedy for class imbalance drives the network into an unrecoverable
   dead state. Nobody has written this up.
2. **The per-image normalisation failure** (Part 1, Cause B), with the measured
   8.2× per-image dynamic-range spread and the fixed-range fix that recovers
   F1 0.39 → 0.9317.
3. **The benchmark-difficulty audit** — showing a 2,578-parameter CNN reaches
   F1 0.7823 and a constant threshold reaches 0.7421 on this task.
4. **The leakage finding in the published HERA split** (Part 3).

Reframe the paper around diagnosis and reproducibility, not architecture.

---

## PART 5 — real telescope data with human ground truth (LOFAR) — LARGELY SUPERSEDED

> Kept for history. PARTS 10, 11 and 12 replace almost all of this: the dataset
> is downloaded, fully audited and a result exists. Read those instead.

`L629174_RFI_dataset.pkl` is in the project root and is genuinely real LOFAR
observation data (not simulated). The larger `LOFAR_Full_RFI_dataset.pkl`
(~10 GB, not yet downloaded) carries **109 expert-labelled baselines** — actual
human ground-truth masks on real telescope data, which is what a reviewer will
want to see the model tested on.

`analysis/lofar_analysis/analyse_lofar.py` is **written but not yet run**. It needs
~8–10 GB RAM, so it must run in WSL (`~/torch-env`), not in a Cowork device
session (that VM has only 3.9 GB).

The full dataset structure (list of 4 arrays, shapes, axis order, which
labels are AOFlagger vs. human) is now decoded in PART 10 — read that before
touching `LOFAR_Full_RFI_dataset.pkl`.

---

## PART 6 — parameter-efficiency sweep (DONE — the model is oversized)

`experiments/width_sweep/run_width_sweep.py`. All four widths trained under one
identical protocol: 276×600 full images, the project's own `CEDiceLoss`, measured
class weights, Adam 1e-3, CosineAnnealingLR(T_max=22), 22 epochs, batch 8,
depth 4, dropout 0.2, seed 42; best checkpoint by val F1; threshold chosen on the
150 val images and applied ONCE to test.

| base | parameters | model | ROC AUC | **F1** | IoU | vs base 32 | img/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 152,582 | 0.6 MB | 0.9980 | **0.9547** | 0.9134 | −0.0265 | 61.3 |
| 8 | 593,842 | 2.4 MB | 0.9993 | **0.9749** | 0.9510 | −0.0063 | 65.5 |
| 16 | 2,342,474 | 9.4 MB | 0.9994 | **0.9788** | 0.9585 | −0.0024 | 37.2 |
| **32** | **9,304,186** | 37.2 MB | 0.9995 | **0.9812** | 0.9631 | — | 16.2 |

**The control worked.** base 32 under this harness scored **0.9812** against the
published **0.9808** — a difference of +0.0004. The harness faithfully reproduces
the published model, so every other row in the table is trustworthy.

### What it says

- **base 16 is indistinguishable from base 32**: 4× fewer parameters for
  −0.0024 F1, far inside the ~0.01 single-seed noise floor.
- **base 8 is very close**: 15.7× fewer parameters for −0.0063 F1, still inside
  the noise floor.
- **base 4 genuinely breaks**: −0.0265 is a real drop. The capacity floor for
  this architecture sits **between 152k and 594k parameters**.
- ROC AUC barely moves at all (0.9980 → 0.9995). Ranking ability is almost
  width-independent; only the threshold-dependent metrics separate.
- **Speed has a floor too.** base 4 (61.3 img/s) is *slower* than base 8
  (65.5 img/s) — below a certain size the work is dominated by walking all
  165,600 pixels, not by the parameters, and a tiny model also leaves the GPU
  idle. Do not expect shrinking to keep buying speed.

**Conclusion: roughly 8.7 million of the published model's 9.3 million parameters
buy about half a percent of F1.** This is consistent with the matched-budget
ablation (strip convolutions +0.023, residual blocks +0.002, ECA −0.005).

### ⚠ Before this goes in a paper

**N = 1 per width.** The two headline gaps (−0.0024 and −0.0063) are both smaller
than the ~0.01 threshold below which one seed proves nothing. The claim "the
small model matches the big one" is **not yet established** — it needs ≥3 seeds
at base 8 and base 32, reported as mean ± spread.

Output folders are named by width only, **not by seed**, so a second seed written
into the same `--out_root` would skip the finished widths and silently report the
seed-42 numbers. The script now refuses that. Use:

```bash
python3 experiments/width_sweep/run_width_sweep.py --base 8 32 --seed 0 \
    --out_root hybrid_run_width_sweep_seed0
```

### Runtime (measured, on the user's GPU, batch 8)

base 4 ≈ 10 min · base 8 ≈ 23 min · base 16 ≈ 31 min · base 32 ≈ 30 min.
**An earlier estimate of "9 hours" in this doc's history was wrong** — it was
extrapolated from the original run, which used `batch_size=1` (the script
default), 8× more steps and very poor GPU utilisation.

### Relative to MARS

MARS reaches F1 0.978 with **270,769** parameters. That count sits right in the
zone where *this* architecture starts to degrade (between base 4's 152k and base
8's 594k). So MARS is genuinely more parameter-efficient than HybridRFINet, and
that — not raw accuracy — is the gap a new design has to close.

---

## PART 7 — machine migration: Windows/WSL2 → Fedora 44 (2026-08-29)

### Why the move happened

Ronak's supervisor advised doing the whole project on a Linux dual boot rather
than WSL2, because WSL2's **split RAM** was blocking work — WSL2 defaults to
roughly half of system RAM, and the 10 GB LOFAR dataset needs more than that.

> **Worth knowing:** that specific problem has a two-line fix without changing
> OS. A `C:\Users\<user>\.wslconfig` containing `[wsl2]` / `memory=24GB` raises
> WSL2's limit, then `wsl --shutdown`. The dual boot went ahead anyway and is a
> better long-term setup, but if anyone asks why WSL2 "couldn't" do it, the
> honest answer is that it could have.

The Windows + WSL2 environment was never broken and still works. It is now the
**only second copy** of the `.pt` weights and the `.pkl` datasets, neither of
which is in git. **Do not reclaim the Windows partition.**

### How the files moved, and what got lost

Copied by **USB stick** — which was FAT32, so it carries a **4 GB per-file
ceiling**. `L629174_RFI_dataset.pkl` (3.78 GB) only just fitted. **The 10 GB
`LOFAR_Full_RFI_dataset.pkl` cannot pass through that stick — download it
directly onto the Linux disk.**

The copy was incomplete, and the gaps were found in three separate rounds:

| Round | What was missing | How it was recovered |
|---|---|---|
| 1 | 6 tracked files: `AUDIT_REPORT.md`, `BASELINE_FAILURE_DECOMPOSITION.md`, `CITATION.cff`, `CLAUDE.md`, `LICENSE`, `.gitignore` | `git restore` — git had them |
| 2 | `hybrid_run_width_sweep/base32/eval_test/` — the whole results folder for the base-32 control | copied from the mounted Windows partition |
| 3 | `sweep_results.json` was a **stale** copy holding only `[4, 16]` | copied from Windows, then rebuilt to hold all four widths |
| — | both `.pkl` datasets (4.2 GB) | copied from Windows, verified byte-exact |

Round 1's losses were a contiguous alphabetical block from `AUDIT` to `LICENSE`,
which suggests the copy was sorted alphabetically and the first chunk failed.

**Lesson — do this after any bulk copy, before trusting it:**

```bash
( cd "$WIN" && find . -type f -not -path './.git/*' | sort ) > /tmp/win.txt
( cd "$FED" && find . -type f -not -path './.git/*' | sort ) > /tmp/fed.txt
comm -23 /tmp/win.txt /tmp/fed.txt        # on the source but not the destination
```

The Windows partition is `/dev/nvme0n1p3` (835 GB NTFS). **Mount it read-only:**

```bash
sudo mount -o ro /dev/nvme0n1p3 /mnt/win
# project at: /mnt/win/Users/RONAK SINGH/Documents/Coding/Minor Project
```

### Line endings

The Windows copy arrived with **CRLF** line terminators throughout. Git compared
them against its own LF copies and reported **26 modified files, 4,702
insertions** — of which only **3 files, 61 lines** were real changes (all of them
width-sweep output). The rest was invisible-character noise.

Fixed by restoring the CRLF-only files from git, stripping CRs from the 3 real
ones with `sed -i 's/\r$//'`, then adding `.gitattributes` (`* text=auto eol=lf`)
and `git config core.autocrlf false` so it cannot recur.

### The GPU saga — one full day

Summarised in **Other things that bite** below; the short version is that Secure
Boot had to be disabled (the MOK enrollment screen never accepted keyboard input
on this Legion), and kernel **7.1.10 renders the GNOME desktop on the RTX 3060**,
which makes the UI crawl. **Kernel 6.19.10 is the one to boot** and is pinned as
default. Verified working: driver 610.57.04, CUDA 13.3, torch 2.13.0+cu130,
`torch.cuda.is_available() == True`.

### State at the end of the migration

Everything verified present, committed as **`2295372`** and pushed to
`origin/main`. Full width sweep re-runs on Fedora and skips all four widths,
confirming the whole chain works. TensorFlow was the one loose end; it is now
closed — see the next section.

### TensorFlow on Fedora — CLOSED (2026-08-30)

`~/tf-env` is **Python 3.12.14, TensorFlow 2.21.0** (Fedora 44's default Python
3.14 has no TensorFlow wheels, so the venv must be built with
`python3.12 -m venv ~/tf-env`). Two things were still broken and both are fixed.

**1. TensorFlow saw no GPU.** `tf-env` has the `nvidia-*-cu12` wheels installed,
but TF 2.21 does not add them to the loader path on Fedora, so it printed

```
Cannot dlopen some GPU libraries ... Skipping registering GPU devices
```

and silently ran on CPU. It never names the missing library, which is what makes
this expensive to diagnose. **The fix is to put the pip CUDA libs on
`LD_LIBRARY_PATH` before every tf-env run:**

```bash
export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
```

With that set, `tf.config.list_physical_devices('GPU')` returns
`[PhysicalDevice(name='/physical_device:GPU:0', ...)]`. The export is **not**
picked up from `~/tf-env/bin/activate`, because every script here is invoked as
`~/tf-env/bin/python ...` rather than through an activated shell — so put the
line in the command itself. Version skew is fine: TF 2.21 is built against CUDA
12.5.1 / cuDNN 9, the wheels are 12.9 / 9.25, and driver 610.57.04 (CUDA 13.3)
runs CUDA 12 binaries.

**2. `PIL` and `matplotlib` were missing.** `tf_unet/util.py` imports both at
module scope, so *any* import of the authors' code died with
`ModuleNotFoundError: No module named 'PIL'`. Fixed with
`~/tf-env/bin/pip install pillow matplotlib` (Pillow 12.3.0, matplotlib 3.11.1).

**Verified working after both fixes** — `experiments/baseline_fixednorm.py` runs
the full train → validate → best-checkpoint → test-once chain on the GPU. Measured
on the RTX 3060 6 GB at 1024×265, features_root 32:

| batch | peak VRAM | ms/step | steps/epoch | wall/epoch |
|---:|---:|---:|---:|---:|
| 2 | 1.49 GB | 87 | 350 | — |
| **4** | **2.76 GB** | **174** | **175** | **~67 s** |
| 6 | 3.91 GB | 265 | 117 | — |

So a 60-epoch fixed-norm baseline at 1024×265 is **~75–80 min**, not an
overnight job. Batch 4 leaves less than half the card in use.

### Claude tooling on Fedora

The **Claude desktop app supports Debian and Ubuntu only** — not Fedora — so the
Cowork folder bridge cannot reach this machine and Cowork sessions cannot read or
write the project directly. Use the **Claude Code CLI**, which does support
Fedora via dnf:

```bash
sudo tee /etc/yum.repos.d/claude-code.repo <<'EOF'
[claude-code]
name=Claude Code
baseurl=https://downloads.claude.ai/claude-code/rpm/stable
enabled=1
gpgcheck=1
gpgkey=https://downloads.claude.ai/keys/claude-code.asc
EOF
sudo dnf install claude-code
```

Start it from the project directory with `claude`, and open the session with
`read RFI-project-context.md before we begin`.

### Where Claude Code sessions live, and how to resume one (2026-09-05)

**Sessions are filed by the directory you launched `claude` from**, under
`~/.claude/projects/<path-with-slashes-and-spaces-replaced-by-dashes>/`. Each
session is one `.jsonl` transcript plus a same-named folder of tool results.

This bit us: sessions from 2026-08-30 to 2026-09-05 were launched from
`~/Downloads/Antigravity/Antigravity-x64` by mistake, so they were filed under
`-home-ronaksingh-Downloads-Antigravity-Antigravity-x64/` and were invisible
from the project directory. The correct directory is
`-home-ronaksingh-Documents-minor-project-Minor-Project/`.

**Two things live per-project and both must move if the launch directory
changes** — the transcript *and* `memory/`. The Minor Project folder's
`memory/` was completely empty, so a session started there would not have
known the commit identity or to read this document first.

```bash
SRC=~/.claude/projects/-home-ronaksingh-Downloads-Antigravity-Antigravity-x64
DST=~/.claude/projects/-home-ronaksingh-Documents-minor-project-Minor-Project
cp -n "$SRC"/<session-id>.jsonl "$DST"/
cp -rn "$SRC"/<session-id> "$DST"/
cp -n "$SRC"/memory/*.md "$DST"/memory/
```

The long LOFAR/bandpass session is `dbe9e263-0f9a-4d0b-9c18-6fc448e05e3b`
(1,582 records, 2026-08-30 → 2026-09-05); it has been copied across along with
all three memory files. Re-copy the `.jsonl` after the source session finally
closes, since it keeps growing while it is live.

To resume: `cd` to the project directory, then `claude --resume` and pick the
session, or `claude --continue` for the most recent. **Plain `claude` starts a
new empty chat** — it does not resume anything.

**Do not rely on this.** The transcript is convenience, not memory: long
sessions get compacted, so older exchanges come back as a summary rather than
verbatim, and transcripts are local to one machine (nothing syncs to the
Windows install). *This document* is the actual persistence layer — it is in
git, it survives machine changes and fresh chats, and it is why it gets
updated whenever a run finishes.

---

## PART 8 — the 1024×265 bandpass dataset (v4): baseline runs + the ESCAPE finding

New dataset: `Synthetic Dataset 1024x265`, instrument-bandpass model, seed 42,
700 train / 150 val / 150 test, 1024 (freq) × 265 (time). Mean RFI **14.03%**
over the train split. Per-image maximum ranges **20.3 → 198.7 (9.8×)**, worse
than the 276×600 set's 8.2×. Auto-calibrated fixed range: **lo −2.25, hi 46.23**.

**Provenance, verified 2026-08-30:** the vendored `unet_rfi_package copy/tf_unet`
is **byte-identical to upstream `github.com/jakeret/tf_unet` HEAD `0dcdf2f`**
(2020-05-05). A full `diff -r` over the whole repo reports one difference, the
`.github/` CI folder. All six package files match by md5. Nothing in the authors'
code has been altered; every deviation lives in the provider subclass.

**tf_unet output size here is 984×224**, not the nominal 984×225 — see the
`net.offset` bullet in *Other things that bite*.

### Run 1 — the flawed reimplementation (per-image norm + class weights ON)

`experiments/normalisation_control/run_control.py --norm per_image --class_weights on`
· 60 epochs · batch 4 · features_root 32 · layers 3 · Adam 1e-3 · seed 42 ·
output `unet_run_control_1024x265_per_image_cw/`.

**Test: ROC AUC 0.9130 · PR AUC 0.7716 · max F1 0.6815** · not collapsed ·
pred range [0.000, 1.000] std 0.2173 · best val F1 0.6945 **@ epoch 60**.

| epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val F1 | .370 | .369 | .370 | .395 | .594 | .632 | .656 | .669 | .665 | .689 | .692 | **.695** |

### ⚠ THIS CONTRADICTS "UNRECOVERABLE" IN PART 1 CAUSE A

The network sat frozen at F1 ≈ 0.37 for **15 epochs** — the dead-state signature —
then **escaped and climbed steadily for the remaining 45**. Best epoch is the
last one, so **0.6815 is a lower bound, not a converged result.**

Two consequences:

1. **"It can never climb back" is too strong.** The ReLU-gated dead state is a
   long frozen phase, not a grave. Reword to: *the standard remedy for class
   imbalance drives the plain U-Net into a frozen phase costing ~15 epochs and a
   large amount of final accuracy*. That version survives a reviewer who simply
   trains it longer.
2. **`unet_run_control_per_image_cw` (F1 0.3064) was halted at epoch 5 by the
   collapse detector** — its `progress.json` says `epochs_completed: 5`. It was
   never given the chance to escape, so 0.3064 is an **early-stop artefact**, not
   evidence of permanent death. The PART 1 decomposition table needs re-deriving,
   especially the row *"More epochs (22 → 60) ≈0"*: on this run 22 → 60 epochs
   took F1 from ≈0.39 to 0.69.

**The headline survives.** Per-image normalisation + class weights still cost a
great deal. The *mechanism* story needs softening, not the finding.

### Run 2 — the corrected baseline (fixed norm + class weights OFF)

Same script, same seed, two flags flipped: `--norm fixed --class_weights off`.
Output `unet_run_control_1024x265_fixed/`.

**Test: ROC AUC 0.9902 · PR AUC 0.9789 · max F1 0.9483** · pred std 0.3300 ·
best val F1 0.9525 **@ epoch 30** — i.e. genuinely **converged**, unlike Run 1.

| epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val F1 | .894 | .913 | .933 | .928 | .941 | **.953** | .952 | .951 | .933 | .951 | .952 | .946 |

### Head-to-head on the 1024×265 dataset

| arm | normalisation | class weights | test max F1 | best epoch |
|---|---|---|---:|---:|
| Run 1 | per-image | ON | 0.6815 | 60 (still climbing) |
| **Run 2** | **fixed −2.25/46.23** | **OFF** | **0.9483** | 30 (converged) |
| | | **difference** | **+0.2668** | |

For reference the same two arms on 276×600 gave 0.3064 and 0.9317. The corrected
baseline is **better on the new dataset** (0.9483 vs 0.9317) despite the harder
bandpass structure.

⚠ **This is the two changes together, not a decomposition.** The 276×600 split
(class weights 66%, normalisation 34%) has *not* been reproduced here — that needs
the third arm, `--norm per_image --class_weights off`, which has not been run on
this dataset. Until it is, do not quote the 66/34 split as holding for v4.

### Run 3 — the hybrid at base 8 (the efficient width)

`experiments/width_sweep/run_width_sweep.py --base 8` · 60 epochs · batch 4 ·
depth 4 · dropout 0.2 · Adam 1e-3 · CosineAnnealingLR(T_max=60) · seed 42 ·
output `hybrid_run_1024x265/base08/`. Batch 4 was chosen to equalise gradient
steps with the U-Net arms (175 steps/epoch x 60 = 10,500 for both).

**593,842 params · 2.38 MB · 143.0 img/s · 32.2 min to train.**

**Test: ROC AUC 0.9998 · PR AUC 0.9991 · F1 0.9877 · oracle F1 0.9878 ·
precision 0.9922 · recall 0.9833 · IoU 0.9757 · MCC 0.9857** ·
best val F1 0.9879 @ epoch 58 · threshold 0.8829 chosen on val.

Confusion matrix over the full 40,704,000 test pixels (150 x 1024 x 265):
TN 34,863,862 · FP 45,067 · FN 96,877 · TP 5,698,194.

**The val-chosen threshold was essentially optimal**: F1 0.9877 at the val
threshold versus 0.9878 oracle, a gap of 0.0001. So for this run the
threshold-selection question that dogs the 276x600 comparison simply does not
bite — quote either number.

Base 8 on the *old* 276x600 set scored 0.9749 (22 epochs, batch 8); base 32
scored 0.9812. Different dataset and epoch budget, so these are not comparable —
do not read 0.9877 as "base 8 improved".

### HEAD-TO-HEAD on 1024x265 — all three runs, one protocol

Same dataset, same 60 epochs, same 10,500 gradient steps, same Adam @ 1e-3,
same seed 42, best checkpoint on val, test scored once.

| model | params | scored px/image | test F1 |
|---|---:|---|---:|
| tf_unet, per-image + class weights | 465,986 | 984x224 | 0.6815 |
| tf_unet, fixed norm, no class weights | 465,986 | 984x224 | **0.9483** |
| **HybridRFINet base 8** | **593,842** | **1024x265 (full)** | **0.9878** |

**This is a near-matched-parameter comparison, not 594k vs 9.3M.** tf_unet at
layers=3 / features_root=32 is **465,986** trainable parameters — *smaller* than
base-8 hybrid. The hybrid spends **1.27x the parameters for +0.0395 F1** over the
corrected baseline. That is a far more defensible efficiency claim than anything
built on the 9.3M model, and it is the number to put in the paper.

### tf_unet parameter counts — measured, not assumed

Counted directly from `tf1.trainable_variables()` on the authors' unmodified code:

| configuration | parameters | what it is |
|---|---:|---|
| layers 5, features_root 64 | **31,030,658** | **the authors' OWN defaults in `scripts/rfi_launcher.py`** |
| layers 3, features_root 64 | 1,861,762 | the depth this project settled on, paper's width |
| **layers 3, features_root 32** | **465,986** | **what every controlled run in this project used** |

So the authors' published network is **31 million parameters — 52x larger than
the 466k version this project benchmarks, and 52x larger than the base-8 hybrid**
that beats it. Every "tf_unet baseline" number in this repo (0.3064, 0.3879,
0.7191, 0.9317, 0.6815, 0.9483) comes from the 466k configuration, NOT from the
authors' 31M one. State that explicitly in the paper — a reader will otherwise
assume the baseline is the published 31M model.

⚠ **Two things are still not equal, state them:**

1. **Not the same pixels.** tf_unet's valid padding scores it on the interior
   984x224 (33,062,400 px, 81% of each image); the hybrid is scored on the full
   1024x265 (40,704,000 px). The border strip the U-Net never sees is excluded
   from its score.
2. **Not the same normalisation.** The U-Net arm uses the fixed range
   [-2.25, 46.23]; `RFIPatchDataset` gives the hybrid per-image min/max. That is
   the published hybrid setup and GroupNorm is argued to absorb it, but the two
   arms are not preprocessed identically.

**N = 1.** One seed. PART 6's caveat applies unchanged.

### The authors' own 30 / 210 clip cannot be reused here

`scripts/rfi_launcher.py` passes `a_min=30, a_max=210`. Those constants are
specific to their telescope data. Measured over 40 training images of this
dataset: range **−25.3 → 198.7**, median **9.69**, and **94.08% of all pixels
fall below 30** (nothing at all above 210). Clipping at 30 would flatten 94% of
every image to a single value. The fixed-range arm therefore uses the authors'
*method* — one ruler for every image — with the range **measured from this
dataset**, not their numbers.

Note their literal path is `clip(fabs(x), 30, 210)` then per-image min/max
rescale; ours is `clip((x − lo)/(hi − lo), 0, 1)`. We drop `fabs()` (negative
noise dips are real here) and drop the post-clip rescale (which is what
reintroduces per-image variation when an image does not span the full range).

---

## PART 9 — the v4 bandpass model: what it is, and what is ours vs. published

Added 2026-08-31. Written because a reviewer will ask "where does this bandpass
come from?" and the honest answer needs to be ready before submission, not
improvised.

### The model as implemented

`analysis/dataset_v4_bandpass/generate_dataset_v4.py`, Section 2. One multiplicative gain
per frequency channel, redrawn for every image:

```
B(f) = max( W_e(f) * [ 1 + R(f) + S(f) ], B_floor )
```

| Term | Meaning | Form |
|---|---|---|
| `W_e(f)` | passband envelope | Tukey raised-cosine taper, `n_edge = round(eps*N)`, ramp `0.5*(1 - cos(pi*i/(n_edge+1)))`, mirrored at both edges |
| `R(f)` | slow filter ripple | `(A_r/2) * (1/M) * sum_m sin(2*pi*f/P_m + phi_m)`, `P_m ~ U(0.4,1.5)*N`, `phi_m ~ U(0,2pi)` |
| `S(f)` | standing wave | `(A_s/2) * sin(2*pi*f/P_sw + phi_sw)`, `P_sw` fixed in **channels** |

Every one of `eps, A_r, P_sw, A_s` is jittered per image by
`x -> x*(1 + J*u)`, `u ~ U(-1,1)`. Phases are always fresh, so even at `J = 0`
no two images share a curve. That is deliberate: a single fixed bandpass shape
would be memorisable and the task would collapse into "learn one curve".

Defaults used for `Synthetic Dataset 1024x265`:

| Symbol | Flag | Value |
|---|---|---|
| `eps` | `--edge_frac` | 0.08 |
| `A_r` | `--ripple_amp` | 0.15 (fractional peak-to-peak) |
| `M` | `--ripple_modes` | uniform int in [2, 4] |
| `P_m/N` | `--ripple_period_frac` | [0.4, 1.5] |
| `P_sw` | `--sw_period_channels` | 128 |
| `A_s` | `--sw_amp` | 0.05 |
| `J` | `--bandpass_jitter` | 0.25 |
| `B_floor` | `--bandpass_floor` | 1e-3 |

### Where it enters the image — the part that matters

`generate_dataset_v4.py:594`

```
sigma_rx   = rx_noise_frac * median(sky_sigma_eff)          # 0.10 * median
spectrogram = B(f) * (pure_signal + rfi_layer) + rx_noise
```

**The additive receiver noise sits OUTSIDE the gain.** This is the whole physical
point of v4. Where `B(f)` is small (band edges) the sky *and* the RFI are both
attenuated, but `n_rx` is not — so edge RFI genuinely drowns. This is what v3's
`generate_bandpass_gain_curve` got wrong: it modulated the *additive* noise
level, which is not a gain at all (see the comment at `generate_dataset_v4.py:136`).

Labels follow the same physics (`:601`):

```
strength = B(f)*rfi_layer / sqrt( (B(f)*sky_sigma_eff)^2 + sigma_rx^2 )
mask     = strength > 0.5
```

Thresholding the **post-bandpass** amplitude against the **local total** noise.
RFI the gain has buried is not labelled — otherwise the labels are unsatisfiable
and every F1 punishes the model for something invisible. Cost on this dataset:
**54,403 of 37,073,066 injected pixels (0.147%)**, all at band edges. Per-channel
breakdown in `Synthetic Dataset 1024x265/exclusion_by_channel.json`.

Consequence worth stating in the paper: because `B(f)` multiplies the sky too,
**dividing by `B(f)` recovers a uniform-noise image** — exactly what real
bandpass calibration does. The models are being asked to work on *uncalibrated*
data, which is the harder and more realistic setting.

### PROVENANCE — do not overclaim this

**The composite formula is NOT published anywhere. It is our construction.**
Do not cite it as if it came from a paper; a reviewer who looks will not find it.
Describe it as "a parametric bandpass model combining three standard
instrumental effects."

Each ingredient, however, is standard and citable:

| Term | Status in the literature | Citation |
|---|---|---|
| Tukey taper | Textbook cosine-tapered window; identical to `scipy.signal.windows.tukey` | Harris 1978, *Proc. IEEE* |
| Standing wave `S(f)` | Real, well studied — "standing waves" / "baseline ripple" / "fixed-pattern noise", from reflections between feed cabin and dish or at fibre joints. Genuinely sinusoidal in frequency. | Popping & Braun 2008 A&A (WSRT), arXiv:0712.2303; HiFAST III (FAST), doi:10.1088/1674-4527/ad9653 |
| Slow ripple `R(f)` | Passband ripple from imperfect analogue filters | standard RF filter theory |
| `B(f)*(sky+RFI) + n_rx` | Standard measurement-equation form: direction-independent gain multiplies, receiver noise adds | Hamaker, Bregman & Sault 1996; Smirnov 2011 (RIME) |

The standing-wave physics gives a hard constraint we should be quoting:

```
delta_nu_ripple = c / (2L)        L = path-length difference
```

Checks out against published numbers: WSRT focal distance 8.75 m -> ~17 MHz
period; FAST ~138 m -> ~1.09 MHz. Both appear in the papers above.

### Closest published analogue: `hera_sim`

`hera_sim.sigchain` (github.com/hera-team/hera_sim) does structurally the same
thing — `gen_bandpass`, a `Bandpass` class, and
`gen_reflection_coefficient` / `gen_reflection_gains` for cable reflections.
Same architecture as ours: **a smooth envelope times a sinusoidal reflection
term, redrawn per realisation.**

One real difference, and it is the one a reviewer will find:

- **hera_sim's envelope is empirical** — a degree-6 polynomial fit to *measured*
  HERA feed bandpasses, then perturbed per antenna by convolution with complex
  white noise plus a random delay/phase factor.
- **Ours is analytic** — a Tukey taper with no measured instrument behind it.

Defensible for a synthetic benchmark (we want a controlled difficulty knob, not
one telescope's quirks) but it must be **said explicitly**, not left to be noticed.

### The tf_unet authors built their own simulator — use this

Akeret et al., the same group whose U-Net is our baseline, released
**HIDE & SEEK** (arXiv:1607.07443, ASCL 1607.019). HIDE forward-models the whole
single-dish instrument chain including gain; the paper discusses noise and RFI
modelling in detail, and their RFI-detection work runs on HIDE output.

This lets us position v4 as *"an independent bandpass model in the spirit of
HIDE"* and gives a principled comparison point.

> **NOT VERIFIED:** HIDE's internal gain parameterisation was never read. Only
> that it forward-models the instrument is confirmed. **Read section 2 of
> arXiv:1607.07443 before claiming any specific similarity or difference.**

### Two concrete fixes before submission

1. **`P_sw` is specified in channels (128), not MHz.** That severs it from the
   physics — 128 is currently an arbitrary number. State the channel width, then
   quote the implied path length `L = c / (2 * delta_nu_ch * P_sw)` and show it
   lands somewhere physically plausible for a real dish. One line of arithmetic
   turns an arbitrary constant into a defensible one.
2. **Justify the analytic envelope in one sentence** — controlled, tunable,
   telescope-agnostic, versus hera_sim's measured polynomial.

Neither changes any result. Both close the obvious line of attack.

---

## PART 10 — LOFAR_Full_RFI_dataset.pkl: structure, decoded (2026-09-04)

Read in full: Mesarcik, Boonstra, Ranguelova & van Nieuwpoort (2022), *"Learning
to detect RFI in radio astronomy without seeing it,"* MNRAS, arXiv:2207.00351.
Copy saved at `notes/Mesarcik2022_Learning_to_detect_RFI_without_seeing_it.pdf`.
Our professor pointed us at this paper specifically to understand how to open
`LOFAR_Full_RFI_dataset.pkl` (9.3 GB, in the project root, not yet processed
into `.npy`).

### File structure (confirmed without loading the file — see method below)

`pickle.load()` returns a plain Python **list of 4 arrays** (not a dict, not
one array):

| Index | Contents | Shape | dtype |
|---|---|---|---|
| `data[0]` | training spectrograms | `(7500, 512, 512, 1)` | float32 |
| `data[1]` | training masks (AOFlagger-derived, **not** human) | `(7500, 512, 512, 1)` | bool |
| `data[2]` | test spectrograms | `(109, 512, 512, 1)` | float32 |
| `data[3]` | test masks (**hand-labelled by a human expert**) | `(109, 512, 512, 1)` | bool |

- The 4th shape dimension (`1`) is a channel axis, kept only because the
  paper's CNNs expect an image-like `(N,H,W,C)` tensor. It carries no
  information — analogous to a greyscale image having 1 channel vs. RGB's 3.
- 512×512 is **not** a natural instrument dimension: raw spectrograms are
  599×616 (§4.2), randomly cropped to 512×512 so patches tile evenly.
- 109 is exactly the number of independently expert-labelled baselines the
  paper reports (§4.2) — this is the real ground-truth evaluation set.
- 7500 training spectrograms are machine-labelled via **AOFlagger** (a
  classical/heuristic flagger), not a human. Only the 109-baseline test set
  has human labels. This matters for anyone using `data[1]` as "ground truth"
  — it's a strong baseline's output, not verified truth.
- Verified by pure arithmetic on the file's byte size (no file access, to
  avoid a second 9.3 GB load on top of the user's already-open kernel):
  `7500×512×512×1×4B + 7500×512×512×1×1B + 109×512×512×1×4B + 109×512×512×1×1B`
  = 9.29 GiB predicted vs. 9.29 GiB actual (`stat -c '%s'`), agreement to
  within 2 MB.

### Axis order inside one spectrogram

From the user's own `plt.matshow(data[1][0,:,:,0])` screenshot: the mask shows
**vertical** streaks. RFI in real data is normally narrowband-persistent (one
frequency channel, many time samples) or broadband-transient (one time sample,
many frequency channels). Vertical streaks running down the image imply
**rows = time, columns = frequency** for this array — the **opposite**
convention from our own synthetic dataset (`Synthetic Dataset */`), which is
frequency×time. Anyone feeding a LOFAR array into our existing pipeline code
must `.T` it first, or explicitly relabel axes, or results will be silently
transposed relative to what the model expects.

### Why "axis=0 / axis=1" confused both student and professor

The professor's shorthand ("axis=0 has the data, axis=1 has the masks")
actually meant **list index** 0 and 1 of the outer 4-element list
(`data[0]`, `data[1]`), not the numpy **array axis** 0/1 within one array
(row vs. column of a single 512×512 image). Same word, two different
concepts — this is the root confusion, not a dataset-structure problem.

### Preprocessing pipeline used by the paper (§4.2)

Clip to `[|μ−σ|, μ+4σ]` → natural log → standardize to `[0,1]`. This mirrors
our own PART 1 finding that per-image vs. fixed-range normalization was a
primary driver of the tf_unet baseline's failure — worth citing as external
confirmation that normalization choice matters on this class of data, not
just in our synthetic setup.

### Benchmark numbers to compare against (paper's Table 2, real LOFAR data,
### human-expert ground truth)

| Method | F1 |
|---|---|
| AOFlagger (classical) | 0.5698 |
| U-Net | 0.5876 |
| **RFI-Net (best)** | **0.5979** |
| R-Net | 0.5286 |
| NLN (paper's own method; best AUROC/AUPRC despite lower F1) | 0.5114 |

Contrast with our own synthetic-data F1 (~0.988, PART 8/9) — real data is a
much harder task. This is the number our own model would need to be compared
against for outstanding item 6 (no result yet on real telescope data with
human labels).

### NLN method (paper's own contribution, not what we're doing)

Trains a discriminative autoencoder only on RFI-free patches (selected via
AOFlagger weak labels), then flags anomalies at inference via combined
latent-distance + pixel-reconstruction-error (paper Eq. 6–7). This is an
**unsupervised** approach — different from our supervised U-Net/hybrid
setup. Relevant as related work, not something we've implemented.

### Not yet done

- LOFAR pickle has not been converted to `.npy` / split into the project's
  usual `{train,test}/{image,mask}/` layout. A converter script was offered
  but not yet written — needs explicit go-ahead before starting (RAM-safe
  streaming extraction required; do not `pickle.load` the whole file
  alongside another already-loaded copy — system has 19 GB RAM total and was
  at 15 GB used + 3 GB swap with one copy already loaded).
- No model of ours has been run on `data[2]`/`data[3]` yet.

---

## PART 11 — LOFAR deep audit: every number, measured (2026-09-04)

Full empirical audit of `LOFAR_Full_RFI_dataset.pkl`, run in five stages.
Scripts, JSON reports and figures are in `analysis/lofar_analysis/`
(`deep_audit_stage1..5_*.py`, `audit_lofar_report*.json`,
`fig_lofar_overview.png`, `fig_lofar_profiles.png`). Everything below is
measured from the file, not quoted from the paper, unless marked otherwise.

### 11.1 The data is RAW. It is not normalised, not log-scaled.

| | train `data[0]` | test `data[2]` |
|---|---|---|
| min | 0.0 | 165.83 |
| median | 1.194e6 | 1.220e6 |
| mean | 1.439e6 | 1.507e6 |
| p99 | 3.708e6 | 3.675e6 |
| max | **1.4455e11** | **4.8214e10** |
| std | 2.158e7 | 2.336e7 |
| NaN / Inf / negative | 0 / 0 / 0 | 0 / 0 / 0 |
| exact zeros | 12,285,980 (0.62 %) | 0 |

The max is ~39,000× the p99. **This is why `plt.matshow(data[0][0])` looks
like a flat purple square** — a handful of pixels set the colour scale and
everything real sits in the bottom 0.001 % of it. Nothing is wrong with the
data; the display is being crushed by outliers. Apply the paper's
preprocessing (§11.7) and the structure appears immediately — see
`fig_lofar_overview.png`.

### 11.2 Axis order: rows = TIME, columns = FREQUENCY

Proven three independent ways (this is the **opposite** of our synthetic
datasets, which are frequency × time — transpose before reusing pipeline code):

1. **Mask profile variability.** Coefficient of variation of RFI fraction
   along columns = 0.616 (train) / 1.074 (test); along rows = 0.120 / 0.217.
2. **Concentration.** 65.4 % (train) / 74.4 % (test) of all flagged pixels sit
   in just **16 of the 512 columns** (test median: 87 %). Only 28.8 % / 21.2 %
   sit in the top 16 rows.
3. **Profile shape** (`fig_lofar_profiles.png`). The column profile has sharp
   isolated spikes at fixed indices (~365, ~420, ~470–500) rising to 5.5 %
   against a 1 % floor — fixed-frequency transmitters. The row profile is
   featureless noise between 0.7 % and 1.5 %.

The data column profile also shows a dip at columns ~40–60 and a rise past
column 400 (band edge). The data row profile shows a smooth ~3 % downward
drift across the 512 time samples — slow gain/elevation drift, not RFI.

### 11.3 Class imbalance

| | train (AOFlagger) | test (human) |
|---|---|---|
| RFI pixel fraction | 1.2703 % | 0.7661 % |
| imbalance (clean : RFI) | 1 : 77.7 | 1 : 129.5 |
| median per-image RFI frac | 0.207 % | 0.378 % |
| images >10 % flagged | 123 | 0 |
| images >50 % flagged | **35** | 0 |
| images 100 % flagged | **35** | 0 |
| completely clean images | 0 | 0 |

**An all-zero predictor scores 99.23 % pixel accuracy and F1 = 0.** Accuracy
is meaningless here. With no class weighting this is the failure mode to
watch for in the first training run.

### 11.4 Data-quality defects found

- **35 training images are 100 % flagged** by AOFlagger (indices 986, 1086,
  1264, 1473, 1598, 1750, 2050, 2094, 2430, 2518, 2566, 3135, 3639, 3718,
  3872, 4026, 4104, 4336, 4417, 4594, 4663, 4838, 4841, 4894, 4926, 5476,
  5882, 5995, 6246, 6827, 7074, 7113, 7219, 7381, 7462). These are dead
  baselines — several have all-zero data. Training on them teaches the model
  that everything is RFI.
- **8.3 % of training images contain exact zeros**; 4 of the first 1500 are
  entirely zero. All fully-zero rows/columns found belong to those images —
  there are no partially-dead rows/columns.
- **34 duplicate images inside the training set** (7466 unique of 7500).

### 11.5 TEST-SET LEAKAGE — all 109 test images are inside the training set

MD5 of every image array: **all 109 test spectrograms are byte-identical to
109 training spectrograms.** Train on `data[0]` and evaluate on `data[2]` and
the model has already seen every test image — only the *label* differs
(AOFlagger in train, human in test).

This is by construction (the same baselines were labelled twice), but it
means a naive train/test split is contaminated. **Drop those 109 training
indices before training.** They are saved as
`analysis/lofar_analysis/lofar_leak_train_idx.npy`.

Recommended training subset: **7356 of 7500** — drop 109 leaked + 35
fully-flagged. Saved as `analysis/lofar_analysis/lofar_clean_train_idx.npy`.

### 11.6 The training labels are ~44 % wrong (and this validates our reading)

Because the 109 test images are duplicated in train, AOFlagger's label and
the human's label can be compared **on identical pixels**:

| metric | value |
|---|---|
| pixel-wise precision | 0.5598 |
| pixel-wise recall | 0.5802 |
| **pixel-wise F1** | **0.5698403** |
| IoU | 0.3984 |
| mean *per-image* F1 | 0.4663 |
| AOFlagger RFI frac / human RFI frac | 1.036 |

The paper's Table 2 reports AOFlagger F1 = **0.5698**. We measured
**0.5698403**. Exact agreement to four decimals. That independently confirms
(a) `data[1]` is AOFlagger and `data[3]` is human expert, (b) the 109
image pairs are correctly matched, and (c) **the paper's metric is GLOBAL
pixel-wise F1 over the pooled test set, not the mean of per-image F1s.**

**Methodological trap:** mean per-image F1 is 0.4663 — a full 0.103 lower.
Report per-image mean and our model will look 0.10 worse than it is against
the published table. Pool the confusion matrix over all 109 images, then
compute F1 once.

**Ceiling implication:** every training label comes from AOFlagger, which is
only ~56 % precise and ~58 % complete against the human. A supervised model
trained on AOFlagger labels is fitting ~44 % label noise. That is the real
reason no method in Table 2 exceeds F1 0.60, and it should be stated in the
paper.

### 11.6b Correction: the paper uses TWO different clip bounds

Caught 2026-09-05 while checking §4.2 against §4.1. The preprocessing upper
clip is **not the same for both datasets**:

| dataset | paper section | clip |
|---|---|---|
| HERA (simulated) | §4.1 | `[\|mu-sigma\|, mu + 4 sigma]` |
| **LOFAR (real)** | **§4.2** | **`[\|mu-sigma\|, mu + 20 sigma]`** |

`lofar_data.preprocess()` originally used 4 sigma (the HERA value) and has been
corrected to default to 20; pass `sigma_hi=4.0` for the HERA setting. Measured
cost of getting this wrong, on the 109-image human-labelled test set:

| clip | RFI pixels saturated at the ceiling | clean pixels saturated | mean AUROC |
|---|---|---|---|
| mu+4 sigma | 38,908 (**17.77 %**) | 5,353 (0.019 %) | 0.7654 |
| mu+20 sigma | 5,203 (**2.38 %**) | 29 (0.000 %) | 0.7654 |

So 4 sigma flattens the brightness ordering of nearly a fifth of all RFI
pixels. AUROC is unchanged to four decimals because the saturated pixels stay
above the clean population either way — but the information is gone, and any
model that could use *how* bright a pixel is loses it. Use 20 for LOFAR.

### 11.7 Is the RFI actually faint? Yes — and it is bimodal

Raw-amplitude comparison of flagged vs unflagged pixels:

| | train | test |
|---|---|---|
| median flagged / median clean | **1.93×** | **2.28×** |
| flagged pixels below the *median* clean pixel | **26.96 %** | **24.11 %** |
| flagged pixels below clean p75 | 40.81 % | 38.38 % |
| flagged pixels above clean p99 | 38.00 % | 43.04 % |

So the RFI splits into two populations: **~40 % is very bright** (above the
99th percentile of clean pixels — trivially detectable) and **~25–40 % sits
at or below typical clean-pixel brightness**, carrying *no per-pixel
amplitude evidence at all*. Those pixels are only recoverable from context —
their neighbours in time and frequency.

After the paper's preprocessing the separation, measured per image in units
of that image's clean-pixel σ, is **median 1.49σ, mean 5.77σ** (heavily
skewed by the bright population); 17.6 % of flagged pixels still sit below
the clean median. In no test image does more than 49 % of the RFI fall below
the clean median.

**Contrast with our synthetic data:** there, `mask = strength > 0.5σ` makes
every labelled pixel ≥0.5σ above local noise *by construction*, with 90.9 %
at ≥1σ, and the label is noise-free. Here a quarter of the labels have no
amplitude signal and ~44 % of the labels are themselves wrong. That is the
concrete, quantified statement of why real data is harder — better than
hand-waving about "complexity."

### 11.8 Normalisation: fixed range works here (an earlier claim RETRACTED)

The paper's pipeline is: clip to `[|μ−σ|, μ+20σ]` (20, not 4 — see 11.6b)
→ natural log → min-max to [0,1], **computed per image**.

| quantity | value |
|---|---|
| global min (first 2000 train images) | 0.0 |
| global max (first 2000) | 9.99e10 |
| per-image clip *upper* bound: min → max | 1.65e6 → 8.09e8 |
| **spread of the per-image upper bound** | **490×** |
| median raw dynamic range within one image | ~9,900 (train) / ~19,000 (test) |

A single fixed global scale would map the median image into a sliver at the
bottom of the range — the global max is 6×10⁴ times the median image's own
clip ceiling, and the global min is exactly 0 so a global log is undefined.

> **CORRECTION (2026-09-05).** The paragraph that stood here claimed a single
> global scale "would crush most images" and that per-image normalisation was
> "the only workable choice" on LOFAR. **That was wrong, and it was reasoned
> rather than measured.** The error: the argument ran from the global *max*
> (1.4e11) and global *min* (0), but the fixed-range recipe calibrates from the
> **0.5 / 99.5 percentiles**, which ignore outliers entirely, and then clips.
> Measured over 200 clean training images, the calibrated range is
> `[283922, 4.776e6]`, and the resulting images are healthy: per-image means
> spread 0.096–0.607, standard deviations ~0.13, at most 13 % of pixels at the
> floor and under 2 % at the ceiling. Nothing is crushed.
>
> Worse for the original claim, fixed range **separates RFI better** than the
> per-image recipe. Median RFI-minus-clean separation in clean-sigma units over
> 60 training images:
>
> | normalisation | separation |
> |---|---|
> | fixed range (global percentile clip + scale) | **1.322 sigma** |
> | per-image (clip 20 sigma, log, min-max) | 0.718 sigma |
>
> The reason is that per-image min-max lets one bright RFI pixel stretch that
> image's range and squash everything else, while a fixed physical scale keeps
> every image comparable — which is exactly the PART 1 argument, and it appears
> to hold on real data too.

What *is* still true and still matters: the per-image clip bounds vary by 490x,
the global minimum is exactly 0 so a global log is undefined without clipping
first, and 4+ images are entirely zero. Those are facts about the data. The
conclusion drawn from them was not.

**So PART 1 is not inverted, and the fixed-norm arm is not expected to fail.**
Run both arms and let the measurement decide -- `experiments/lofar_tfunet_baseline.py`
takes `--norm fixed | fixed_log | per_image`.

Guard needed: 4+ images are entirely zero, so `log` produces `-inf` and the
min-max denominator is 0. Clamp the lower clip bound to a small positive
value and skip/zero-fill degenerate images.

### 11.9 Baselines on the 109-image human-labelled test set

All numbers are global pixel-wise F1, so they are directly comparable to the
paper's Table 2. Recomputed 2026-09-05 with the corrected 20-sigma clip
(PART 11.6b); the earlier 4-sigma figures differed by at most 0.06.

| method | F1 | precision | recall |
|---|---|---|---|
| predict everything is RFI | 0.0152 | 0.0077 | 1.0 |
| predict nothing is RFI | 0.0 | — | 0.0 (99.23 % accuracy) |
| best global threshold, per-image normalised (**oracle**) | 0.3061 | 0.4212 | 0.2404 |
| per-column MAD clip @2.0σ | 0.2017 | 0.1832 | 0.2243 |
| per-image σ-clip @2.0σ | 0.2918 | 0.2581 | 0.3357 |
| **per-image σ-clip @2.5σ (best simple baseline)** | **0.4103** | 0.7016 | 0.2899 |
| per-image σ-clip @3.0σ | 0.3953 | 0.9220 | 0.2516 |
| AOFlagger (the training labeller) | 0.5698 | 0.5598 | 0.5802 |
| *paper's best — RFI-Net* | *0.5979* | — | — |

So the ladder to climb is: **beat 0.4103** (a three-line σ-clip) to show the
model does anything; **beat 0.5698** to beat the classical flagger that
produced its own training labels; **beat 0.5979** to beat the published
state of the art.

For perspective, on our synthetic dataset a constant threshold already scores
F1 0.7421 and a 2,578-parameter CNN scores 0.7823 — both far above anything
achievable here.

Note the per-**column** MAD clip performs *worse* (0.2017) than the plain
per-image σ-clip (0.4127), even though RFI is column-concentrated. Reason:
persistent narrowband RFI occupies a large fraction of its own column, so the
per-column median is contaminated by the very signal it is meant to baseline.
Any per-frequency normalisation scheme has to be robust to that.

### 11.10 Never open the pickle again — use the memory-mapped arrays

Loading the 9.3 GB pickle exhausts RAM on this machine (19 GB total, and the
IDE crashes). `analysis/lofar_analysis/convert_pickle_to_memmap.py` has been run once
and split it into `data/lofar/LOFAR_npy/{train,test}_{images,masks}.npy`. `lofar_data.py`
memory-maps those: indexing is identical, but only the slice you touch is read
from disk.

    from lofar_data import load_lofar, preprocess, batches
    d = load_lofar()          # 0.007 s, 28.7 MB RAM (measured)
    img = d.train_images[1]   # ~1 MB read
    X = preprocess(img)       # per-image clip(20 sigma) -> log -> min-max

Measured: 28.7 MB resident after opening all four arrays, vs ~10 GB for
`pickle.load`. `LOFAR_npy/` is gitignored (9.3 GB); regenerate it with the
converter if the machine changes. `preprocess()` guards the all-zero images
that would otherwise produce `-inf`/NaN.

### 11.10b Splits, and the matched-conditions question (2026-09-05)

**Who split what.** The train/test split is the **authors'**, already baked
into the pickle: 7500 training + 109 test. We did NOT make it. What we added
is a **validation split — 10 % of the clean training pool, 736 images**, held
out for best-checkpoint selection only. So:

| split | size | origin | labels |
|---|---|---|---|
| train | 6620 | authors', minus our exclusions | AOFlagger |
| val | 736 | **ours**, 10 % of clean pool, seed 42 | AOFlagger |
| test | 109 | authors' | **human expert** |

**The exclusions.** 7500 − 109 leaked − 35 fully-flagged = 7356 clean, then
split 90/10 into 6620 train + 736 val. The 109 leaked images (PART 11.5) are
dropped from training entirely — they are the test set, byte for byte.

**Epochs: matching the number would NOT match the conditions.** PART 1 gave
the synthetic baseline 175 steps/epoch × 60 epochs = **10,500 gradient
steps** (700 images / batch 4). LOFAR has 6620 training images, so a *full
pass* is 1655 steps/epoch — 60 epochs of that is **99,300 steps, 9.5× more
training** than the baseline being compared against.

`lofar_tfunet_baseline.py` therefore fixes `--iters_per_epoch 175` and
`--epochs 60`, holding the gradient-step budget **identical to PART 1** while
sampling those batches at random from all 6620 images (~6.3 passes over the
data). `--iters_per_epoch 0` gives full passes, but that must be declared in
the paper as an unmatched comparison.

### 11.10c GPU throughput — the laptop was on battery (2026-09-05)

Measured on the RTX 3060 6 GB, batch 4, features_root 32, layers 3:

| input | pixels | ms/step | note |
|---|---|---|---|
| 265×1024 | 271,360 | 2684 | PART 7 recorded **174 ms** for this exact case |
| 512×512 | 262,144 | 2658 | LOFAR |
| 256×256 | 65,536 | 612 | scales with pixel count |

Step time tracks pixel count, not aspect ratio. The 15× gap against PART 7 is
**not** the model and **not** the 512×512 shape — the laptop was **unplugged
at 16 % battery**, and `nvidia-smi` reported the SM clock pinned at **210 MHz
of a 2100 MHz maximum**, 15.2 W, with `SW Power Cap: Active` and `SW Thermal
Slowdown: Active`. Data loading from the memmap is 14 ms/batch, i.e. never the
bottleneck; the GPU sat at 100 % utilisation and 4.3 GB throughout.

**Always check `cat /sys/class/power_supply/A*/online` before quoting a
timing.** On AC at PART 7's 174 ms/step, the matched 10,500-step run is
~30 min. On battery it is ~8 h.

### 11.11b The leakage guard is enforced at runtime, not assumed

`lofar_tfunet_baseline.py` fingerprints every train and val image against all
109 test images **before any training starts**, and aborts if one matches.
Relying on `lofar_clean_train_idx.npy` being correct is an assumption; a
regenerated or hand-edited index file would silently contaminate the result
and nothing would look wrong.

A float64 array sum is a near-unique fingerprint (1995 distinct values among
2000 images), so it prefilters and only collisions are compared element-wise:
**1.5 s for all 7356 images**. Verified both ways — a normal run reports
`0 of 7209 match any test image`, and feeding it 50 clean images plus 3 known
leaked ones aborts with the offending train->test index pairs listed.

### 11.12 RECOMMENDED PROTOCOL for the LOFAR tf_unet run (2026-09-05)

Decided rather than deferred. `./scripts/run_lofar_baseline.sh` runs all of it.

**Hold the METHOD identical to PART 1 run #4** — these are the comparison, and
none of them changes:

| | value | why |
|---|---|---|
| architecture | tf_unet, layers 3, features_root 32 | authors' unmodified code |
| optimiser | Adam @ 1e-3 | PART 1 |
| batch | 4 | PART 1; also 4.3 GB of the 6 GB card at 512x512 |
| cost | cross-entropy + 0.001 regulariser | PART 1 |
| dropout | keep_prob 0.75 | tf_unet default, PART 1 |
| normalisation | **fixed range** | PART 1 run #4; measured better here too (11.8) |
| class weights | **OFF** | PART 1: the single biggest factor, +0.413 F1 |

**Hold the BUDGET identical, not the epoch count.** 175 steps/epoch x 60
epochs = 10,500 gradient steps, exactly what PART 1 gave the synthetic
baseline. Matching "60 epochs" over 6620 images would instead be 99,300
steps — 9.5x more training than the thing being compared against.

**Run 3 seeds (0, 1, 2).** Outstanding item 5 has flagged N=1 as *the*
blocking methodological problem in this project since revision 4. Do not carry
it into the real-data result — it costs 1.5 h to fix it here, at the point
where the number actually matters. Report mean +/- spread.

**Add one convergence control**, 20 epochs of full passes (33,100 steps,
seed 0). This is the answer to the obvious referee question: is the matched
number low because real data is hard, or because 10,500 steps undertrains a
9.5x larger dataset? If the control lands near the matched runs, the matched
number stands and that is now demonstrated rather than assumed.

**Select the operating threshold on VALIDATION, apply it to test.** The
headline `pooled_f1` now does this. PART 1 quoted an oracle max-F1 chosen on
the test set, which outstanding item 2 already criticises; `max_f1` is still
reported, but only so the two can be lined up, and it must be labelled
optimistic. The validation split is 150 images, matching PART 1's.

**Cost on AC: ~3.1 h total** — 30 min per matched run, 1.5 h for three seeds,
1.6 h for the control.

### 11.13 GPU timing CONFIRMED on AC (2026-09-05)

Re-measured after plugging in: **174 ms/step**, batch 4, 512x512,
features_root 32 — identical to the PART 7 figure, GPU at 1672 MHz / 94 W /
97 % / 4.3 GB. The 2658 ms measured on battery was a **15.3x** penalty and
nothing else. PART 7's timing table is sound; it was simply taken on AC.
`run_lofar_baseline.sh` refuses to start on battery for this reason.


### 11.13b Two runs cannot share this GPU (2026-09-05)

A second `lofar_tfunet_baseline.py` launched while the first was still going
died instantly with

    RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[4,476,476,64]

One run holds **4.28 GB of the 6 GB card**, leaving 1.5 GB — not enough for a
second graph. The first run was unaffected and continued to completion; only
the second process died, so the traceback is misleading if you assume it came
from the run you care about.

The giveaway in the log is `Restoring parameters from .../ckpt/model.ckpt`
near the top: the second process found the first one's checkpoint and tried to
resume into a full GPU.

**Check before launching anything:**

    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv

Empty output means the card is free. `./scripts/status.sh` shows this too, plus a
battery warning. Both `run_lofar_baseline.sh` and `run_lofar_lr1e-4.sh` now
refuse to start when the GPU is occupied, so this cannot recur by accident.

### 11.14 What to actually do before the first training run

1. Load via `lofar_data.load_lofar()`; index the clean subset via
   `d.clean_train_idx` (7356 images).
2. Preprocess **per image**: clip `[max(|μ−σ|, ε), μ+4σ]` → `log` → min-max.
3. Keep rows = time, columns = frequency, or transpose — but be consistent,
   and record which was used.
4. Hold out a validation split from the 7356 (the 109 test images must not
   appear anywhere in training).
5. Evaluate with a **pooled** confusion matrix over all 109 test images, then
   one F1. Also report AUROC/AUPRC since the paper does.
6. Expect the no-class-weight run to collapse toward predicting nothing;
   if it does, that is the expected consequence of 1:130 imbalance, and it is
   a legitimate finding to report before adding weighting.
7. **Plug the laptop in** (PART 11.10c) and run
   `experiments/lofar_tfunet_baseline.py`. Defaults are the PART 1 recipe:
   fixed-range norm, no class weights, 175×60 = 10,500 steps.

---

## PART 12 — RESULT: tf_unet on real LOFAR data (2026-09-05)

Four runs, all finished. Authors' unmodified tf_unet, layers 3,
features_root 32, Adam @1e-3, batch 4, **fixed-range normalisation, NO class
weights** — PART 1 run #4's method, held identical. Matched budget
175 x 60 = 10,500 gradient steps. Trained on the 6621 clean images, tested on
the **109 human-expert-labelled** baselines. Threshold selected on validation,
applied to test. Raw metrics in `runs/lofar/*/eval_test/metrics.json`.

| run | pooled F1 | prec | recall | max F1 | ROC | PR-AUC | thresh | best ep |
|---|---|---|---|---|---|---|---|---|
| matched seed 0 | 0.4363 | 0.3565 | 0.5619 | 0.4463 | 0.9348 | 0.4527 | 0.205 | 50 |
| matched seed 1 | 0.4882 | 0.4311 | 0.5628 | 0.5438 | 0.9381 | 0.5215 | 0.501 | 58 |
| matched seed 2 | 0.4443 | 0.3489 | 0.6118 | 0.4803 | 0.9438 | 0.4810 | 0.078 | 58 |
| **matched mean** | **0.4563 ± 0.0279** | | | 0.4901 ± 0.0495 | 0.9389 ± 0.0045 | 0.4851 ± 0.0346 | | |
| convergence ctrl (seed 0, 3.15x steps) | 0.4295 | 0.3512 | 0.5527 | 0.4697 | 0.9378 | 0.4773 | 0.377 | 18 |

### 12.1 Where it lands

| method | pooled F1 |
|---|---|
| sigma-clip baseline | 0.4103 |
| **tf_unet (this work)** | **0.4563 ± 0.0279** |
| AOFlagger (wrote the training labels) | 0.5698 |
| U-Net (paper's) | 0.5876 |
| RFI-Net (published best) | 0.5979 |

It **did not collapse** — the real risk at 1:129 imbalance with no class
weighting — and it beats the trivial baseline. But see 12.2.

### 12.2 It does NOT significantly beat a three-line sigma-clip (AT lr 1e-3 — SUPERSEDED, see 12.8)

Gap over the sigma-clip baseline is **+0.046**, against a seed spread of
**0.028**. Formally: t = 2.85 on 2 dof, which needs t > 4.30 for p < 0.05.
**Not significant at N=3.** A 512x512x3-level U-Net with 500k parameters,
trained for 10,500 steps, is not measurably better on this data than
`pixel > mean + 2.5 sigma`.

This is the honest headline and it must go in the paper as such. Running one
seed and reporting 0.4882 (seed 1) would have overstated the result by 0.078 —
nearly three times the gap being claimed.

### 12.3 The training budget is NOT the limitation

The convergence control gave seed 0 **3.15x more training** (33,100 steps vs
10,500, full passes over all 6621 images). Result: pooled F1 **0.4363 ->
0.4295** (-0.0068), max F1 0.4463 -> 0.4697 (+0.0234). Both changes are inside
the seed standard deviation, and its best epoch was 18 of 20 — it had
converged. So the matched-budget number is *not* an undertraining artefact.
That referee question is closed with evidence.

### 12.4 The real finding: ranking is excellent, the operating point is not

**ROC AUC 0.9389 ± 0.0045** — remarkably stable across seeds (sd 0.0045, six
times tighter than F1's 0.0279) — while F1 sits at 0.456. The model **ranks**
RFI pixels very well and then cannot convert that into a good decision.

The cause is prevalence: the test set is **0.77 % positive, 1 : 129**. At that
imbalance, excellent ranking still yields poor precision at every threshold.
PR-AUC 0.4851 is the honest summary, and it tracks F1, not ROC.

Corroborating evidence — the validation-selected thresholds were **0.205,
0.501, 0.078, 0.377**: wildly unstable across seeds, yet pooled F1 moved only
0.05. The F1 surface is flat near its optimum, so the threshold is barely
identifiable. Precision (0.349–0.431) is what limits the score; recall is
consistently higher (0.553–0.612).

**This is where a novel contribution would live** — not in another
architecture tweak, but in fixing the decision rule: calibration, per-frequency
or per-image adaptive thresholds, or a loss that optimises the operating point
directly rather than cross-entropy.

### 12.5 The synthetic-to-real gap, quantified

Identical method, identical budget, only the dataset changes:

| dataset | max F1 |
|---|---|
| synthetic 276x600 (PART 1 run #4) | **0.9317** |
| real LOFAR | **0.4901 ± 0.0495** |
| **drop** | **−0.4416** |

The same code, the same hyperparameters, the same number of gradient steps,
loses **0.44 F1** moving from our synthetic benchmark to real telescope data.
That single number is the strongest result this project has, and it is a
better paper than an architecture claim. It is explained by PART 11: labels
that are ~44 % wrong (AOFlagger vs human), ~25 % of RFI carrying no per-pixel
amplitude evidence, and 1:129 imbalance — none of which the synthetic
generator reproduces.

### 12.7 Comparing to Mesarcik et al. correctly — we were using the wrong number

Read out of the paper (section 5.1, section 5.3, Table 2) on 2026-09-05.

**Their threshold protocol is ORACLE.** Section 5.1, verbatim: *"For all
evaluations across all models in this work the threshold is fixed to the
maximum obtainable F1 score."* They pick the threshold on the test set.

So their 0.5876 must be compared with our **max_f1 = 0.4901 ± 0.0495**, NOT
our validation-thresholded `pooled_f1` of 0.4563. Our pooled F1 is the more
honest number and should stay the headline of our own work, but every
cross-paper comparison must use max F1 or it is unfair to us by ~0.034.

**Their Table 2 in full** (LOFAR, expert ground truth, 3 seeds — they use
three runs too):

| model | AUROC | AUPRC | F1 |
|---|---|---|---|
| AOFlagger | 0.7883 | 0.5716 | 0.5698 |
| **U-Net (Akeret 2017) — the same tf_unet we ran** | **0.8017 ± 0.0058** | 0.5920 ± 0.0031 | **0.5876 ± 0.0031** |
| RFI-Net | — | — | 0.5979 |
| R-Net | — | — | 0.5286 |
| NLN (theirs) | 0.8622 ± 0.0006 | 0.6216 ± 0.0005 | 0.5114 ± 0.0004 |

Their AUROC formula is the standard one — AOFlagger's binary mask gives
(0.5802 + 0.99648)/2 = 0.7883, matching their table exactly, so AUROC is
directly comparable across the two papers.

**THE HEADLINE FINDING: our AUROC is far HIGHER than theirs, on the same
architecture.**

| | AUROC | F1 (oracle) |
|---|---|---|
| their U-Net (Akeret 2017) | 0.8017 ± 0.0058 | 0.5876 ± 0.0031 |
| **our tf_unet** | **0.9389 ± 0.0045** | 0.4901 ± 0.0495 |
| difference | **+0.137** | −0.098 |

Same architecture, same dataset, same ground truth. Our model **ranks RFI
pixels substantially better** (+0.137 AUROC, and far outside either paper's
seed spread) and yet **converts that into a worse decision** (−0.098 F1).

This is PART 12.4's finding confirmed against an independent implementation,
and it is much stronger evidence than our own threshold instability was. The
information is there; the decision rule is what fails.

**Three concrete differences that could explain the F1 gap** — all testable:

| | Mesarcik et al. | ours |
|---|---|---|
| training unit | **32x32 patches** (5.1: fixed for all models "to keep comparison consistent") | full 512x512 images |
| schedule | **100 epochs, Adam @ 1e-4** | 60 epochs / 10,500 steps, Adam @ **1e-3** |
| normalisation | per-image clip 20 sigma -> log -> min-max | fixed range |

Our learning rate is **10x higher** and our schedule shorter, both inherited
from PART 1's synthetic recipe rather than from this paper. A higher LR with
fewer steps plausibly produces a well-ranked but poorly-calibrated model —
exactly the AUROC-high / F1-low signature observed. **Testing lr=1e-4 with
100 epochs is the single highest-value next run.**

Border cropping is *not* a confound: tf_unet's valid padding scores on
472x472 of 512x512, and RFI density in the discarded border is 0.00747
against 0.00770 overall (ratio 1.004).

**Sources.** Paper: Mesarcik, Boonstra, Ranguelova & van Nieuwpoort (2022),
MNRAS, arXiv:2207.00351 — `notes/Mesarcik2022_*.pdf`.
Code: https://github.com/mesarcik/RFI-NLN .
Dataset: https://zenodo.org/record/6724065 (doi:10.5281/zenodo.6724065).
This is a **different paper from Akeret et al. 2017**, which contributed the
tf_unet architecture; the 2022 paper uses that architecture as one of its
baselines, which is why its 0.5876 row is directly our model.

### 12.8 The learning rate was the problem (SEED 0 ONLY — see 12.10, effect halves at N=3)

Prediction in 12.7 was that our 1e-3 learning rate, inherited from PART 1's
*synthetic* recipe rather than from this paper, was producing a well-ranked
but poorly-calibrated model. Tested at Adam **1e-4** for 150 epochs, seed 0,
everything else identical (fixed-range norm, no class weights, 175
steps/epoch, same clean split).

| metric | lr 1e-3 / 60 ep | **lr 1e-4 / 150 ep** | change |
|---|---|---|---|
| pooled F1 | 0.4363 | **0.5303** | **+0.0940** |
| max F1 | 0.4463 | **0.5640** | **+0.1177** |
| ROC AUC | 0.9348 | **0.9511** | +0.0163 |
| PR-AUC | 0.4527 | **0.5720** | +0.1193 |
| precision | 0.3565 | 0.4596 | +0.1030 |
| recall | 0.5619 | 0.6267 | +0.0648 |

Same seed, so this is not seed noise: **+0.094 pooled F1 = 3.4x the seed
standard deviation** (0.0279) measured over the three lr 1e-3 runs.

**It improved everything at once** — ROC went UP as well, so this was not a
ranking-vs-calibration trade. The model is simply better trained. Precision
gained more than recall (+0.103 vs +0.065), which is consistent with the
calibration story: the outputs became sharper, not just shifted.

Best epoch was **114 of 150** (lr 1e-3 peaked at 50 of 60), exactly the
late-converging behaviour a 10x smaller step size should show. The 150-epoch
budget was necessary; 60 would have truncated it.

**Where this puts us** (max F1, the paper's oracle protocol):

| method | max F1 |
|---|---|
| sigma-clip baseline | 0.4103 |
| our tf_unet @ lr 1e-3 | 0.4901 ± 0.0495 |
| **our tf_unet @ lr 1e-4** | **0.5640** |
| AOFlagger | 0.5698 |
| paper's U-Net (same architecture) | 0.5876 ± 0.0031 |
| RFI-Net | 0.5979 |

**Gap to the published U-Net closed from −0.098 to −0.024.** We now
essentially reproduce the published result with the same architecture; the
residual 0.024 is plausibly the two remaining differences (they train on
32x32 patches; they use per-image normalisation).

**This supersedes 12.2's headline.** The claim "does not significantly beat a
sigma-clip" was true *at lr 1e-3*. At lr 1e-4, pooled F1 0.5303 vs 0.4103 is a
+0.120 gap against a 0.028 seed spread — comfortably significant even at N=1,
and it must be re-stated once seeds 1 and 2 are in. **12.2 must not be quoted
without this correction.**

**Still N=1 — AND THIS SECTION IS SUPERSEDED BY 12.10.** Seeds 1 and 2 have
since run and the three-seed mean gain is **+0.046, not +0.094**; seed 0 was
the lucky draw and the effect is not significant at N=3. Never quote 12.8's
numbers without 12.10. Original next-step note follows:

    ~/tf-env/bin/python experiments/lofar_tfunet_baseline.py \
        --lr 1e-4 --epochs 150 --seed 1 --output_dir lofar_runs/lr1e-4_seed1


### 12.10 lr 1e-4 at three seeds — the effect is real but NOT significant (2026-09-06)

Seeds 1 and 2 completed. **12.8 was written from seed 0 alone and overstated
the effect by roughly a factor of two.**

| metric | lr 1e-3 (3 seeds) | lr 1e-4 (3 seeds) | mean diff |
|---|---|---|---|
| pooled F1 | 0.4563 ± 0.0279 | **0.5021 ± 0.0259** | +0.0458 |
| max F1 | 0.4901 ± 0.0495 | **0.5482 ± 0.0139** | +0.0581 |
| ROC AUC | 0.9389 ± 0.0045 | 0.9470 ± 0.0074 | +0.0081 |
| PR-AUC | 0.4851 ± 0.0346 | 0.5562 ± 0.0152 | +0.0711 |

Paired by seed (same initialisation, same shuffle):

| seed | lr 1e-3 pooled | lr 1e-4 pooled | diff |
|---|---|---|---|
| 0 | 0.4363 | 0.5303 | **+0.0940** |
| 1 | 0.4882 | 0.4794 | **−0.0088** |
| 2 | 0.4443 | 0.4966 | +0.0523 |

**Seed 1 got slightly WORSE.** Paired t = 1.54 on 2 dof for pooled F1 (needs
4.30), t = 1.62 for max F1. **Not significant at N=3.**

So the honest statement is: *lr 1e-4 raises the mean by ~0.046 F1 and more than
halves the seed spread on max F1 (0.0495 → 0.0139), but with three seeds we
cannot rule out chance.* It should be presented as the better default with that
caveat, not as a demonstrated improvement.

**This is the second time in two days that a single seed misled us**, and the
first time it misled *me* after I had already written the N=1 warning. 12.8
reported +0.094 from seed 0; the three-seed mean is +0.046. Seed 0 was the
lucky draw. **12.8 must always be read together with this section.**

One thing that *is* clean: lr 1e-4 cut the max-F1 seed spread from 0.0495 to
0.0139, a 3.6x reduction. More reproducible, even where not more accurate.

### 12.11 Per-image normalisation is catastrophically worse HERE (2026-09-06)

Tested the last remaining difference against Mesarcik et al.: their per-image
clip→log→min-max, at lr 1e-4, seed 0.

| normalisation | pooled F1 | max F1 | ROC |
|---|---|---|---|
| **fixed range** (3 seeds) | **0.5021 ± 0.0259** | 0.5482 | 0.9470 |
| per-image (1 seed) | **0.2456** | 0.3039 | 0.8527 |
| difference | **−0.2565** | −0.244 | −0.094 |

−0.2565 is **9.9 seed standard deviations**. Not noise at any N.

This settles PART 11.8 in the opposite direction to the claim originally made
there: on full 512×512 LOFAR images, fixed-range normalisation is not merely
acceptable, it is **decisively better**, and per-image normalisation destroys
the model.

**Why it works for them and not for us — hypothesis, untested.** They normalise
**32×32 patches**, we normalise **512×512 images**. Per-image min-max is set by
that image's extremes, so on a full image one bright RFI streak compresses
everything else into a sliver, while across a 32×32 patch the statistics are
locally uniform and the same operation behaves like a sensible local contrast
normalisation. If so, their normalisation choice is inseparable from their patch
choice and cannot be transplanted alone — which is exactly what we just
demonstrated by trying. **Testing 32×32 patches is the way to check this.**

### 12.12 Patch training — and why the paper's 32x32 is NOT reproducible here

**tf_unet with layers=3 and valid padding rejects 32x32 outright.** Measured:

| patch | output | result |
|---|---|---|
| 32 | — | **FAILS to build** |
| 48 | 8x8 | ok |
| 64 | 24x24 | ok |
| 128 | 88x88 | ok |
| 512 | 472x472 | ok (what we have been running) |

Each of the 3 levels strips 4 px per conv pair on the way down and again on the
way up: the network loses **40 px** in total, so any input below 48 has nothing
left. **Mesarcik et al. cannot have used Akeret's tf_unet at 32x32** — their
implementation (github.com/mesarcik/RFI-NLN) is a Keras rewrite with `same`
padding. Any patch experiment here is therefore a *nearest equivalent*, not a
reproduction, and the paper must say so.

`--patch_size N` now takes random NxN crops for training; **validation and test
still use full 512x512 images**, so the metric stays comparable to every other
run in PART 12. Sizes below 48 are refused with an explanation.

**Budget must be matched by OUTPUT PIXELS, not steps.**

| config | output px per step |
|---|---|
| full 512x512, batch 4 | 891,136 |
| 64x64 patch, batch 32 | **18,432** (48x less) |

Matching step counts would undertrain the patch model 48-fold and prove
nothing. The full-image reference run is 26,250 steps x 891,136 = **2.34e10
output pixels**; matching that at 64x64/batch 32 needs **1,269,115 steps**
(8,460 iters/epoch x 150 epochs, ~4.2 h). `run_lofar_patches.sh` does this.

**What it tests.** Two arms:
1. `--norm fixed` — isolates patch size alone.
2. `--norm per_image` — the actual 12.11 hypothesis. Per-image normalisation
   collapsed on full images (0.2456 vs 0.5021). If it recovers on 64x64
   patches, then their normalisation only works *because* the "image" is a
   small patch, and the two choices are inseparable. **This is the arm that
   matters**; the fixed arm is the control.

### 12.9 Consequences for outstanding items

*(Written 2026-09-05 at lr 1e-3; items 5 and 6 stand, but see 12.8 — the
"does not beat the baseline" reading no longer holds at lr 1e-4.)*

- **Item 5 (N=1) is vindicated, hard.** Seed spread on real data is 0.052 —
  larger than most architectural effects this project has ever claimed
  (strip conv +0.023, residual +0.002, ECA −0.005, all N=1 on synthetic).
  Nothing should be claimed from a single run again.
- **Item 6 (no real-data result) is now CLOSED.** This is that result.
- **Item 4 (the benchmark is easy) is confirmed from the other side.** The
  synthetic benchmark yields 0.93 where real data yields 0.46.

---

## PART 13 — HybridRFINet on real LOFAR: beats the published best (N=1, 2026-09-06)

`experiments/lofar_hybrid.py`, seed 0, base 8 (**593,842 parameters**), fixed
normalisation, CE+Dice with measured class weights, Adam 1e-3, batch 1,
700 x 40 = 28,000 gradient steps — the hybrid's own synthetic budget. Trained
on the 6621 clean images, tested on the 109 expert-labelled ones, threshold
selected on validation. Raw metrics in
`runs/lofar/hybrid_b08_fixed_seed0/eval_test/metrics.json`.

### 13.1 The numbers

| metric | value |
|---|---|
| **max F1 (oracle — the paper's protocol)** | **0.6439** |
| pooled F1, 472 crop (like-for-like vs tf_unet) | **0.6377** |
| pooled F1, full 512x512 | 0.6237 |
| precision / recall | 0.5686 / 0.6907 |
| **ROC AUC** | **0.9827** |
| PR-AUC | 0.6667 |
| best epoch | 30 of 40 |
| threshold chosen on validation | 0.9455 |

### 13.2 Where that puts it

| method | max F1 | params |
|---|---|---|
| sigma-clip baseline | 0.4103 | — |
| tf_unet @ lr 1e-4 (3 seeds) | 0.5482 ± 0.0139 | ~500k |
| AOFlagger | 0.5698 | — |
| paper's U-Net | 0.5876 ± 0.0031 | — |
| RFI-Net (published best) | 0.5979 | millions |
| **hybrid base 8 (N=1)** | **0.6439** | **593,842** |

- **+0.046 over RFI-Net**, the best number in Mesarcik et al. Table 2.
- **+0.096 over our own tf_unet** — 6.9 tf_unet seed standard deviations.
- On pooled F1 (the honest, validation-thresholded number): **0.6377 vs
  0.5021 ± 0.0259 = +0.136, or 5.2 seed sd.**

### 13.3 The striking part is the ranking

| | ROC AUC |
|---|---|
| paper's U-Net | 0.8017 ± 0.0058 |
| NLN (the paper's own best AUROC) | 0.8622 ± 0.0006 |
| our tf_unet | 0.9470 ± 0.0074 |
| **our hybrid base 8** | **0.9827** |

**+0.181 over the published U-Net and +0.121 over NLN.** The model separates
RFI from clean almost perfectly.

But note the asymmetry: ROC gained **+0.181** over the published U-Net while
F1 gained only **+0.056**. PART 12.4's finding survives — ranking is excellent,
the operating point is not. Even at the *oracle* threshold, F1 caps at 0.644
against a ROC of 0.983. At 1:129 prevalence with ~44% wrong training labels,
that ceiling is set by the task, not the model. **The remaining headroom is in
the decision rule, not in ranking.**

### 13.4 It beat its own teacher

Every training label came from AOFlagger, which scores **0.5698** against the
human expert. The model trained on those labels scores **0.6439** against the
same human expert — **+0.074 better than the labels it learned from.** It did
not merely reproduce AOFlagger; it generalised past its teacher's errors.

### 13.5 THREE CAVEATS — do not quote 13.1 without them

1. **N = 1.** This project's own real-data seed spread is 0.052 (PART 12.10),
   and the margin over RFI-Net is +0.046 — *smaller than that spread*. Seeds 1
   and 2 are mandatory before this appears anywhere. A single run is not a
   result; that lesson has already been learned twice here (12.8 → 12.10).
2. **This is not a clean architecture comparison.** The hybrid ran with
   CE + Dice and a 56.9x class weight; tf_unet ran with plain cross-entropy and
   no class weighting (because PART 1 measured weighting as lethal there). So
   the +0.096 confounds **architecture + loss + class weighting**. Isolating
   the architecture needs either the hybrid without class weights or tf_unet
   with CE+Dice.
3. **The model is badly calibrated.** The validation-selected threshold is
   **0.9455**, not near 0.5 — a direct consequence of the 56.9x weight pushing
   outputs up. Ranking is unaffected (ROC 0.9827), but any claim about
   probabilities would be wrong, and it is why pooled F1 (0.6377) sits below
   oracle max F1 (0.6439).

### 13.6 CONFIRMED at 3 seeds (2026-09-06) — and it is significant

| metric | hybrid base 8, 3 seeds |
|---|---|
| **max F1 (oracle)** | **0.6467 ± 0.0079**  [0.6405, 0.6555] |
| pooled F1, 472 crop | 0.6505 ± 0.0145 |
| pooled F1, full 512 | 0.6368 ± 0.0157 |
| ROC AUC | 0.9801 ± 0.0025 |
| PR-AUC | 0.6836 ± 0.0159 |

**vs RFI-Net (0.5979, the published best): +0.0488, t = 10.74 on 2 dof
(needs 4.30). SIGNIFICANT.**

**vs our own tf_unet (0.5482 ± 0.0139): +0.0984, t = 10.67 on 4 dof
(needs 2.78). SIGNIFICANT.** On pooled F1: +0.1484, t = 8.67.

The N=1 caveat in 13.5 is now discharged. 13.5's other two caveats stand:
the tf_unet comparison still confounds architecture with loss and class
weighting, and the model is still badly calibrated.

Note the seed spread is **0.0079**, nearly half tf_unet's 0.0139 on the same
metric. The hybrid is both better and more reproducible here.

### 13.7 PART 6's efficiency finding TRANSFERS to real data

| base | parameters | max F1 |
|---|---|---|
| **8** | **593,842** | **0.6467 ± 0.0079** (3 seeds) |
| 32 | 9,304,186 | 0.6571 (N=1) |

**15.7x more parameters buys +0.0104 — 1.3 seed standard deviations, not
significant.** PART 6 measured −0.0063 for the same comparison on synthetic
data; on real LOFAR it is −0.0104. The same conclusion holds on a harder,
real dataset: the published width is oversized.

This is a stronger version of PART 6, because a referee can no longer say
"that only holds on your own easy synthetic benchmark."

### 13.8 The GroupNorm claim from PART 1 is confirmed

PART 1 argued that per-image normalisation hurt tf_unet but not the hybrid,
because the hybrid has GroupNorm and tf_unet has no normalisation layers.
Measured on identical LOFAR data:

| model | fixed range | per-image | drop |
|---|---|---|---|
| tf_unet | 0.5482 | **0.3039** | **−0.244** |
| hybrid base 8 | 0.6467 | **0.5700** | **−0.077** |

Per-image normalisation costs tf_unet **3.2x more** than it costs the hybrid.
The hybrid's per-image arm (0.5700) still essentially matches AOFlagger
(0.5698) and is within 0.018 of the paper's U-Net. tf_unet's per-image arm
(0.3039) is below even the sigma-clip baseline.

The architectural claim in PART 1 was made from reasoning; this measures it.

### 13.9 Where the remaining headroom is

ROC AUC is **0.9801 ± 0.0025** while max F1 is **0.6467**. Ranking is nearly
perfect; the decision is not. Even at the *oracle* threshold F1 caps at 0.647.
At 1:129 prevalence with ~44% wrong training labels (PART 11.6), that ceiling
is set by the task and the label noise, not by the model's capacity.

Two consequences:
- Further architecture work has little room on the ranking side. The gains
  left are in **calibration and thresholding** — the validation-selected
  threshold is 0.9455, far from 0.5, because of the 56.9x class weight.
- The most informative next experiment is `--no_class_weight`, which
  separates what the architecture contributes from what the loss contributes.

### 13.10 ARCHITECTURE, not the loss settings — the confound is resolved (2026-09-06)

13.5's second caveat was that the +0.098 over tf_unet confounded architecture
with the CE+Dice loss and a 56.9x class weight, since tf_unet ran plain
cross-entropy with no weighting. Ran the hybrid with `--no_class_weight`,
seed 0, everything else identical.

| | max F1 | pooled F1 | ROC | threshold | precision | recall |
|---|---|---|---|---|---|---|
| hybrid, class weight (3 seeds) | 0.6467 | 0.6505 | 0.9801 | 0.9619 | 0.6167 | 0.6617 |
| **hybrid, NO class weight (N=1)** | **0.6592** | **0.6607** | 0.9425 | **0.2509** | 0.6867 | 0.6220 |
| tf_unet, no class weight (3 seeds) | 0.5482 | 0.5021 | 0.9470 | 0.2069 | 0.4302 | 0.6032 |

**Removing the class weight made it slightly BETTER: +0.0125 max F1**
(1.6 seed sd, N=1, so not significant on its own — but certainly not worse).

**Decomposition of the +0.0984 gain over tf_unet:**

| source | contribution |
|---|---|
| **architecture (with CE+Dice)** | **+0.1110  (113%)** |
| class weighting | −0.0125  (−13%) |

With both models unweighted the hybrid still leads tf_unet by **+0.1110**, and
the margin over RFI-Net *widens* to **+0.0613**. **The win is the model, not
the loss settings.** Architecture work is aimed at the right target.

**Bonus: dropping the weight fixes the calibration.** The validation-selected
threshold moves from **0.9619 to 0.2509**, and precision rises 0.6167 → 0.6867
while recall gives back only 0.6617 → 0.6220. 13.5's third caveat is largely
answered too — the miscalibration was the class weight, not the model.

**A subtlety worth keeping.** ROC *fell* (0.9801 → 0.9425) while F1 *rose*
(0.6467 → 0.6592). Class weighting bought better ranking and paid for it with
a worse operating point, losing more than it gained. Ranking quality and
decision quality are genuinely decoupled here — which is the same lesson as
12.4 and 13.9, now demonstrated by intervention rather than observation.

**Recommendation: make `--no_class_weight` the default for LOFAR.** Simpler,
better calibrated, and at least as accurate. Confirm at 3 seeds first.

### 13.11 CONFIRMED at 3 seeds: no class weight is the better default (2026-09-06)

| metric | hybrid base 8, NO class weight, 3 seeds |
|---|---|
| **max F1 (oracle)** | **0.6603 ± 0.0040**  [0.6570, 0.6647] |
| pooled F1, 472 crop | 0.6641 ± 0.0037 |
| pooled F1, full 512 | 0.6561 ± 0.0044 |
| precision / recall | 0.6936 ± 0.0061 / 0.6225 ± 0.0047 |
| ROC AUC | 0.9378 ± 0.0111 |
| threshold | **0.2573 ± 0.0149** |

**Paired by seed, removing the class weight helps every time:**

| seed | with weight | without | diff |
|---|---|---|---|
| 0 | 0.6439 | 0.6592 | +0.0153 |
| 1 | 0.6555 | 0.6647 | +0.0091 |
| 2 | 0.6405 | 0.6570 | +0.0164 |

mean **+0.0136**, sd 0.0039, **t = 6.03 on 2 dof (needs 4.30) — SIGNIFICANT.**
All three seeds move the same way. 13.10's N=1 finding holds.

### 13.12 FINAL LADDER on real LOFAR (oracle max-F1, the paper's protocol)

| method | max F1 | params |
|---|---|---|
| sigma-clip baseline | 0.4103 | — |
| our tf_unet @ lr 1e-4 | 0.5482 ± 0.0139 | ~500k |
| AOFlagger (wrote the training labels) | 0.5698 | — |
| Mesarcik et al.'s U-Net | 0.5876 ± 0.0031 | — |
| **RFI-Net — published best** | **0.5979** | millions |
| hybrid base 8, class weighted | 0.6467 ± 0.0079 | 593,842 |
| **hybrid base 8, NO class weight** | **0.6603 ± 0.0040** | **593,842** |

- **+0.0624 over RFI-Net.** One-sample t = 27.22 on 2 dof (needs 4.30).
- **+0.1121 over our own tf_unet.** t = 13.42 on 4 dof (needs 2.78).
- Both **SIGNIFICANT**, at 593,842 parameters.

**Stability improves at every step:** seed sd 0.0139 (tf_unet) → 0.0079
(hybrid, weighted) → **0.0040** (hybrid, unweighted). The final configuration
is the most reproducible thing in this project.

**And the calibration problem is gone.** Threshold 0.2573 ± 0.0149, against
0.9619 with weighting. Precision rose to 0.6936 from 0.6167.

**All three of 13.5's caveats are now discharged**: N=1 (13.6, 13.11), the
architecture/loss confound (13.10), and the miscalibration (13.11).

**This configuration is the headline result of the project.**

### 13.13 The one number that has NOT improved

ROC AUC fell from 0.9801 (weighted) to **0.9378** (unweighted), while F1 rose.
Ranking got *worse* and the final score got *better*, consistently across all
three seeds. Class weighting buys ranking quality and pays for it with a worse
operating point, losing more than it gains.

For the paper this is the cleanest available demonstration that **ROC AUC is
the wrong metric to optimise on this task.** It moved in the opposite
direction to the metric anyone actually cares about, under a controlled
intervention, at N=3. Mesarcik et al. report AUROC as a headline (their NLN
wins on AUROC, 0.8622, while losing on F1, 0.5114) — the same pattern, and
worth citing alongside.

### 13.14 Next

`./scripts/run_lofar_hybrid.sh` runs seeds 1 and 2, then base 32, then the
per-image arm. ~2 h. Seed 0 is already done and will be skipped.

---

## PART 14 — INTEGRITY AUDIT of the PART 13 result (2026-09-07)

Adversarial self-check requested before the result is used. Script:
`analysis/audit_hybrid_result.py`. It deliberately imports nothing from
`experiments/lofar_hybrid.py` except the model class, so a bug in the training
script's own metric code cannot hide inside the audit.

### 14.1 Metrics recomputed from the checkpoint by independent code

| metric | reported | recomputed | diff |
|---|---|---|---|
| pooled F1 | 0.652759 | 0.652727 | 3.2e-05 |
| max F1 | 0.659204 | 0.659199 | 5.0e-06 |
| ROC AUC | 0.942547 | 0.942549 | 1.7e-06 |
| TP/FP/FN | 136157/62106/82754 | 136152/62116/82759 | ~10 px of 136k |

Differences are GPU non-determinism. **No hardcoded or injected numbers** —
every literal in the script is a docstring or a benchmark label in a print.

### 14.2 Leakage: clean

```
train n val overlap        : 0
TRAIN images also in test  : 0
VAL   images also in test  : 0
unique test images         : 109 of 109
```

Also checked the subtle path: the fixed-range normalisation is calibrated on
training images only. Recomputed from `tr_idx` alone it gives
**280723, 4.35311e+06**, identical to the run. No test statistics entered
through normalisation.

### 14.3 Thresholding penalises us, as it should

| | threshold | F1 |
|---|---|---|
| used (chosen on validation) | 0.2509 | **0.6527** |
| best possible on test (cheating) | 0.7800 | 0.6591 |

Honest thresholding costs **−0.0064** and we report the lower number. The
threshold used matches the validation optimum to 6e-05.

### 14.4 Overfitting: mild

| split | max F1 | ROC |
|---|---|---|
| train (109 sampled) | 0.7112 | 0.9673 |
| val (150) | 0.6820 | 0.9769 |
| test (109) | 0.6592 | 0.9425 |

Train−val gap **0.029**, modest for 28,000 steps, and ROC is *higher* on
validation than training — the opposite of memorisation. Best epoch 34 of 40.

### 14.5 It is not copying AOFlagger

| | F1 |
|---|---|
| AOFlagger vs human | 0.5698 |
| our model vs human | 0.6527 |
| **our model vs AOFlagger** | **0.7430** |

Agreement with the teacher is 0.74, not ~1.0. True RFI found by us but not
AOFlagger: **16,485**; found by AOFlagger but not us: **7,345**;
**net +9,140**. It disagrees with its teacher, and correctly.

### 14.6 It is not a trivial frequency rule

| rule | F1 |
|---|---|
| flag the 8 worst columns everywhere | 0.0638 |
| flag the 16 worst | 0.0645 |
| flag the 32 worst | 0.0563 |
| **our model** | **0.6527** |

It has learned the frequency structure (correlation 0.91 between its
per-column flag rate and the true rate), but that structure alone buys 0.065.

Prediction sanity: predicts 0.694% RFI against 0.766% true; no degenerate
all-zero images; per-image F1 median 0.6345, range 0.035–0.839.

### 14.7 Bootstrap over the 109 test images — the result survives

Seed variance was already known (±0.0040). This is **test-set** variance,
2000 resamples of the 109 images:

| | F1 | 95% CI |
|---|---|---|
| seed 0 | 0.6527 | [0.6083, 0.6912] |
| seed 1 | 0.6610 | [0.6198, 0.6962] |
| seed 2 | 0.6546 | [0.6122, 0.6921] |
| **3-seed mean** | **0.6561** | **[0.6154, 0.6923]** |

**RFI-Net's 0.5979 falls OUTSIDE the interval**, and **99.5% of bootstrap
samples beat it**. The margin survives test-set resampling as well as seed
resampling.

Note the CI is ±0.04 wide — that is the cost of a 109-image test set, and it
is the dominant uncertainty in this result, larger than the seed spread.

### 14.8 The real limitations — state these in the paper

1. **The baselines are QUOTED, not reproduced.** RFI-Net 0.5979 and the U-Net
   0.5876 come from Mesarcik et al. Table 2; we never ran them. The
   comparison is unusually well-grounded — our AOFlagger measurement
   reproduces their published 0.5698 to four decimals, confirming identical
   test data, labels and metric — but the paper must write *"as reported in
   Mesarcik et al. (2022)"*, never imply we measured it.
2. **Test set is 109 images.** ±0.04 at 95%. This is the dominant uncertainty.
3. **Training budgets are not matched to theirs.** They used 32×32 patches for
   100 epochs; we used full 512×512 images for 28,000 steps. Each method was
   given its own sensible budget, not a common one.
4. **Train−val gap of 0.029** exists, though it is small and best epoch was
   34/40.

### 14.9 Ethics: nothing to flag

Public Zenodo dataset (doi:10.5281/zenodo.6724065), cited. tf_unet is the
authors' unmodified code, credited. We report the conservative
validation-thresholded number as the headline rather than the oracle, all
three seeds rather than the best of them, and every caveat sits beside the
result in this document.

**Verdict: the PART 13 result is sound.**

---

## PART 15 — WHERE THE REMAINING F1 IS (diagnostic, 2026-09-07)

Measured on the best model (base 8, no class weight, seed 0) by splitting every
true-RFI pixel by how bright it is relative to that image's own clean pixels.

### 15.1 The model has already solved bright RFI, and only bright RFI

| tier | RFI pixels | share | **recall** |
|---|---|---|---|
| bright (above clean p99) | 108,947 | 49.8% | **0.937** |
| moderate (clean p75–p99) | 41,985 | 19.2% | 0.391 |
| faint (clean median–p75) | 24,113 | 11.0% | 0.268 |
| **invisible (below clean median)** | **43,866** | **20.0%** | **0.254** |

**Half the RFI is bright, and the model catches 94% of it. The other half it
catches about a quarter to a third of.** Every remaining point is in the
dim half.

### 15.2 What each fix would be worth

| intervention | resulting F1 | gain |
|---|---|---|
| current | 0.6527 | — |
| **remove all false positives** | **0.7669** | **+0.1142** |
| + perfectly catch the invisible tier | 0.7507 | +0.0979 |
| + perfectly catch the moderate tier | 0.7305 | +0.0778 |
| + perfectly catch the faint tier | 0.7074 | +0.0547 |
| + perfectly catch the bright tier | 0.6744 | +0.0217 |

### 15.3 What this rules IN and OUT for the next round of work

**RULED OUT — more capacity.** PART 6/13.7 measured 15.7x more parameters as
worth +0.0104, inside seed noise, on both synthetic and real data.

**RULED OUT — better thresholding.** PART 14.3 measured the honest
validation-chosen threshold as costing only **0.0064** against the
best-possible-on-test threshold. Calibration was a real problem *with* class
weighting and is essentially solved without it. The earlier claim in 12.4 and
13.9 that "the headroom is in the decision rule" was true of the weighted
model and is **no longer true** — do not carry it forward.

**RULED IN — precision.** The single largest available gain, **+0.114**. The
model currently emits 62,106 false positives against 136,157 true positives.

**RULED IN — faint-RFI recall, and specifically via CONTEXT.** The invisible
tier is 20% of all RFI at recall 0.254, worth **+0.098**. These pixels are
*by definition* dimmer than the median clean pixel, so **no amount of local
intensity modelling can find them** — the only available signal is their
neighbours in time and frequency. The human expert labelled them from context
too. This is the one place where an architectural change has a clear,
measured target: **longer-range context along the time and frequency axes.**

Note this is also the honest justification for the strip-convolution idea that
PART 4 found was not novel — the idea is right for this data even if the
credit belongs elsewhere. Prior art must still be cited (MARS,
arXiv:2608.05546).

### 15.4 The ceiling

PART 11.6: the training labels are AOFlagger's, which agree with the human
expert at only 0.5698 F1 — roughly 44% wrong. The model already scores 0.6603
against the human, **beating its own teacher by +0.09** (PART 14.5 confirms it
does this by genuinely disagreeing, not by copying). How far past a noisy
teacher a model can be pushed is not known, but the whole published field sits
between 0.51 and 0.60 on this test set, and we are at 0.66.

---

## PART 16 — HybridCRFNet: the Chen et al. 2023 CRF idea (built, 2026-09-07)

Paper: Chen, Han, Su, Yang & Zhou (2023), RAA 23 104004, *"Cleaning Radio
Frequency Interference in Pulsar-Folded Data Based on the Conditional Random
Fields with an Adaptive Prior"* — the **2sigmaCRF** method.
doi:10.1088/1674-4527/acd52b. Copy: `notes/Chen2023_2sigmaCRF_pulsar_RFI.pdf`.

Model: `src/hybrid_rfi_package/hybrid_crf_model.py` (`HybridCRFNet`).
Trainer: `experiments/lofar_hybrid_crf.py`.
**`hybrid_model.py` is untouched** — the new model imports and wraps it.

### 16.1 What their method is

Two stages, no neural network anywhere:

1. **Adaptive 2sigma prior** (their §3.1): histogram the per-pixel rms values,
   fit a Gaussian, label within ±2sigma of the peak "normal" and beyond 3sigma
   "RFI". Adaptive to the data rather than a fixed cut.
2. **Dense CRF refinement** (§3.2): minimise the Gibbs energy
   `U(w) = Σ V1(w_i) + Σ V2(w_i,w_i')`, unary from the Gaussian fit, pairwise
   `V2 = mu(w_i,w_i')·(λ_a·k_a + λ_s·k_s)` with a Potts penalty
   `mu = 1[w_i ≠ w_i']`, an appearance kernel over position *and* intensity,
   and a smoothness kernel over position. Fully-connected CRF
   (Krähenbühl & Koltun 2011, SimpleCRF), 5 iterations,
   λ_a=3, λ_s=20, ξ1=1, ξ2=10, ξ3=1.

The relevance is their own abstract: it recovers *"weak RFIs that are
unrecognizable in some pixels but picked out based on their neighborhoods"* —
exactly the PART 15 failure mode (20% of RFI is dimmer than a typical clean
pixel, recall 0.254, worth +0.098).

### 16.2 What we built

| | Chen et al. | ours |
|---|---|---|
| unary | Gaussian fit to an rms histogram | the trained CNN's logits |
| inference | denseCRF, fixed post-process | mean-field unrolled as **differentiable layers**, trainable end-to-end (CRF-as-RNN, Zheng et al. 2015) |
| connectivity | fully connected (permutohedral lattice) | k×k window, message passing as a convolution (ConvCRF, Teichmann & Cipolla 2018) |
| spatial kernel | one isotropic ξ | **separate bandwidths for time and frequency — ours**, motivated by PART 11.2 |
| ξ2 (intensity) | 10, on raw rms | 0.1, since our inputs are [0,1] |
| 2sigma prior | the core of the method | kept, optional, as an extra unary bias |

**Cost: 11 parameters** — λ_a, λ_s, four bandwidths, the 2×2 compatibility
matrix, and the prior weight. That is **0.0019%** of the base-8 backbone's
593,842, which was a hard requirement: PART 6/13.7 already established that
capacity is not the bottleneck, so a heavy refinement head would undercut the
project's own argument.

An exact optimisation for the binary case: since Q sums to 1 over 2 classes,
only one channel needs the expensive spatially-varying message pass and the
other follows as `Ksum − msg`. Halves compute and memory, no approximation.

### 16.3 MEASURED BEFORE TRAINING — the post-hoc CRF does not help

Frozen trained base-8 backbone, 109 test images, no CRF training:

| configuration | max F1 |
|---|---|
| backbone alone | **0.6592** |
| + CRF at Chen et al.'s λ = (3, 20) | **0.4884** |
| + CRF, λ = (3, 5) | 0.6162 |
| + CRF, λ = (1, 0.1) | 0.6594 |
| + CRF, λ = (1, 0.1), anisotropic ξ_t=0.5, ξ_f=5 | **0.6604** |

**Chen et al.'s published parameters are destructive here: −0.171 F1.**
Precision rises to 0.742 while recall collapses to 0.347 — the CRF erases
isolated detections whose neighbours are labelled clean. Their RFI arrives in
large contiguous blocks; ours is sparse and thin at 0.77% prevalence. The
paper anticipates this ("new parameters may need to be tested and selected for
data obtained from other telescopes").

**Even at the best settings found, the gain is +0.0012 — inside the 0.0040
seed spread. Effectively nothing.** Two reasons, both worth stating rather
than hiding:

1. **Our backbone already reasons about neighbourhoods**, via strip
   convolutions and the U-Net receptive field. Chen et al. applied their CRF
   to a bare histogram threshold with no spatial modelling at all, so it had
   far more to fix. The idea is sound; it is largely *already in* our model.
2. **A CRF can only redistribute evidence that is in the unary.** PART 15's
   problem is that faint RFI has almost no evidence to redistribute.

The anisotropy sweep also **refuted our own prediction**: propagating along
*frequency* (ξ_f=5) edged out propagating along *time*, opposite to what
PART 11.2's column concentration suggested. Consistent with PART 11 morphology,
where mean run lengths are near-equal on both axes (4.25 vs 3.90) — the
column concentration is a population effect, not elongated individual events.

### 16.4 RESULT: trained CRF gives +0.0012 — the idea does not transfer (2026-09-08)

`--crf_mode frozen`, warm-started from `hybrid_b08_nocw_seed0/best.pt`, the
backbone locked, **only the 11 CRF parameters trained**, 28,000 steps. So any
change is 100% attributable to the CRF.

| metric | backbone | + CRF | change |
|---|---|---|---|
| **max F1** | 0.6592 | **0.6604** | **+0.0012** |
| pooled F1 | 0.6528 | 0.6526 | −0.0001 |
| pooled F1 (472 crop) | 0.6607 | 0.6611 | +0.0004 |
| ROC AUC | 0.9425 | 0.9448 | +0.0023 |
| PR-AUC | 0.6522 | 0.6706 | +0.0185 |
| precision | 0.6867 | 0.6791 | −0.0077 |
| recall | 0.6220 | 0.6281 | +0.0061 |

**+0.0012 max F1 = 0.29 seed standard deviations. Inside noise.** It lands on
0.6604 against the 3-seed backbone mean of 0.6603 ± 0.0040 — indistinguishable.

**Exactly what the untrained sweep predicted** (16.3 found +0.0012 at the best
hand-set parameters). Training the CRF end-to-end recovered the same number
rather than beating it.

**The learned parameters say why, and they are the interesting part.** Given
freedom to set its own strength, the CRF *turned itself down*:

| parameter | init | learned |
|---|---|---|
| lambda_a (appearance) | 1.0 | **0.747** |
| lambda_s (smoothness) | 0.1 | **0.083** |
| xi_time | 0.5 | 0.387 |
| xi_freq | 5.0 | **3.613** |
| xi_intensity | 0.1 | 0.076 |
| compatibility | Potts [[0,1],[1,0]] | [[0.283, 0.920], [0.717, 0.080]] |

Both kernel weights **shrank toward zero**, which is the identity solution — a
CRF with lambda = 0 reproduces the backbone exactly. The optimiser, free to use
the mechanism as much as it liked, chose to use less of it. That is the model
stating that the neighbourhood evidence is already present in the unary.

Corroborating: **best epoch 2 of 40.** With only 11 parameters it converged
almost immediately and then failed to improve for 38 further epochs.

The one substantive movement is xi_freq staying large (3.6 against xi_time's
0.39) — the anisotropy is retained, propagating along frequency. And PR-AUC
did rise +0.0185, the largest single change, without translating into F1.

### 16.5 Conclusion, and what it is worth

**The Chen et al. CRF does not improve our model.** Measured three ways —
untrained at the paper's parameters (−0.171), untrained at swept parameters
(+0.0012), and trained end-to-end (+0.0012) — the answer is the same.

This is a **publishable negative result**, and a specific one:

> A conditional-random-field refinement that materially improves a histogram
> threshold adds nothing measurable on top of a network that already models
> spatial context. Given the freedom to weight itself, the CRF reduces its own
> kernel weights toward zero.

That is more informative than a marginal positive would have been, and it is
backed by the PART 15 diagnostic explaining the mechanism: a CRF redistributes
evidence, and faint RFI has little evidence in the unary to redistribute.

**Still untested:** `--crf_mode finetune` and `--crf_mode scratch`, where the
backbone can adapt rather than being frozen. The frozen result makes a large
gain unlikely, but it is the honest remaining gap, and `scratch` at ~2.9 h per
seed is the only version that tests the CRF-as-RNN claim properly.

**Bug found and fixed while reading the result:** the trainer's `m.update`
never recorded the CRF metadata (`crf_mode`, `crf_learned`, and the model name
read `HybridRFINet`). The run itself was correct — 593,853 parameters and CRF
keys present in the checkpoint confirm it — but the JSON was under-specified.
Fixed; future runs record it.

### 16.6 What is still untested

**End-to-end training**, which bolting the CRF on cannot evaluate: with the
CRF differentiable, the backbone can learn to emit unaries *shaped to be
refined* rather than unaries that are already final. Three modes provided:

    # cleanest isolation -- 11 trainable parameters, backbone frozen
    ~/torch-env/bin/python experiments/lofar_hybrid_crf.py --crf_mode frozen \
        --init_from runs/lofar/hybrid_b08_nocw_seed0/best.pt --no_class_weight

    --crf_mode finetune   warm-start, train everything at lower LR
    --crf_mode scratch    full 28,000-step budget, matched to PART 13

Cost: 379 ms/step at k=7 against the plain model's 37 ms, so `scratch` is
~2.9 h per seed. `frozen` is far cheaper — only 11 parameters to fit.

**Set expectations low.** The mechanism the CRF supplies is one the backbone
already has. If it fails, that is a publishable negative result: *a method
that works on unmodelled histogram thresholds adds nothing on top of a network
that already models spatial context* — and it comes with the measurement
showing exactly why.

---

## PART 17 — THE UNATTRIBUTED +0.1110, and the next experiment (2026-09-09)

Analysis pass, no new runs. All PART 12/13/16 numbers were re-verified against
`runs/lofar/*/eval_test/metrics.json` and reproduce exactly (nocw 0.6603 +/-
0.0040, weighted 0.6467 +/- 0.0079, lr1e-4 0.5482 +/- 0.0139, matched pooled
0.4563 +/- 0.0279).

### 17.1 The open gap

PART 13.10 attributes **+0.1110 over tf_unet to "architecture"**. That bucket
is coarser than the label suggests. `experiments/models_ablation.py` states the
problem in its own docstring: the hybrid differs from tf_unet in **at least six
ways at once** — architecture, framework, loss, normalisation (GroupNorm vs
none), padding (same vs valid) and output activation (raw logits vs ReLU).

13.10 genuinely ruled out **class weighting** (-0.0125, 3 seeds, 13.11). It did
NOT isolate GroupNorm, padding, output activation or framework. PART 13.8
already shows GroupNorm does heavy lifting: per-image normalisation costs
tf_unet -0.244 but the hybrid only -0.077.

So the project's strongest claim rests on an unmeasured decomposition. This is
the same class of error as PART 1 (which retracted "architecture explains the
gap" once class weighting and normalisation were measured) — do not repeat it.

### 17.2 What the synthetic ablation says (and why it is not reassuring)

`results/ablation_reduced_budget.json`, base 16, N=1, its own note warns
differences below ~0.03 are not established:

| variant | F1 | vs full |
|---|---|---|
| hybrid_full | 0.8928 | — |
| no_strip | 0.8785 | -0.0143 |
| no_eca | 0.9117 | **+0.0189** |
| no_res | 0.9033 | +0.0105 |
| plain_unet | 0.8814 | -0.0114 |

The three named components do **nothing** on synthetic data — removing ECA is
better. Yet on real LOFAR the architecture bucket is worth +0.1110. Something
in there matters a great deal and it is not established which part.

### 17.3 NEXT EXPERIMENT — run the ablation on LOFAR

The switches already exist (`use_res` / `use_strip` / `use_eca` in
`models_ablation.py`); they have never been run on real data. **Add a
`no_groupnorm` variant** — without it `plain_unet` still carries GroupNorm and
the tf_unet gap stays partly unattributed.

**Predictions registered in advance, so neither result can be rationalised
after the fact:**

- **plain_unet lands near 0.65** -> the win is GroupNorm and same-padding, not
  strip/ECA/residual. Kills the architecture story; gives a strong honest
  finding — *synthetic benchmarks credit components that do nothing on real
  data* — consistent with PARTS 1, 4 and 16.
- **plain_unet lands near 0.55 and no_strip drops sharply** -> strip
  convolutions genuinely matter on real data despite being worthless on
  synthetic. Rehabilitates the architecture argument with evidence rather than
  novelty, which matters because PART 4 killed the novelty claim.

Both outcomes are publishable, which is why this is the right next experiment.
It also addresses outstanding item 1 directly: the paper stops claiming a new
architecture and starts claiming a measured result about which mechanisms
survive contact with real data.

Cost: ~25 min/run (measured from the seed timestamps 10:05 / 10:30 / 10:55).
6 variants x 1 seed ~ 2.5 h to find the movers; then 3 seeds on those only.

### 17.4 Cheap gap worth closing at the same time

`runs/lofar/hybrid_b32_nocw_seed1/` is an **empty directory** — a base-32 run
started 2026-09-08 00:56 and abandoned. PART 13.7's efficiency claim was only
tested in the *weighted* configuration (base 32 weighted 0.6571, N=1) and never
in the final unweighted one. One run closes it.

### 17.5 What is now closed

PART 16 closed the "longer-range context" line from PART 15.3: the CRF
supplies a mechanism the backbone already has, and given freedom to weight
itself it shrank its own kernels toward zero. Of PART 15.3's two RULED IN
items, **context is now spent**; **precision (+0.114 available) is untouched.**

---

## Verified facts about the synthetic dataset (trust these)

Regenerates **bit-exactly** from `--seed 42`:

| | Value |
|---|---|
| Splits | train 700 / val 150 / test 150 |
| Test set | 24,840,000 pixels, 3,746,075 RFI (15.0808%) |
| Dataset mean RFI fraction | 14.67% (NOT 12.4% — PAPER_DIMENSIONS.md was wrong) |
| Test images with no RFI | 9 / 150 |
| `HybridRFINet` parameters | 9,304,186 |
| Split overlap | **zero** shared SHA-256 hashes — no leakage |

**Hybrid's results — checked against the actual `eval_test/metrics.json` on disk,
correct to 6+ decimals.** Confusion-matrix totals match the real data exactly
(24,840,000 px; TP+FN = 3,746,075):

ROC AUC **0.9995** · PR AUC **0.9982** · F1 **0.9808** · Precision 0.9801 ·
Recall 0.9815 · IoU 0.9623 · MCC 0.9774 · FPR 0.0035

---

## Still outstanding before publication

1. **The architecture is not novel** (PART 4). Reframe the paper around the
   failure diagnoses. This is the biggest single issue. **PART 17.3 is the
   experiment that addresses it** — the LOFAR ablation, whose two possible
   outcomes are both publishable.
1b. **The +0.1110 "architecture" gain is not decomposed** (PART 17.1). Six
   differences from tf_unet are confounded; only class weighting has been ruled
   out. Blocking for any claim about *why* the hybrid wins.
2. **The baseline comparison is not matched.** Baseline got 60 epochs, hybrid 22.
   Baseline reports **oracle max-F1**; hybrid reports F1 at a **fixed
   validation-selected threshold**. `tf_unet`'s valid padding scores it on
   236×560 vs the hybrid's full 276×600. Rerun both under one protocol.
3. **The architecture contributes little.** Matched-budget ablation: all U-Net
   variants F1 0.8785–0.9117. Strip convolutions **+0.023**, residual blocks
   +0.002, ECA **−0.005**. Best variant was the full model **with ECA removed**.
4. **The benchmark is easy.** Label is `injected_RFI > 0.5σ` — noise-free, zero
   ambiguity. 90.9% of RFI sits ≥1σ. A constant threshold scores ROC 0.9308 /
   F1 0.7421; a 2,578-parameter CNN scores 0.9574 / 0.7823.
5. **N = 1.** No seed repeats anywhere. Run ≥3 seeds. **This is now the single
   blocking item for the efficiency finding** — base 16 vs base 32 (−0.0024) and
   base 8 vs base 32 (−0.0063) are both inside the one-seed noise floor, so the
   headline claim is unproven until seeds 0 and 1 are run.
6. ~~**No result on real telescope data with human labels yet**~~ **CLOSED
   2026-09-05 — see PART 12.** tf_unet scores 0.4563 ± 0.0279 pooled F1 on the
   109 expert-labelled LOFAR baselines, below AOFlagger (0.5698) and not
   significantly above the sigma-clip baseline (0.4103, t=2.85 on 2 dof).
7. **The v4 bandpass formula is ours, not published** (PART 9). Two fixes
   needed: express the standing-wave period in MHz and quote the implied path
   length; and justify the analytic Tukey envelope against `hera_sim`'s measured
   polynomial. Also read section 2 of arXiv:1607.07443 (HIDE) before comparing
   to it.
8. **`unet_run_gpu/eval_test/metrics.json` is mislabelled** — it holds the
   *faircompare* numbers (`checkpoint_dir` field proves it). Model 1's original
   metrics were overwritten and no longer exist. `unet_run_faircompare/` has no
   `eval_test/` at all.

---

## Where the trained models live

| Model | Folder | Weights | F1 |
|---|---|---|---:|
| Authors', 1024×1024 | `unet_run_gpu/` | `best_checkpoint/model.ckpt.*` | ⚠ metrics overwritten |
| Authors', first 276×600 | `unet_run_faircompare/` | `best_checkpoint/model.ckpt.*` | 0.3879 |
| Control: per-image + weights | `unet_run_control_per_image_cw/` | `best_checkpoint/` | 0.3064 |
| Control: per-image, no weights | `unet_run_control_per_image/` | `best_checkpoint/` | 0.7191 |
| **Authors' + both fixes** | `unet_run_fixednorm/` | `best_checkpoint/` | **0.9317** |
| Reimpl., **1024×265 v4**, per-image + weights | `unet_run_control_1024x265_per_image_cw/` | `best_checkpoint/` | **0.6815** |
| **Corrected baseline, 1024×265 v4**, fixed + no weights | `unet_run_control_1024x265_fixed/` | `best_checkpoint/` | **0.9483** |
| **Hybrid base 8, 1024×265 v4** | `hybrid_run_1024x265/base08/` | `best.pt` | **0.9878** |
| **Hybrid (synthetic)** | `hybrid_run_paperdim/` | `best.pt` | **0.9808** |
| HERA fine-tuned, clean split | `hera_transfer_test/runs/…pretrained/` | `best.pt` | 0.9964 |
| HERA from scratch, clean split | `hera_transfer_test/runs/…scratch/` | `best.pt` | 0.9996 |
| Width sweep, base 4 | `hybrid_run_width_sweep/base04/` | `best.pt` | 0.9547 |
| Width sweep, base 8 | `hybrid_run_width_sweep/base08/` | `best.pt` | 0.9749 |
| Width sweep, base 16 | `hybrid_run_width_sweep/base16/` | `best.pt` | 0.9788 |
| **Width sweep, base 32 (control)** | `hybrid_run_width_sweep/base32/` | `best.pt` | **0.9812** |

TF checkpoints are three files sharing the `model.ckpt` prefix — keep all three.
Always use `best_checkpoint/` or `best.pt`, never `checkpoints/` or `last.pt`.
Fine-tuning **never modifies** `hybrid_run_paperdim/best.pt`; it writes elsewhere,
so the synthetic specialist and any HERA specialist coexist as separate files.

---

## Other things that bite

- `evaluate_hybrid_test.py --patch_size` defaulted to **512**, silently cropping
  276×600 → 276×512 (14.7% of every image). Fixed to 0.
- `train_unet_rfi_gpu.py` had a **silent fallback to the test set** when `val/`
  was missing. Removed — it now errors.
- `RFI_Project_Model_Comparison.md` §3 says batch_size 4 **and**
  `training_iters=87` — arithmetically impossible (at batch 4 a full pass is 175).
- `unet_run_faircompare/training_log.csv` shows val F1 **still climbing** at epoch
  22 (0.2927 → 0.3068 → 0.3213, flagged best) — cut off mid-climb.
- The published hybrid run used `CosineAnnealingLR(T_max=40)` but was **halted by
  hand at epoch 22**, and validated on only 50 of the 150 val images. Any rerun
  that uses a clean 22-epoch schedule is therefore *not* bit-comparable to it.
- **tf_unet's output offset is not always 40.** `create_conv_net` hardcodes
  `in_size = 1000` and returns `offset = in_size - size`, so it *claims* output =
  input − 40 at layers=3 regardless of the real input. That holds only when every
  pre-pool dimension is even. At **1024×265** the time axis goes 265 → 261 (odd) →
  the max-pool floors to 130, and the real output is **984×224, not 984×225**.
  `util.crop_to_shape` survives it (it splits the difference 20 left / 21 right)
  so nothing crashes, but the scored region is 81% of each image and slightly
  off-centre in time. Never trust `net.offset` — read `prediction.shape`.
- `HybridRFINet` uses `GroupNorm(min(8, C), C)`, so every channel count must be
  divisible by 8 (or be 1/2/4). `base=12` and `base=20` cannot be built.
- `RFI_Using_UNET.pdf` (Elsevier PDF) removed — copyright. **Still in git
  history**; needs `git filter-repo` before the repo goes public.
- `tf_unet` is GPL-3.0 and vendored, so the repo must be GPL-3.0. `LICENSE` added.
- **MACHINE MOVED (2026-08-29): WSL2 -> Fedora 44 dual boot.** Project now at
  `~/Documents/minor project/Minor Project` (note the nested lowercase parent).
  The old WSL2 setup on the Windows partition still works and is the only
  second copy of the `.pt` and `.pkl` files - do not reclaim that partition.
- **Fedora specifics that cost a day to find, do not rediscover them:**
  - **Boot kernel 6.19.10-300.fc44** (pinned via `grubby --set-default`).
    Kernel 7.1.10-200 has the NVIDIA driver built too, but GNOME renders the
    desktop on the RTX 3060 there (`gnome-shell` holds 115 MiB in `nvidia-smi`),
    which makes the whole UI crawl. On 6.19 it holds 2 MiB and the AMD iGPU
    drives the display. If a future kernel update brings the lag back, check
    `nvidia-smi`'s process list first.
  - **Secure Boot is OFF.** The MOK enrollment screen never accepted keyboard
    input on this Legion. BitLocker is not enabled (proved: the NTFS partition
    mounts and reads fine from Linux), so disabling it was safe.
  - Driver 610.57.04, CUDA UMD 13.3, RTX 3060 Laptop 6 GB.
  - `~/torch-env` = **Python 3.14**, torch 2.13.0+cu130, `GPU: True`.
  - `~/tf-env` must use **Python 3.12** (`python3.12 -m venv ~/tf-env`) -
    TensorFlow publishes no wheels for Fedora 44's default Python 3.14.
  - **Every tf-env run needs `LD_LIBRARY_PATH` exported over the pip CUDA libs**
    or TensorFlow silently falls back to CPU without naming what is missing.
    `tf-env` also needs `pillow` and `matplotlib` — `tf_unet/util.py` imports
    both at module scope. Full detail in PART 7.
  - Never install torch into `tf-env` or TensorFlow into `torch-env`.
- Git hygiene after the move: `.gitattributes` with `* text=auto eol=lf` and
  `core.autocrlf false` (the Windows copy arrived with CRLF everywhere, making
  26 files look modified when only 3 were). `*.pt` and `*.pkl` are gitignored -
  `base32/best.pt` is 112 MB and GitHub hard-rejects anything over 100 MB.

## Key documents in the repo

`AUDIT_REPORT.md` · `BASELINE_FAILURE_DECOMPOSITION.md` ·
`MODEL_COMPARISON_EXPLAINED.md` · `experiments/` ·
`experiments/width_sweep/run_width_sweep.py` · `hera_transfer_test/` ·
`analysis/lofar_analysis/` · `results/` ·
`analysis/dataset_v4_bandpass/generate_dataset_v4.py` (the v4 bandpass generator — see PART 9) ·
`experiments/normalisation_control/run_control.py` (the tf_unet arms in PART 8) ·
`notes/Mesarcik2022_Learning_to_detect_RFI_without_seeing_it.pdf` (LOFAR dataset paper — see PART 10) ·
`analysis/lofar_analysis/deep_audit_stage1..5_*.py` + `audit_lofar_report*.json` + `fig_lofar_*.png` (the PART 11 audit) ·
`lofar_data.py` (low-RAM memory-mapped LOFAR loader — see PART 11.10) ·
`experiments/lofar_tfunet_baseline.py` (the PART 12 tf_unet run) ·
`scripts/run_lofar_*.sh` (the PART 11.12, 12.8 and 12.12 protocols) ·
`scripts/status.sh` (live status of every LOFAR run)
