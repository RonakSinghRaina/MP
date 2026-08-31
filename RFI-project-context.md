# RFI Project — shared context for any Claude chat in this project

Updated 2026-08-31. Body through PART 7 is the fourth revision (2026-08-27);
PART 8 added 2026-08-30, PART 9 added 2026-08-31. **Read this first.** It carries the findings
from a deep audit so any new chat, Cowork session, or Claude Code terminal
session starts with the same picture instead of re-deriving it.

> **If you read nothing else:**
> 1. The strip-convolution architecture is **no longer novel** — see PART 4
>    (MARS, arXiv:2608.05546, does the same thing at 34× fewer parameters).
> 2. The model is **oversized** — see PART 6. base 16 (2.3M params) scores
>    0.9788 against base 32's (9.3M) 0.9812. Roughly 8.7M parameters buy about
>    half a percent of F1.
> 3. The paper should be reframed around the failure diagnoses in PART 1 and the
>    efficiency finding in PART 6, not around the architecture.

---

## The project in one paragraph

Detecting radio frequency interference (RFI) in synthetic radio-telescope
spectrograms as pixel-wise segmentation. Two models: a reproduction of the U-Net
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

## PART 5 — real telescope data with human ground truth (LOFAR)

`L629174_RFI_dataset.pkl` is in the project root and is genuinely real LOFAR
observation data (not simulated). The larger `LOFAR_Full_RFI_dataset.pkl`
(~10 GB, not yet downloaded) carries **109 expert-labelled baselines** — actual
human ground-truth masks on real telescope data, which is what a reviewer will
want to see the model tested on.

`lofar_analysis/analyse_lofar.py` is **written but not yet run**. It needs
~8–10 GB RAM, so it must run in WSL (`~/torch-env`), not in a Cowork device
session (that VM has only 3.9 GB).

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

`dataset_v4_bandpass/generate_dataset_v4.py`, Section 2. One multiplicative gain
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
   failure diagnoses. This is the biggest single issue.
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
6. **No result on real telescope data with human labels yet** (PART 5). This is
   what a reviewer will ask for.
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
`lofar_analysis/` · `results/` ·
`dataset_v4_bandpass/generate_dataset_v4.py` (the v4 bandpass generator — see PART 9) ·
`experiments/normalisation_control/run_control.py` (the tf_unet arms in PART 8)
