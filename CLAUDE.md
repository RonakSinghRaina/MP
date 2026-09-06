# Project Context: RFI Detection — Baseline vs. Hybrid U-Net

This file is read automatically by Claude Code at the start of every session
in this folder. It covers **environment, hardware and precautions**. All
results, datasets and model numbers live in `RFI-project-context.md`.

Rewritten 2026-09-06 for Fedora 44 and the reorganised repository. Every
number below was measured on this machine, not taken from a spec sheet.

---

## 0. RULES — these come before everything else

**0.1 — Read `RFI-project-context.md` before answering anything about this
project.** It is the authoritative record: 13 parts, every dataset audit,
every model result, and an explicit list of superseded claims. It is kept
current and committed. This file (`CLAUDE.md`) covers environment and
hardware only. Do not answer a question about results, datasets, parameters
or next steps without reading it first — several claims in the older parts
have been retracted, and answering from memory reproduces retracted claims.

**0.2 — Every number must come from a measurement made in this session or
recorded in `RFI-project-context.md`. Never estimate, never round from
memory, never invent.** If a number is not to hand, compute it — read the
`metrics.json`, run the script, measure the timing — and say where it came
from. If it genuinely cannot be computed, say so plainly rather than
producing a plausible figure.

This rule exists because it has already gone wrong three times here:
a preprocessing clip bound quoted from the wrong section of a paper
(PART 11.6b), a normalisation claim reasoned from global min/max instead of
measured (PART 11.8), and a learning-rate gain reported from a single seed
that halved when two more were run (PART 12.10). Each was caught only by
measuring. Guessing costs more time than checking.

**Corollary:** when reporting a result, give the spread as well as the mean.
The real-data seed spread here is 0.052, larger than most architectural
effects this project has ever claimed. A single run is not a result.

---

## 1. Hardware — measured, not spec-sheet

- **GPU:** NVIDIA RTX 3060 Laptop, **6144 MiB**, driver **610.57.04**.
- **Usable VRAM is close to the full 6 GB on Fedora.** *(An earlier version of
  this file said "~3.5 GB usable because the Windows compositor shares the
  card." That was a Windows fact and no longer applies.)* Measured at
  512×512:

  | model | batch | VRAM | ms/step |
  |---|---|---|---|
  | tf_unet layers=3 features_root=32 | 4 | 4.29 GB | 174 |
  | hybrid base 32 | 1 | 1.56 GB | 128 |
  | hybrid base 32 | 2 | 3.01 GB | 240 |
  | hybrid base 32 | 4 | OOM | — |
  | hybrid base 8 | 1 | 0.37 GB | 37 |
  | hybrid base 8 | 8 | 2.92 GB | 216 |

- **PLUG THE LAPTOP IN.** This is the single biggest performance factor here.
  On battery the SM clock is pinned at **210 MHz of 2100 MHz** with
  `SW Power Cap: Active`, making training **15.3× slower** — tf_unet measured
  2684 ms/step on battery against 174 ms/step on AC. Check before quoting any
  timing:

  ```bash
  cat /sys/class/power_supply/A*/online     # 1 = plugged in
  ```

  `./scripts/status.sh` warns about this, and the run scripts refuse to start
  on battery. Full detail in PART 11.10c / 11.13 of the context doc.

- **Only one training run fits on this card.** One run holds ~4.3 GB, so a
  second dies immediately with `RESOURCE_EXHAUSTED` while the first continues
  unaffected — which makes the traceback misleading. Check first:

  ```bash
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
  ```

## 2. OS and environment

- **Fedora 44**, kernel **6.19.10-300.fc44.x86_64**. Migrated from
  Windows 11 + WSL2 on 2026-08-29; see PART 7 of the context doc for the
  migration, including the GPU/Secure-Boot saga and the CRLF cleanup.
- **Boot kernel 6.19.10.** Kernel 7.1.10 renders the GNOME desktop on the
  RTX 3060 and makes the UI crawl. 6.19.10 is pinned as default.
- The old Windows partition can be mounted read-only if an old file is needed:
  `sudo mount -o ro /dev/nvme0n1p3 /mnt/win`.

## 3. Two Python virtual environments — do not merge them

| Env | Purpose | Python | Framework |
|---|---|---|---|
| `~/tf-env` | TensorFlow / the authors' `tf_unet` | **3.12.14** | TF **2.21.0** |
| `~/torch-env` | PyTorch / the hybrid, and all analysis scripts | **3.14.3** | torch **2.13.0+cu130** |

- **They were one environment once, and it broke.** `pip install torch`
  silently downgraded the cuDNN TensorFlow needed. The two require
  incompatible cuDNN versions and cannot share an environment.
- **tf-env must be Python 3.12.** Fedora 44's default Python 3.14 has no
  TensorFlow wheels — build it with `python3.12 -m venv ~/tf-env`.
- Scripts are invoked as `~/tf-env/bin/python ...` or
  `~/torch-env/bin/python ...` rather than through an activated shell, so
  anything that must be in the environment has to be in the command itself
  (see §4).

## 4. TensorFlow needs LD_LIBRARY_PATH set explicitly

TF 2.21 does not add its own pip CUDA libraries to the loader path on Fedora.
Without this it prints `Cannot dlopen some GPU libraries ... Skipping
registering GPU devices` — it never names the missing library — and silently
runs on CPU:

```bash
export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
```

Because scripts are run as `~/tf-env/bin/python` and not through `activate`,
this must go in the command line itself. **PyTorch does not need it.**

Also set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow imports (the
training scripts do this themselves).

## 5. VRAM preflight checks — a bug that recurred twice

A "does this fit" preflight was added to both the TensorFlow and PyTorch
scripts and **both had the same bug independently**: they probed a *square*
guessed size instead of the real image shape (e.g. `276×276` when the images
are `276×600`). The check passed and training then OOM'd. When writing any
memory preflight: **probe with the actual shape read from a real file, and
the true batch size, never 1.**

## 6. Do not use the paper's original learning rate at low batch size

Akeret et al. 2017 specify momentum, `lr=0.2`, batch 32. **This kills the
network at small batch size.** `tf_unet` applies ReLU directly to the logits;
once both go negative ReLU zeroes them, softmax outputs exactly `[0.5, 0.5]`,
and the gradient is exactly zero. Measured: dead at iteration 4 with lr=0.2,
batch 1. Both training scripts default to `adam, lr=1e-3`, with collapse
detection (loss ≈ ln 2 = 0.6931 and ROC ≈ 0.5000 is the signature).

**Nuance added later:** on the 1024×265 dataset the network *escaped* this
state after 15 epochs (PART 8), so the freeze is long, not always permanent.

**And on LOFAR the learning rate mattered more than any architecture change
measured in this project:** tf_unet at Adam 1e-4 for 150 epochs beat 1e-3 for
60 epochs by +0.046 pooled F1 on the 3-seed mean. That gain was NOT
statistically significant at N=3 (paired t=1.54, 2 dof) — see PART 12.10 —
but it did more than halve the seed spread. Do not assume a learning rate
carried over from the synthetic dataset is right for LOFAR.

## 7. tf_unet's "epoch" is not a full pass over the data

`tf_unet.Trainer.train(..., epochs=N, training_iters=M)` runs exactly `M`
gradient steps per "epoch", not one pass over the training set. Any fair
comparison must set `training_iters` deliberately.

**This matters more than it looks.** On LOFAR, matching the *epoch count* to
the synthetic baseline would have given the model 9.5× more training, because
LOFAR has 9.5× more images. The comparable quantity is **gradient steps**
(PART 11.10b). For patch training it is smaller still — a 64×64 patch at
batch 32 yields 18,432 output pixels per step against 891,136 for a full
512×512 image at batch 4, so there the budget must be matched by **output
pixels** (PART 12.12).

## 8. Evaluation must be batched carefully

Forward passes over many evaluation images at once can OOM even when training
fit, because evaluation does not get the same memory-saving treatment.
Evaluation code here processes 1–2 images at a time and concatenates before
computing metrics. Preserve that.

`tf_unet`'s `net.predict()` also reloads the checkpoint from disk on **every
call** (~2.5 s each). All evaluation code here restores once and calls
`net.predicter` directly — roughly 50× faster.

## 9. Checkpoints, model selection, and leakage

- Use `best_checkpoint/` (TF) or `best.pt` (PyTorch), never `checkpoints/` or
  `last.pt` — those are merely the latest epoch.
- **Architecture must match exactly to restore a checkpoint.** To inspect one:
  ```python
  import tensorflow.compat.v1 as tf1; tf1.disable_v2_behavior()
  for name, shape in tf1.train.list_variables('path/to/model.ckpt'): print(name, shape)
  ```
- **Select checkpoints on VALIDATION, never test, and with enough images.**
  An early run selected on 10 random patches; per-image RFI content varies
  0–60%, so taking the max across noisy checks produced a fake 0.81 that
  collapsed to 0.34 on the real test set — winner's curse. Scripts use 150
  validation images.
- **Choose the operating threshold on validation too.** Reporting an oracle
  max-F1 chosen on the test set is optimistic; on LOFAR the two differ by
  about 0.05. Both are reported by the LOFAR scripts, with `pooled_f1`
  (validation-selected) as the headline and `max_f1` labelled optimistic.
  Mesarcik et al. quote oracle max-F1, so cross-paper comparisons must use
  `max_f1` (PART 12.7).

## 10. Datasets — four of them now

| Path | What | Notes |
|---|---|---|
| `data/synthetic/Synthetic Dataset` | original, 1024×1024 | Model 1 only; a "before" reference, not a controlled comparison |
| `data/synthetic/Synthetic Dataset 276x600` | paper-matched dims | PARTS 1, 4, 6. RFI morphology was rescaled proportionally when generated |
| `data/synthetic/Synthetic Dataset 1024x265` | v4, with instrument bandpass | PARTS 8, 9 |
| `data/lofar/` | **real LOFAR**, 9.3 GB pickle + memmap arrays | PARTS 10–12. Human expert labels on 109 baselines |
| `data/hera/HERA_04-03-2022_all.pkl` | HERA transfer test | PARTS 2, 3 |

**Three LOFAR traps that silently corrupt results** (all guarded in code, but
know them):

1. **All 109 test images are byte-identical to 109 training images.** Train on
   the full 7500 and you test on images the model has seen. Use
   `d.clean_train_idx` (7356). Both LOFAR scripts verify this at startup and
   abort if violated.
2. **The metric must be a pooled pixel-wise F1**, not a mean of per-image F1s
   — they differ by 0.103.
3. **Never open the 9.3 GB pickle.** It exhausts RAM and crashes the IDE. Use
   `from lofar_data import load_lofar` — memory-mapped, 28.7 MB resident,
   0.007 s.

## 11. Command reference

```bash
# ---- LOFAR: tf_unet baseline (PART 12) --------------------------------------
export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
~/tf-env/bin/python experiments/lofar_tfunet_baseline.py          # 175x60 steps, fixed norm, no class weights
./scripts/run_lofar_baseline.sh                                   # 3 seeds + convergence control
./scripts/run_lofar_lr1e-4.sh                                     # the lr 1e-4 arms

# ---- LOFAR: the hybrid ------------------------------------------------------
~/torch-env/bin/python experiments/lofar_hybrid.py                # base 8 default
./scripts/run_lofar_hybrid.sh                                     # 3 seeds + base32 + per-image arms

# ---- status of any running work ---------------------------------------------
./scripts/status.sh          # or -w to refresh every 30 s

# ---- synthetic work ---------------------------------------------------------
~/tf-env/bin/python experiments/baseline_fixednorm.py --features_root 32 --epochs 60
~/torch-env/bin/python experiments/width_sweep/run_width_sweep.py --base 8 32 --seed 0 \
    --out_root runs/hybrid/hybrid_run_width_sweep_seed0
```

All training scripts are **resumable** — rerun the identical command and they
continue from `progress.json`. Do not change architecture flags between a
stopped run and its resume; that fails confusingly rather than cleanly.

## 12. Current status (2026-09-06)

**Read `RFI-project-context.md` for numbers. Summary only:**

- **The "architecture explains the gap" claim is dead.** An earlier version of
  this file said the hybrid's F1 0.98 vs baseline 0.39 was "attributed to
  architecture." PART 1 refuted that: class weighting accounted for 66% of the
  gap and normalisation 34%. Under matched conditions the gap fell from 0.59
  to about 0.05. **Do not repeat the old claim.**
- The strip-convolution idea is **not novel** (PART 4, MARS arXiv:2608.05546).
- The model is **oversized** (PART 6): base 8 at 593,842 params scores 0.9749
  against base 32's 9,304,186 params at 0.9812. base 4 genuinely breaks.
- **Real-data results now exist** (PART 12). tf_unet on the 109 expert-labelled
  LOFAR baselines: pooled F1 0.4563 ± 0.0279 at lr 1e-3, 0.5021 ± 0.0259 at
  lr 1e-4. The synthetic-to-real gap is **−0.44 F1** for identical code and
  budget — the strongest single result this project has.
- **N=1 is not safe.** The real-data seed spread is 0.052, larger than most
  architectural effects claimed here. Always run ≥3 seeds and report spread.
