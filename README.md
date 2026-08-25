# RFI Detection in Synthetic Radio Spectrograms — Baseline vs. Hybrid U-Net

Detecting radio frequency interference (RFI) in synthetic radio-telescope
spectrograms as a pixel-wise segmentation problem, comparing the U-Net of
Akeret et al. (2017) against a hybrid architecture (residual blocks +
multiscale anisotropic strip convolutions + efficient channel attention).

> ### ⚠ Read `AUDIT_REPORT.md` before citing any number from this repository.
>
> An independent audit (2026-08-22) reproduced the dataset bit-exactly and found
> that the headline comparison does not currently support its stated conclusion.
> In particular a **2,578-parameter, 3-layer CNN** and even a **constant global
> threshold** both outscore the reported `tf_unet` baseline on the same test set.
> Several documented configurations are also mutually inconsistent. The audit
> lists what must change before submission.

---

## Repository layout

```
.
├── AUDIT_REPORT.md              <- publication-readiness audit. Start here.
├── CLAUDE.md                    <- environment, hardware and known-failure notes
├── RFI_Project_Model_Comparison.md   <- results tables and file locations
│
├── hybrid_rfi_package/          <- the proposed model (PyTorch)
│   ├── hybrid_model.py                  architecture (9,304,186 parameters)
│   ├── train_hybrid.py                  training, validation-only model selection
│   ├── evaluate_hybrid_test.py          final test evaluation, run once
│   ├── make_val_split.py                carve val/ out of train/ (only if missing)
│   ├── dataset_generator_v3_strength.py dataset generator + RFI strength maps
│   └── PAPER_DIMENSIONS.md              why 276x600, and how to regenerate
│
├── unet_rfi_package copy/       <- the ACTIVE baseline package (see note below)
│   ├── tf_unet/                         Akeret et al.'s unmodified code (GPL-3.0)
│   ├── train_unet_rfi_gpu.py            training harness around tf_unet
│   ├── run_fair_comparison.py           matched-settings baseline run
│   └── evaluate_test_set.py             final test evaluation
│
├── unet_rfi_package/            <- SUPERSEDED earlier copy, kept for history
│
└── experiments/                 <- controls the comparison needs (added by audit)
    ├── dataset_difficulty.py            how hard is this benchmark?
    ├── classical_baselines.py           no-learning reference methods
    ├── models_ablation.py               component-switchable model variants
    └── run_ablation.py                  matched-budget architecture ablation
```

> **Which baseline folder is real?** `unet_rfi_package copy/` is. The scripts that
> produced every reported baseline number (`run_fair_comparison.py`,
> `evaluate_test_set.py`, `train_unet_rfi_gpu.py`) exist **only** there;
> `unet_rfi_package/` holds older, different scripts and a byte-identical second
> copy of `tf_unet`. The folder name is an artefact and should be renamed before
> release — see `AUDIT_REPORT.md` §C3.

## Data and results are not in this repository

`.gitignore` excludes the dataset and every training output. Nothing here lets a
reader verify a reported number. The dataset is fully regenerable (below); the
checkpoints, `training_log.csv`, `progress.json` and `eval_test/metrics.json`
are not, and **should be committed or archived** (Zenodo/OSF) for publication.

## Reproducing the dataset

The dataset is deterministic given the seed. Regenerating with `--seed 42`
reproduces the published statistics exactly (verified in the audit):

```bash
python3 hybrid_rfi_package/dataset_generator_v3_strength.py \
    --n_images 1000 --n_freq 276 --n_time 600 --seed 42 \
    --output_dir "Synthetic Dataset 276x600"
```

| Property                  | Value                                               |
| ------------------------- | --------------------------------------------------- |
| Splits                    | train 700 / val 150 / test 150                      |
| Image size                | 276 × 600 (frequency × time),`float32` `.npy` |
| Mean RFI pixel fraction   | 14.67% (test split: 15.0808%)                       |
| Test set totals           | 3,746,075 RFI of 24,840,000 pixels                  |
| Images with no RFI at all | 9 / 150 in test                                     |
| Size on disk              | ~1.1 GB                                             |

Each split also carries `metadata.jsonl` (per-image density, RFI types, strength
percentiles) and `strength/` maps giving the injected RFI amplitude in units of
the local noise sigma — needed for `--strength_report`.

## Environments

TensorFlow and PyTorch need **incompatible cuDNN versions here and must not
share a virtualenv** (see `CLAUDE.md` §3):

```bash
python3 -m venv ~/tf-env    && source ~/tf-env/bin/activate    && pip install -r requirements-tf.txt
python3 -m venv ~/torch-env && source ~/torch-env/bin/activate && pip install -r requirements-torch.txt
```

Create venvs on the Linux filesystem (`~/`), never on a `/mnt/c/...` mount.

## Running

```bash
# baseline (tf-env)
cd "unet_rfi_package copy"
python3 run_fair_comparison.py --batch_size 4 --features_root 32
python3 evaluate_test_set.py --checkpoint_dir "../unet_run_faircompare/best_checkpoint" \
    --dataset_dir "../Synthetic Dataset 276x600" --patch_size 0 --features_root 32

# hybrid (torch-env)
cd hybrid_rfi_package
python3 train_hybrid.py --dataset_dir "../Synthetic Dataset 276x600" \
    --output_dir "../hybrid_run_paperdim" --patch_size 0 --batch_size 8 \
    --n_val_images 150 --deterministic --early_stop_patience 3
python3 evaluate_hybrid_test.py --dataset_dir "../Synthetic Dataset 276x600" \
    --output_dir "../hybrid_run_paperdim" --patch_size 0 \
    --per_image_csv --strength_report

# audit controls (torch-env)
python3 experiments/dataset_difficulty.py  --dataset_dir "Synthetic Dataset 276x600"
python3 experiments/classical_baselines.py --dataset_dir "Synthetic Dataset 276x600"
python3 experiments/run_ablation.py        --dataset_dir "Synthetic Dataset 276x600"
```

`--patch_size 0` is required everywhere: any other value centre-crops the
276 × 600 images and produces numbers that do not match the reported ones.

## Licence

**GPL-3.0-or-later.** This repository redistributes `tf_unet`
(© Joel Akeret, GPL-3.0) under `unet_rfi_package*/tf_unet/`, so the combined work
is GPL-3.0. See `LICENSE`.

## Citing the original method

Akeret, J., Chang, C., Lucchi, A., & Refregier, A. (2017).
*Radio frequency interference mitigation using deep convolutional neural networks.*
Astronomy and Computing, 18, 35–39. https://doi.org/10.1016/j.ascom.2017.01.002

The publisher's PDF is **not** redistributed here; obtain it from the publisher
or from arXiv:1609.09077.
