# `experiments/` — the controls the comparison needs

These scripts exist because the headline claim in `RFI_Project_Model_Comparison.md`
— *"the performance gap is attributable to architectural limitations of the plain
U-Net rather than training configuration"* — is not supported by any experiment
currently in this repository.

Each script here answers one question a reviewer will ask. Run them before
submitting anything.

| Script | Question it answers |
|---|---|
| `dataset_difficulty.py` | How hard is this benchmark? What fraction of the labelled RFI is even visible above the noise? |
| `classical_baselines.py` | Do methods that use no learning at all already solve it? |
| `run_ablation.py` | Which of the hybrid's components actually contribute, holding everything else fixed? |
| `models_ablation.py` | Model definitions used by `run_ablation.py` (component switches + trivial references). |

All of them use the same protocol as `evaluate_hybrid_test.py`: any free
threshold is chosen on **validation** and applied **once** to test.

## Quick start

```bash
# 1. regenerate the dataset (~40 s, 1.1 GB)
python3 hybrid_rfi_package/dataset_generator_v3_strength.py \
    --n_images 1000 --n_freq 276 --n_time 600 --seed 42 \
    --output_dir "Synthetic Dataset 276x600"

# 2. characterise difficulty (seconds)
python3 experiments/dataset_difficulty.py --dataset_dir "Synthetic Dataset 276x600"

# 3. no-learning baselines (minutes)
python3 experiments/classical_baselines.py --dataset_dir "Synthetic Dataset 276x600"

# 4. matched-budget architecture ablation (hours on CPU, minutes on GPU)
python3 experiments/run_ablation.py --dataset_dir "Synthetic Dataset 276x600" \
    --out results/ablation.json
```

## Reference results (CPU, reduced budget)

Measured on the `seed=42`, 276×600 dataset. Raw output in `../results/`.

The ablation used a deliberately reduced budget (`base=16`, 400 training images,
200 gradient steps, batch 4) so all seven variants fit on 2 CPU cores.
**Absolute F1 is therefore below the 0.98 reported for the full-budget hybrid;
the point is the ordering and the gaps, measured under an identical budget for
every row.**

| Variant | Res | Strip | ECA | Params | ROC AUC | F1 | MCC |
|---|:---:|:---:|:---:|---:|---:|---:|---:|
| `logistic_pixel` | – | – | – | 4 | 0.3165 | 0.2621 | 0.0000 |
| `tiny_cnn` | – | – | – | 2,578 | 0.9574 | 0.7823 | 0.7493 |
| `plain_unet` | ✗ | ✗ | ✗ | 1,942,306 | 0.9838 | 0.8814 | 0.8614 |
| `no_strip` | ✓ | ✗ | ✓ | 2,029,386 | 0.9859 | 0.8785 | 0.8579 |
| `no_res` | ✗ | ✓ | ✓ | 2,255,418 | 0.9900 | 0.9033 | 0.8867 |
| `no_eca` | ✓ | ✓ | ✗ | 2,342,450 | **0.9919** | **0.9117** | **0.8963** |
| `hybrid_full` | ✓ | ✓ | ✓ | 2,342,474 | 0.9889 | 0.8928 | 0.8737 |

Marginal effect per component (mean F1 present − mean F1 absent):
**strip convolutions +0.0226**, residual blocks +0.0020, ECA **−0.0050**.

No-learning baselines on the same test set:

| Method | ROC AUC | PR AUC | F1 | MCC |
|---|---:|---:|---:|---:|
| Constant global threshold | 0.9308 | 0.8257 | 0.7421 | 0.7092 |
| Per-channel sigma clipping | 0.5787 | 0.2169 | 0.2788 | 0.0851 |
| SumThreshold-lite | 0.6432 | 0.3608 | 0.3460 | 0.2507 |
| *reported `tf_unet` baseline* | *0.6681* | *0.4350* | *0.3879* | – |

**One seed per cell.** Differences below ~0.03 F1 are not established by this
run. See `AUDIT_REPORT.md` §A2 for the full interpretation and caveats.

## Why a matched-budget ablation and not a full retrain

The hybrid's reported run is 1936 gradient steps at batch 8 on full 276×600
images with `base=32` — roughly 29 hours on 2 CPU cores per variant. Seven
variants is not feasible without a GPU. Reducing width and step count equally
for every variant keeps the comparison controlled, which is the property that
matters. Rerun with `--base 32 --steps 1936 --batch 8` on the RTX 3060 to get
the publication table.
