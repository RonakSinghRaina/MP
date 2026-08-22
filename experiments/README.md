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

Measured on the `seed=42`, 276×600 dataset. The ablation used a deliberately
reduced budget (`base=16`, 400 training images, 200 gradient steps, batch 4)
so that all variants fit on 2 CPU cores. **Absolute numbers are therefore below
the 0.98 reported for the full-budget hybrid; the point is the ordering and the
gaps, which are measured under an identical budget for every row.**

See `AUDIT_REPORT.md` for the full table and interpretation.

## Why a matched-budget ablation and not a full retrain

The hybrid's reported run is 1936 gradient steps at batch 8 on full 276×600
images with `base=32` — roughly 29 hours on 2 CPU cores per variant. Seven
variants is not feasible without a GPU. Reducing width and step count equally
for every variant keeps the comparison controlled, which is the property that
matters. Rerun with `--base 32 --steps 1936 --batch 8` on the RTX 3060 to get
the publication table.
