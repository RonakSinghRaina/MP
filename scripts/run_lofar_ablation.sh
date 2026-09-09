#!/usr/bin/env bash
# LOFAR architecture ablation -- PART 17.3.
#
# Decomposes the +0.1110 "architecture" gain over tf_unet that PART 13.10
# claims but never isolated. Everything except the model is held fixed:
# same data, splits, leakage checks, normalisation, loss, optimiser, step
# budget, validation-selected threshold and metrics.
#
# Config matches the headline run (PART 13.11): base 8, fixed norm,
# --no_class_weight, 700 x 40 = 28,000 gradient steps.
#
# Usage:
#   ./scripts/run_lofar_ablation.sh            # 6 variants x seed 0   (~2.5 h)
#   ./scripts/run_lofar_ablation.sh 0 1 2      # 6 variants x 3 seeds  (~7.5 h)
#
# Each run is resumable: rerun the identical command and it continues from
# progress.json. Completed runs are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$HOME/torch-env/bin/python"

# ---- refuse to run on battery -------------------------------------------
# CLAUDE.md section 1: on battery the SM clock is pinned at 210 MHz of 2100
# and training is 15.3x slower. This turns a 2.5 h sweep into 38 h.
if [ -e /sys/class/power_supply/AC/online ] || compgen -G "/sys/class/power_supply/A*/online" > /dev/null; then
    if [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" != "1" ]; then
        echo "REFUSING TO START: the laptop is on battery." >&2
        echo "The GPU is capped at 210 MHz of 2100 MHz, which makes this 15.3x slower" >&2
        echo "(about 38 hours instead of 2.5). Plug in and rerun." >&2
        exit 1
    fi
fi

# ---- refuse to run alongside another training job -----------------------
# Only one run fits on a 6 GB card; a second dies with RESOURCE_EXHAUSTED
# and the traceback blames the wrong thing.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "REFUSING TO START: another process is already on the GPU:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
    exit 1
fi

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0)

# hybrid_full is the control: it must reproduce HybridRFINet (593,842 params).
# If it does not land on 0.6603 +/- 0.0040 the ablation skeleton is unfaithful
# and nothing else in the table can be trusted.
VARIANTS=(hybrid_full no_strip no_eca no_res plain_unet no_groupnorm)

echo "=========================================================================="
echo "  LOFAR ablation  |  ${#VARIANTS[@]} variants x ${#SEEDS[@]} seed(s)"
echo "  reference: hybrid base 8, no class weight = 0.6603 +/- 0.0040 max F1"
echo "=========================================================================="

for seed in "${SEEDS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    out="runs/lofar/abl_${v}_seed${seed}"
    if [ -f "$out/eval_test/metrics.json" ]; then
        echo "-- skip $v seed $seed (already complete)"
        continue
    fi
    echo
    echo "-- $v, seed $seed  -> $out"
    "$PY" experiments/lofar_hybrid.py \
        --variant "$v" \
        --base 8 \
        --norm fixed \
        --no_class_weight \
        --seed "$seed" \
        --output_dir "$out"
  done
done

echo
echo "=========================================================================="
echo "  DONE. Summarise with:"
echo "    $PY analysis/summarise_ablation.py"
echo "=========================================================================="
