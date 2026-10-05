#!/usr/bin/env bash
# Reproduce Mesarcik et al.'s LOFAR U-Net (published max F1 0.5876 +/- 0.0031)
# with their own code and pipeline -- PART 21.
#
#   ./scripts/run_mesarcik_repro.sh            # paper_code + clean, seed 0
#   ./scripts/run_mesarcik_repro.sh 0 1 2      # both setups, three seeds
#
# Resumable: rerun the same command and finished runs are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" != "1" ]; then
    echo "REFUSING TO START: the laptop is on battery (GPU capped at 210 of 2100 MHz)." >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "REFUSING TO START: another process is using the GPU." >&2
    exit 1
fi

export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
PY="$HOME/tf-env/bin/python"

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0)
for seed in "${SEEDS[@]}"; do
  for setup in paper_code clean; do
    out="runs/lofar/mesarcik_unet_${setup}_seed${seed}"
    if [ -f "$out/eval_test/metrics.json" ]; then echo "-- skip $setup seed $seed"; continue; fi
    echo; echo "-- $setup, seed $seed -> $out"
    "$PY" experiments/mesarcik_repro/train_mesarcik_unet.py --setup "$setup" --seed "$seed" \
        2>&1 | grep --line-buffered -vE "^WARNING|E0000|I0000|W0000|oneDNN|cuda_|computation placer|rebuild TensorFlow"
  done
done
