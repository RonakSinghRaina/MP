#!/usr/bin/env bash
# The recommended LOFAR protocol. See PART 11.12 of RFI-project-context.md.
#
#   ./run_lofar_baseline.sh
#
# Runs the matched-budget arm at three seeds, then one convergence control.
# Refuses to start on battery -- the GPU is clamped to 1/10 clock there.
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" != "1" ]; then
  echo "REFUSING TO START: laptop is on battery."
  echo "The GPU clamps to ~210 MHz of 2100 MHz unplugged (PART 11.10c),"
  echo "making this run ~15x slower. Plug in and re-run."
  exit 1
fi

export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
PY=~/tf-env/bin/python

for SEED in 0 1 2; do
  echo "=== matched budget, seed $SEED ==="
  $PY experiments/lofar_tfunet_baseline.py \
      --norm fixed --epochs 60 --iters_per_epoch 175 --seed "$SEED" \
      --output_dir "lofar_runs/matched_seed${SEED}"
done

echo "=== convergence control (full passes, seed 0) ==="
$PY experiments/lofar_tfunet_baseline.py \
    --norm fixed --epochs 20 --iters_per_epoch 0 --seed 0 \
    --output_dir lofar_runs/converged_seed0

echo
echo "All runs finished. Metrics:"
for f in lofar_runs/*/eval_test/metrics.json; do
  echo "--- $f"
  $PY -c "import json,sys; m=json.load(open('$f')); print('   pooled_f1 %.4f  max_f1 %.4f  roc %.4f  th %.3f'%(m['pooled_f1'],m['max_f1'],m['roc_auc'],m['threshold']))"
done
