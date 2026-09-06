#!/usr/bin/env bash
# HybridRFINet on real LOFAR, 3 seeds, matched to the hybrid's own synthetic
# budget (700 x 40 = 28,000 gradient steps). See PART 13 of RFI-project-context.md.
#
#   ./scripts/run_lofar_hybrid.sh
#
# ~1 h per seed on AC. Seeds first; the per-image-normalisation arm last, since
# the hybrid's GroupNorm is the reason PART 1 gave for per-image not hurting it.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" != "1" ]; then
  echo "REFUSING TO START: on battery. The GPU throttles ~15x (PART 11.10c)."; exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
  echo "REFUSING TO START: GPU is busy."
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv; exit 1
fi

PY=~/torch-env/bin/python

for SEED in 0 1 2; do
  echo "=== hybrid base32, fixed norm, seed $SEED  ($(date '+%H:%M')) ==="
  $PY experiments/lofar_hybrid.py --base 32 --norm fixed --seed "$SEED" \
      --output_dir "runs/lofar/hybrid_b32_fixed_seed${SEED}"
done

echo "=== hybrid base32, PER-IMAGE norm, seed 0  ($(date '+%H:%M')) ==="
$PY experiments/lofar_hybrid.py --base 32 --norm per_image --seed 0 \
    --output_dir runs/lofar/hybrid_b32_perimage_seed0

echo; echo "Done. ./scripts/status.sh for the summary."
