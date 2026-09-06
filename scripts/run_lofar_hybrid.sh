#!/usr/bin/env bash
# HybridRFINet on real LOFAR, 3 seeds, matched to the hybrid's own synthetic
# budget (700 x 40 = 28,000 gradient steps). See PART 13 of RFI-project-context.md.
#
#   ./scripts/run_lofar_hybrid.sh
#
# ~25 min per base-8 seed on AC (37 ms/step), ~1 h for the base-32 arm. Seeds first; the per-image-normalisation arm last, since
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

# base 8 -- PART 6's efficiency pick, 593,842 params. Three seeds, because the
# real-data seed spread is 0.052 (PART 12.10) and one run proves nothing.
for SEED in 0 1 2; do
  echo "=== hybrid base8, fixed norm, seed $SEED  ($(date '+%H:%M')) ==="
  $PY experiments/lofar_hybrid.py --base 8 --norm fixed --seed "$SEED" \
      --output_dir "runs/lofar/hybrid_b08_fixed_seed${SEED}"
done

# base 32 -- the published width, one seed, to test whether PART 6's
# "the model is oversized" finding transfers from synthetic to real data.
echo "=== hybrid base32, fixed norm, seed 0  ($(date '+%H:%M')) ==="
$PY experiments/lofar_hybrid.py --base 32 --norm fixed --seed 0 \
    --output_dir runs/lofar/hybrid_b32_fixed_seed0

# per-image normalisation on base 8 -- PART 1 argued the hybrid's GroupNorm is
# exactly why per-image does not hurt it the way it hurt tf_unet. Untested here.
echo "=== hybrid base8, PER-IMAGE norm, seed 0  ($(date '+%H:%M')) ==="
$PY experiments/lofar_hybrid.py --base 8 --norm per_image --seed 0 \
    --output_dir runs/lofar/hybrid_b08_perimage_seed0

echo; echo "Done. ./scripts/status.sh for the summary."
