#!/usr/bin/env bash
# HybridRFINet trained with Mesarcik et al. (2022)'s LOFAR settings (PART 31).
#
#   ./scripts/run_hybrid_mesarcik.sh            # seed 0, about 7.5 h
#   ./scripts/run_hybrid_mesarcik.sh 0 1 2      # three seeds, about 22 h
#
# Identical to their U-Net's leak-free run (runs/lofar/mesarcik_unet_clean_seed0,
# max F1 0.6166): the same 6621 training images (test copies removed), the same
# clip -> log -> min-max preprocessing (bounds from training data, and the very same
# cached file), the same 32x32 tiles in the same batch order (same seed), batch 1024,
# Adam lr 1e-3 constant, 100 epochs, no class weights, no augmentation, the model
# after the last epoch, max F1 over all thresholds on the 109 expert-labelled images.
#
# One difference, unavoidable: the loss. Their code passes the two loss inputs in the
# wrong order; HybridRFINet collapses with that loss within the first epoch (test F1
# 0.015 at lr 1e-3 and 1e-4, PART 31.1), so it uses the binary cross-entropy that
# their paper describes.
#
# Resumable: rerun the same command; finished seeds are skipped and an interrupted
# seed continues exactly where it stopped. Output: runs/lofar/hybrid_mesarcik_clean_bce_seed<S>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# keep the laptop awake for the whole run (idle sleep, suspend, lid close)
if [ -z "${HYBRID_MESARCIK_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export HYBRID_MESARCIK_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --who="RFI training" \
         --why="HybridRFINet, Mesarcik settings" --mode=block \
         "$ROOT/scripts/$(basename "$0")" "$@"
fi

if [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" != "1" ]; then
    echo "REFUSING TO START: the laptop is on battery (GPU capped at 210 of 2100 MHz)." >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "REFUSING TO START: another process is using the GPU (only one run fits)." >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
    exit 1
fi

mkdir -p runs/lofar/logs
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0)
for seed in "${SEEDS[@]}"; do
    out="runs/lofar/hybrid_mesarcik_clean_bce_seed${seed}"
    log="runs/lofar/logs/hybrid_mesarcik_clean_bce_seed${seed}.log"
    if [ -f "$out/eval_test/metrics.json" ]; then
        echo "-- seed $seed already finished, skipping ($out)"
        continue
    fi
    echo "== HybridRFINet, Mesarcik settings, seed $seed  ($(date '+%F %H:%M'))" | tee -a "$log"
    "$HOME/torch-env/bin/python" experiments/mesarcik_repro/train_hybrid_mesarcik_settings.py \
        --setup clean --loss bce --eval_every 10 --seed "$seed" \
        2>&1 | grep --line-buffered -v "UserWarning" | tee -a "$log"
done

echo
echo "============ HybridRFINet with Mesarcik et al.'s settings (leak-free data) ============"
"$HOME/torch-env/bin/python" - "${SEEDS[@]}" <<'EOF'
import json, os, sys
import numpy as np
v = {k: [] for k in ("max_f1", "pooled_f1", "roc_auc")}
for s in sys.argv[1:]:
    f = f"runs/lofar/hybrid_mesarcik_clean_bce_seed{s}/eval_test/metrics.json"
    if not os.path.exists(f):
        print(f"  seed {s}: not finished"); continue
    m = json.load(open(f))
    print(f"  seed {s}: max F1 {m['max_f1']:.4f} | pooled F1 {m['pooled_f1']:.4f} | ROC AUC {m['roc_auc']:.4f}")
    for k in v:
        v[k].append(m[k])
if len(v["max_f1"]) > 1:
    print("  mean +/- sd: " + " | ".join(
        f"{k} {np.mean(x):.4f} +/- {np.std(x, ddof=1):.4f}" for k, x in v.items()))
print("  their U-Net, same pipeline (their code's loss), seed 0: max F1 0.6166 | pooled F1 0.6159")
EOF
