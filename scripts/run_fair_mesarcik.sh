#!/usr/bin/env bash
# Fair comparison under Mesarcik et al. (2022)'s LOFAR training settings (PART 31).
#
#   ./scripts/run_fair_mesarcik.sh            # seed 0  (~3 h U-Net + ~7.4 h HybridRFINet)
#   ./scripts/run_fair_mesarcik.sh 0 1 2      # three seeds (~31 h)
#
# For each seed it trains, one after the other, through ONE shared pipeline
# (data split, clip/log/min-max preprocessing, 32x32 tiles, batch order, metric):
#   1. their U-Net        experiments/mesarcik_repro/train_mesarcik_unet.py
#                         -> runs/lofar/mesarcik_unet_clean_bce_seed<S>
#   2. HybridRFINet       experiments/mesarcik_repro/train_hybrid_mesarcik_settings.py
#                         -> runs/lofar/hybrid_mesarcik_clean_bce_seed<S>
# Identical for both: 6621 leak-free training images (735 validation only for the
# pooled-F1 threshold), clip [|mu-3sd|, mu+95sd] from training clean pixels -> ln ->
# min-max, non-overlapping 32x32 tiles, batch 1024, Adam lr 1e-3 constant, 100 epochs,
# binary cross-entropy (their paper; their code's swapped loss kills HybridRFINet),
# no class weights, no augmentation, model after the last epoch, max F1 over all
# thresholds on the 109 expert-labelled images (tiles reassembled to 512x512).
#
# Resumable: rerun the same command. Finished runs are skipped. HybridRFINet resumes
# exactly; the U-Net script resumes from its last epoch's weights but restarts its
# optimiser state, so avoid stopping during the U-Net if you can.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# keep the laptop awake for the whole run (idle sleep, suspend, lid close)
if [ -z "${FAIR_MESARCIK_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export FAIR_MESARCIK_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --who="RFI training" \
         --why="fair Mesarcik comparison" --mode=block "$ROOT/scripts/$(basename "$0")" "$@"
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

TF_LIBS="$(ls -d "$HOME"/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')"
NOISE="^WARNING|E0000|I0000|W0000|oneDNN|cuda_|computation placer|rebuild TensorFlow|UserWarning"
mkdir -p runs/lofar/logs

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0)
for seed in "${SEEDS[@]}"; do
    log="runs/lofar/logs/fair_mesarcik_seed${seed}.log"
    unet="runs/lofar/mesarcik_unet_clean_bce_seed${seed}"
    hyb="runs/lofar/hybrid_mesarcik_clean_bce_seed${seed}"

    if [ -f "$unet/eval_test/metrics.json" ]; then
        echo "-- seed $seed: U-Net already done, skipping" | tee -a "$log"
    else
        echo "== seed $seed: their U-Net  ($(date '+%F %H:%M'))" | tee -a "$log"
        LD_LIBRARY_PATH="${TF_LIBS}${LD_LIBRARY_PATH:-}" TF_FORCE_GPU_ALLOW_GROWTH=true \
        "$HOME/tf-env/bin/python" experiments/mesarcik_repro/train_mesarcik_unet.py \
            --setup clean --correct_loss --eval_every 10 --seed "$seed" \
            2>&1 | grep --line-buffered -vE "$NOISE" | tee -a "$log"
    fi

    if [ -f "$hyb/eval_test/metrics.json" ]; then
        echo "-- seed $seed: HybridRFINet already done, skipping" | tee -a "$log"
    else
        echo "== seed $seed: HybridRFINet  ($(date '+%F %H:%M'))" | tee -a "$log"
        "$HOME/torch-env/bin/python" experiments/mesarcik_repro/train_hybrid_mesarcik_settings.py \
            --setup clean --loss bce --eval_every 10 --seed "$seed" \
            2>&1 | grep --line-buffered -vE "$NOISE" | tee -a "$log"
    fi
done

echo
echo "================= fair comparison: Mesarcik et al. settings, leak-free data ================="
"$HOME/torch-env/bin/python" - "${SEEDS[@]}" <<'EOF'
import json, os, sys
import numpy as np
rows = {"their U-Net": "runs/lofar/mesarcik_unet_clean_bce_seed{}",
        "HybridRFINet": "runs/lofar/hybrid_mesarcik_clean_bce_seed{}"}
for name, pat in rows.items():
    v = {k: [] for k in ("max_f1", "pooled_f1", "roc_auc")}
    for s in sys.argv[1:]:
        f = os.path.join(pat.format(s), "eval_test", "metrics.json")
        if os.path.exists(f):
            m = json.load(open(f))
            for k in v:
                v[k].append(m.get(k, float("nan")))
    n = len(v["max_f1"])
    if n == 0:
        print(f"  {name:13s}: no finished runs"); continue
    fmt = lambda x: f"{np.mean(x):.4f}" + (f" +/- {np.std(x, ddof=1):.4f}" if len(x) > 1 else "")
    print(f"  {name:13s}: max F1 {fmt(v['max_f1'])} | pooled F1 {fmt(v['pooled_f1'])} | "
          f"ROC AUC {fmt(v['roc_auc'])}   ({n} seed{'s' if n > 1 else ''})")
EOF
