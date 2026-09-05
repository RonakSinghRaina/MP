#!/usr/bin/env bash
# Status of LOFAR training runs.
#   ./status.sh          one snapshot
#   ./status.sh -w       refresh every 30 s until you press Ctrl+C
cd "$(dirname "$0")"

show() {
  echo "================================================================"
  date '+  %H:%M:%S   LOFAR run status'
  echo "================================================================"

  # --- running processes -----------------------------------------------
  local procs
  procs=$(pgrep -af lofar_tfunet_baseline 2>/dev/null || true)
  if [ -z "$procs" ]; then
    echo "  no training process running"
  else
    echo "$procs" | while read -r pid rest; do
      local et cpu out
      et=$(ps -o etime= -p "$pid" | tr -d ' ')
      cpu=$(ps -o %cpu= -p "$pid" | tr -d ' ')
      out=$(echo "$rest" | grep -o '\-\-output_dir [^ ]*' | awk '{print $2}')
      echo "  RUNNING pid $pid   elapsed $et   cpu ${cpu}%"
      echo "          $out"
    done
  fi

  # --- GPU --------------------------------------------------------------
  echo
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm \
             --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
  if [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" != "1" ]; then
    echo "  *** ON BATTERY -- GPU is throttled ~15x. Plug in. ***"
  fi

  # --- progress of every run -------------------------------------------
  echo
  printf "  %-22s %8s %10s %8s %s\n" "run" "epochs" "best_val_F1" "at_ep" "last write"
  for f in lofar_runs/*/progress.json; do
    [ -e "$f" ] || continue
    local name
    name=$(basename "$(dirname "$f")")
    python3 - "$f" "$name" <<'PY'
import json, os, sys, time
f, name = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(f))
except Exception:
    print(f"  {name:<22} (unreadable)"); raise SystemExit
age = time.time() - os.path.getmtime(f)
age_s = f"{age:.0f}s ago" if age < 120 else (f"{age/60:.0f}m ago" if age < 7200 else f"{age/3600:.1f}h ago")
done = os.path.exists(os.path.join(os.path.dirname(f), "eval_test", "metrics.json"))
print(f"  {name:<22} {d['epochs_completed']:>8} {d['best_f1']:>10.4f} {str(d['best_epoch']):>8}  {age_s}"
      + ("  [FINISHED]" if done else ""))
PY
  done

  # --- final results ----------------------------------------------------
  echo
  echo "  finished runs (test set, human labels):"
  local any=0
  for f in lofar_runs/*/eval_test/metrics.json; do
    [ -e "$f" ] || continue
    any=1
    python3 - "$f" <<'PY'
import json, sys, os
m = json.load(open(sys.argv[1]))
n = os.path.basename(os.path.dirname(os.path.dirname(sys.argv[1])))
print(f"    {n:<22} pooled_F1 {m['pooled_f1']:.4f}   max_F1 {m['max_f1']:.4f}   ROC {m['roc_auc']:.4f}")
PY
  done
  [ "$any" = 0 ] && echo "    (none yet)"
  echo
  echo "  benchmarks: sigma-clip 0.4103 | AOFlagger 0.5698 | paper's U-Net 0.5876 (max-F1)"
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; show; echo; echo "  refreshing every 30s -- Ctrl+C to stop"; sleep 30; done
else
  show
fi
