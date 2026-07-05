#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-smolvla_memory_layerwise_0609}"
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/res/VLA-scratch}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/res/install_logs/${RUN_ID}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/2026-06-09/${RUN_ID}}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval/2026-06-09/${RUN_ID}}"
LOG_FILE="${LOG_DIR}/monitor_30min.log"
PID_FILE="${LOG_DIR}/monitor_30min.pid"

mkdir -p "$LOG_DIR"
echo $$ > "$PID_FILE"

while true; do
  {
    echo "===== $(date '+%F %T %Z') ====="
    echo
    echo "processes:"
    ps -eo pid,ppid,etime,stat,cmd | grep -E 'smolvla_memory_layerwise_0609|train_policy.py|eval_libero_lerobot|torch.distributed.run' | grep -v grep || true
    echo
    echo "gpu:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
    echo
    echo "recent train logs:"
    for method in framesamp tokendrop; do
      for suite in goal spatial object; do
        log="${LOG_DIR}/train_${method}_${suite}.log"
        if [[ -f "$log" ]]; then
          echo "--- ${method}/${suite} ---"
          tail -25 "$log"
        fi
      done
    done
    echo
    echo "checkpoints:"
    find "$OUT_DIR" -maxdepth 5 -type f -name model.pt -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -30 || true
    echo
    echo "eval:"
    find "$EVAL_DIR" -name eval_info.json -type f -print 2>/dev/null | sort | while read -r f; do
      printf '%s ' "$f"
      python - "$f" <<'PY' 2>/dev/null || true
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
overall = data.get("overall", {})
print(overall.get("pc_success", overall.get("success_rate", "NA")))
PY
    done
    echo
    echo "disk:"
    df -h /root/autodl-tmp /root 2>/dev/null || true
    echo
  } >> "$LOG_FILE" 2>&1
  sleep 1800
done
