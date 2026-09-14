#!/usr/bin/env bash
# ============================================================================
# run_baselines.sh -- run the LOTUS / Palimpzest baselines end to end (HOST).
# ============================================================================
# Same NLQ task set as test/avito/workloads/run_tasks.sh (horizons 1/3/7,
# candidate categories 60/26/27, TabPFN), in the export-execute-import
# pattern:  export tables from NeurDB once, then each framework runs the
# whole horizon sweep on the exported data, recomputing everything per task
# (no cross-task reuse -- neither system has any).
#
# Prereqs: bash setup_envs.sh all   (conda envs bl_lotus / bl_pz)
#
# Usage:  bash run_baselines.sh [lotus|pz|all(default)]
# Output: logs/<system>.log (TIMING,<step>,<seconds> lines)
#         logs/baseline_results.csv (aggregated)
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

WHICH="${1:-all}"
HORIZONS="${HORIZONS:-1 3 7}"
DEVICE="${DEVICE:-cuda:0}"

run_one() {  # run_one <system> <conda-env> <script>
  local sys=$1 env=$2 script=$3
  echo "== $sys baseline (env $env) =="
  conda activate "$env"
  if [ ! -f "$HERE/data/searchstream.parquet" ] || [ "${FORCE_EXPORT:-0}" = "1" ]; then
    echo ">> export from NeurDB"
    python "$HERE/export_data.py" 2>&1 | tee "$LOGS/${sys}_export.log"
  else
    echo ">> export skipped (data/ already present; FORCE_EXPORT=1 to redo)"
  fi
  python "$HERE/$script" --horizons $HORIZONS --device "$DEVICE" 2>&1 | tee "$LOGS/$sys.log"
  conda deactivate
}

case "$WHICH" in
  lotus) run_one lotus bl_lotus run_lotus.py ;;
  pz)    run_one palimpzest bl_pz run_palimpzest.py ;;
  all)   run_one lotus bl_lotus run_lotus.py
         run_one palimpzest bl_pz run_palimpzest.py ;;
  *)     echo "usage: $0 [lotus|pz|all]"; exit 1 ;;
esac

# ---- aggregate TIMING lines into one CSV ------------------------------------
python3 - "$LOGS" <<'EOF'
import csv, os, re, sys

logs = sys.argv[1]
rows = []
for sys_name in ("lotus", "palimpzest"):
    log = os.path.join(logs, f"{sys_name}.log")
    exp = os.path.join(logs, f"{sys_name}_export.log")
    if not os.path.exists(log):
        continue
    for path in (exp, log):
        if not os.path.exists(path):
            continue
        for line in open(path):
            m = re.match(r"TIMING,([^,]+),([\d.]+)", line.strip())
            if m:
                rows.append({"system": sys_name, "step": m.group(1), "seconds": float(m.group(2))})

out = os.path.join(logs, "baseline_results.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["system", "step", "seconds"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out} ({len(rows)} rows)")
EOF

echo "== done =="
