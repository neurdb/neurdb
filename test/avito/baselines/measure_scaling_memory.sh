#!/usr/bin/env bash
# ============================================================================
# measure_scaling_memory.sh -- peak memory per (scale, system) for the
#                              data-scaling figure.
# ============================================================================
# Methodology (one fresh process set per run, like /usr/bin/time on a script):
#   * LOTUS / Palimpzest: /usr/bin/time -v over the full run on the exported
#     parquet of that scale -> "Maximum resident set size".
#   * NeurEngine: the AI server is restarted fresh, then the full in-database
#     run (DB=<scale db>) executes while a 1s sampler records
#       - RSS of the AI server process (model runtime + task slices), and
#       - the largest single Postgres backend RSS (bounded by shared_buffers
#         + work_mem; the relational data itself stays on disk).
#     Peak = max(server) + max(backend).
#
# The medium baselines are NOT re-measured: logs/e2e_comparison + the earlier
# /usr/bin/time runs on the identical 1x export already produced 6.2 / 6.4 GB;
# those constants are reused below.
#
# Usage:  bash measure_scaling_memory.sh
# Output: logs/scaling/scaling_memory.csv  (scale,system,gb)
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKLOADS="$HERE/../workloads"
LOGS="$HERE/logs/scaling"
mkdir -p "$LOGS"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

CONTAINER=neurdb_dev
RUNTIME=/code/neurdb-dev/aiengine/runtime
DEVICE="${DEVICE:-cuda:0}"
CSV="$LOGS/scaling_memory.csv"
echo "scale,system,gb" > "$CSV"

db_of() { case $1 in mini) echo avito_mini;; small) echo avito_small;;
                     medium) echo avito;; large) echo avito_large;; esac; }

restart_server() {
  docker exec "$CONTAINER" bash -c "pkill -f '[s]erver\.py' || true"
  sleep 3
  docker exec -d "$CONTAINER" bash -lc "cd $RUNTIME && CUDA_VISIBLE_DEVICES=0 \
    NR_LOG_LEVEL=INFO NR_PORT=8090 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
    python -u server.py >> /code/neurdb-dev/test/server_8090.log 2>&1"
  for _ in $(seq 1 30); do
    sleep 2
    docker exec "$CONTAINER" curl -s -o /dev/null http://127.0.0.1:8090/ && return
  done
  echo "AI server failed to restart" >&2
  exit 1
}

ne_mem_gb() {  # ne_mem_gb <scale> <db>  -> appends CSV row
  local scale=$1 db=$2 samples="$LOGS/${scale}_ne_mem.samples"
  restart_server
  : > "$samples"
  ( while true; do
      docker exec "$CONTAINER" bash -c \
        "ps -eo rss,args | awk '/[s]erver\.py/{s+=\$1} /postgres: neurdb $db/{if(\$1>m)m=\$1} END{print s+0, m+0}'" \
        >> "$samples" 2>/dev/null || true
      sleep 1
    done ) &
  local sampler=$!
  DB=$db CACHE=on MODES=on bash "$WORKLOADS/run_tasks.sh" > "$LOGS/${scale}_ne_mem.log" 2>&1
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  awk -v scale="$scale" '{if($1>s)s=$1; if($2>b)b=$2}
       END{printf "%s,neurengine,%.1f\n", scale, (s+b)/1048576}' "$samples" >> "$CSV"
  tail -1 "$CSV"
}

bl_mem_gb() {  # bl_mem_gb <scale> <system> <env> <script>
  local scale=$1 sys=$2 env=$3 script=$4
  conda activate "$env"
  /usr/bin/time -v python "$HERE/$script" --data "$HERE/data_$scale" \
    --out "$LOGS/${sys}_${scale}_mem" --device "$DEVICE" \
    > "$LOGS/${scale}_${sys}_mem.log" 2>&1 || true
  conda deactivate
  local kb
  kb=$(grep "Maximum resident set size" "$LOGS/${scale}_${sys}_mem.log" | grep -o '[0-9]*')
  awk -v scale="$scale" -v sys="$sys" -v kb="$kb" \
      'BEGIN{printf "%s,%s,%.1f\n", scale, sys, kb/1048576}' >> "$CSV"
  tail -1 "$CSV"
}

for scale in mini small medium large; do
  echo "==================== scale=$scale ===================="
  ne_mem_gb "$scale" "$(db_of "$scale")"
  if [ "$scale" = "medium" ]; then
    # measured earlier with the same /usr/bin/time method on the 1x export
    echo "medium,lotus,6.2" >> "$CSV"
    echo "medium,palimpzest,6.4" >> "$CSV"
  else
    bl_mem_gb "$scale" lotus bl_lotus run_lotus.py
    bl_mem_gb "$scale" palimpzest bl_pz run_palimpzest.py
  fi
done

echo "== done =="
cat "$CSV"
