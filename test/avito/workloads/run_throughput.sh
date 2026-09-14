#!/usr/bin/env bash
# ============================================================================
# run_throughput.sh -- distributed-inference throughput sweep (NeurEngine setting)
# ============================================================================
# NLQ: "For each candidate ad, predict CTR over the next 1/2/3/4/5/7 days."
# Six per-horizon PREDICT tasks, run under the FULL NeurEngine setting
# (PIT-feature cache reuse ON, AI-operator scheduling ON), against a pool of
# N TabPFN AI servers (one GPU each).
#
# What the engine does per PREDICT (see nr_pipeline/interface2.c):
#   * the in-context "train" phase is BROADCAST to every registered engine
#     (each fits the same candidate context, returning its own model id);
#   * every inference batch is then SPLIT row-wise into N chunks predicted in
#     parallel, one chunk per engine (DistributedInfer).
#
# Sweep: for each N in SERVERS, register engines 8090..809(N-1) and run the
# task set twice:
#   seq   tasks one after another  -> per-task latency (batch-sharding speedup)
#   conc  all tasks at once        -> NLQ wall time    (task-level parallelism)
#
# Knobs (env):
#   HORIZONS  default "1 2 3 4 5 7"  (h=14 does NOT fit the 25-day data window:
#             cutoffs reserve only 7d of future for labels)
#   CAND      candidate predicate, default "categoryid IN (60, 26, 27)"
#   SERVERS   server counts to sweep, default "1 2 4 8"
#   MODES     "seq conc" (default) -- which passes to run per N
#   BASE_PORT default 8090; server i -> port BASE_PORT+i, CUDA device i
#
# Servers are started on demand inside the container (python server.py) and
# left running. Output: logs/throughput/*.log + TIMING,<step>,<seconds> lines,
# aggregated into logs/throughput_results.csv.
#
# Usage:  bash run_throughput.sh
#         SERVERS="1 8" MODES=conc bash run_throughput.sh
# ============================================================================
set -euo pipefail

CONTAINER=neurdb_dev
PSQL="/code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -d avito -v ON_ERROR_STOP=1"
DIR=/code/neurdb-dev/test/avito/workloads
RUNTIME=/code/neurdb-dev/aiengine/runtime

HORIZONS="${HORIZONS:-1 2 3 4 5 7}"
CAND="${CAND:-categoryid IN (60, 26, 27)}"
SERVERS="${SERVERS:-1 2 4 8}"
MODES="${MODES:-seq conc}"
BASE_PORT="${BASE_PORT:-8090}"
# CPU threads per AI server. TabPFN's fit preprocessing is CPU-heavy and
# numpy/torch default to ~all cores PER PROCESS; with 8 servers fitting the
# same broadcast context simultaneously that oversubscribes the CPU 4-8x and
# fits go 1s -> 6-8s. Cap each server to its fair share (48 cores / 8).
THREADS="${THREADS:-6}"

HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HOSTDIR/logs/throughput"
mkdir -p "$LOGDIR"

MAXN=0
for n in $SERVERS; do [ "$n" -gt "$MAXN" ] && MAXN=$n; done

run() { docker exec "$CONTAINER" bash -lc "$PSQL $*"; }

now() { date +%s.%N; }
dt() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f", b - a }'; }

# ---- 0. AI server pool ------------------------------------------------------
ensure_server() {  # ensure_server <idx>  (port BASE_PORT+idx on GPU idx)
  local i=$1 port=$((BASE_PORT + $1))
  if docker exec "$CONTAINER" curl -s -o /dev/null "http://127.0.0.1:$port/"; then
    echo "   server :$port already up"
    return
  fi
  echo "   starting server :$port (GPU $i, $THREADS cpu threads)"
  docker exec -d "$CONTAINER" bash -lc \
    "cd $RUNTIME && CUDA_VISIBLE_DEVICES=$i NR_LOG_LEVEL=INFO NR_PORT=$port \
     OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
     NUMEXPR_NUM_THREADS=$THREADS \
     python -u server.py >> /code/neurdb-dev/test/server_$port.log 2>&1"
  for _ in $(seq 1 30); do
    sleep 2
    if docker exec "$CONTAINER" curl -s -o /dev/null "http://127.0.0.1:$port/"; then
      return
    fi
  done
  echo "server :$port failed to start" >&2
  exit 1
}

register_engines() {  # register_engines <N>
  local n=$1 sql="DELETE FROM pg_catalog.nr_aiengine;"
  for i in $(seq 0 $((n - 1))); do
    sql+=" SELECT insert_ai_engine('127.0.0.1', $((BASE_PORT + i)));"
  done
  run "-q -c \"$sql\""
}

echo "== throughput sweep: horizons=[$HORIZONS] servers=[$SERVERS] modes=[$MODES] =="
echo ">> 0. server pool (size $MAXN)"
for i in $(seq 0 $((MAXN - 1))); do ensure_server "$i"; done

# ---- 1. prep: build any missing w_task_<h> (cache reuse ON) ------------------
echo ">> 1. prep (cache reuse on; only missing horizons are built)"
for h in $HORIZONS; do
  if run "-t -A -c \"SELECT count(*) FROM w_task_$h\"" 2>/dev/null | grep -qE '^[1-9]'; then
    echo "   w_task_$h exists, skip"
    continue
  fi
  echo "   building h=$h (01 label -> 02 features -> 03 task)"
  run "-q -v h=$h -f $DIR/01_label_adctr.sql" > /dev/null
  run "-q -v h=$h -f $DIR/02_features_adctr_pit.sql" > /dev/null
  run "-q -v h=$h -f $DIR/03_task_table.sql" > /dev/null
done

# ---- 2. warmup: model load + first fit on EVERY server -----------------------
echo ">> 2. warmup (broadcast PREDICT over all $MAXN servers, twice)"
register_engines "$MAXN"
WARMUP_SQL="SET nr_task_batch_size TO 512; SET nr_task_num_batches TO 1; SET nr_task_epoch TO 1; \
SELECT count(*) FROM (PREDICT VALUE OF label_ctr FROM (SELECT * FROM w_task_1 LIMIT 512) AS t TRAIN tabpfn ON *) AS p;"
for w in 1 2; do
  run "-q -c \"$WARMUP_SQL\"" > /dev/null
  echo "   warmup pass $w done"
done

# ---- 3. sweep ----------------------------------------------------------------
predict_one() {  # predict_one <h> <logfile> <engine_pin>
  # engine_pin -1 = broadcast context to all N engines + shard every inference
  # batch across them (intra-task parallelism); >=0 = pin this task's PREDICT
  # to engine (pin % N) so concurrent tasks spread over the pool instead of
  # all replicating their context fit on every server (inter-task parallelism).
  docker exec "$CONTAINER" bash -lc \
    "$PSQL -v h=$1 -v sched=on -v cand='$CAND' -v k=5 \
      -c 'SET nr_pipeline.engine_pin = ${3:--1};' -f $DIR/08_predict_candidates.sql" \
    >> "$2" 2>&1
}

for n in $SERVERS; do
  echo ">> N=$n servers"
  register_engines "$n"

  if echo " $MODES " | grep -q " seq "; then
    # sequential tasks, broadcast+shard: intra-task parallelism only
    log="$LOGDIR/n${n}_seq.log"; : > "$log"
    t0=$(now)
    for h in $HORIZONS; do
      th0=$(now)
      predict_one "$h" "$log" -1
      th1=$(now)
      echo "TIMING,seq_h${h}_n${n},$(dt "$th0" "$th1")" | tee -a "$log"
    done
    t1=$(now)
    echo "TIMING,seq_total_n${n},$(dt "$t0" "$t1")" | tee -a "$log"
  fi

  if echo " $MODES " | grep -q " conc "; then
    # concurrent tasks, one engine per task (round-robin): inter-task parallelism
    log="$LOGDIR/n${n}_conc.log"; : > "$log"
    t0=$(now)
    pids=()
    idx=0
    for h in $HORIZONS; do
      pin=$((idx % n)); idx=$((idx + 1))
      (
        th0=$(now)
        predict_one "$h" "$LOGDIR/n${n}_conc_h${h}.log" "$pin"
        th1=$(now)
        echo "TIMING,conc_h${h}_n${n},$(dt "$th0" "$th1")"
      ) >> "$log" &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    t1=$(now)
    echo "TIMING,conc_wall_n${n},$(dt "$t0" "$t1")" | tee -a "$log"
    grep '^TIMING,conc_h' "$log" || true
  fi
done

# ---- 4. aggregate ------------------------------------------------------------
CSV="$HOSTDIR/logs/throughput_results.csv"
{
  echo "n_servers,mode,horizon,seconds"
  for n in $SERVERS; do
    for mode in $MODES; do
      f="$LOGDIR/n${n}_${mode}.log"
      [ -f "$f" ] || continue
      grep '^TIMING,' "$f" | while IFS=, read -r _ step secs; do
        case "$step" in
          ${mode}_h*_n${n})    h=${step#${mode}_h}; h=${h%%_*}; echo "$n,$mode,$h,$secs" ;;
          ${mode}_total_n${n}) echo "$n,$mode,total,$secs" ;;
          ${mode}_wall_n${n})  echo "$n,$mode,wall,$secs" ;;
        esac
      done
    done
  done
} > "$CSV"

echo
echo "== results =="
column -s, -t < "$CSV"
echo "csv: $CSV"
echo "== done =="
