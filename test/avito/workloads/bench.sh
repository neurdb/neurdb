#!/usr/bin/env bash
# Reuse (feature cache ON) vs No-reuse (cache OFF) comparison, avito AdCTR sweep.
#
# Same per-task feature SQL (02_features) in both modes. The ONLY difference:
#   * REUSE   : reset cache ONCE up front  -> task computes only its DELTA keys.
#   * NO-REUSE: reset cache BEFORE each task -> task recomputes ALL its keys.
# We time the feature-compute step and report rows actually computed.
#
# NOTE: warm-cache wall time of a single run; for the paper run a few times and
#       take the median (consider dropping PG/OS cache between runs).
# Usage:  bash bench.sh            (horizons 1 3 7)
#         HORIZONS="1 2 4" bash bench.sh
set -euo pipefail

CONTAINER=neurdb_dev
DIR=/code/neurdb-dev/test/avito/workloads
PSQL="/code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -d avito -v ON_ERROR_STOP=1 -q"
HORIZONS="${HORIZONS:-1 3 7}"

psqlq() { docker exec "$CONTAINER" bash -lc "$PSQL $*"; }
timeit() {  # $1 = label, rest = psql args ; prints elapsed, returns nothing
  local label="$1"; shift
  local t0 t1
  t0=$(date +%s.%N)
  docker exec "$CONTAINER" bash -lc "$PSQL $* >/dev/null"
  t1=$(date +%s.%N)
  awk -v l="$label" -v a="$t0" -v b="$t1" 'BEGIN{printf "  %-28s %8.2f s\n", l, b-a}'
}
rows_computed() {  # report INSERT count by reading cache growth is messy; just show cache size
  psqlq "-c \"SELECT count(*) AS cache_rows FROM w_feat_cache;\"" | sed -n '3p'
}

echo "== prereqs: cutoffs + labels (built once) =="
psqlq "-f $DIR/tool_cutoffs.sql" >/dev/null
for h in $HORIZONS; do psqlq "-v h=$h -f $DIR/01_label_adctr.sql" >/dev/null; done
echo "  done"

echo "== REUSE (cache ON): reset once, each task computes only its delta =="
psqlq "-f $DIR/tool_feat_cache_init.sql" >/dev/null
t0=$(date +%s.%N)
for h in $HORIZONS; do timeit "features h=$h" "-v h=$h -f $DIR/02_features_adctr_pit.sql"; done
t1=$(date +%s.%N)
awk -v a="$t0" -v b="$t1" 'BEGIN{printf "  %-28s %8.2f s\n", "REUSE feature-compute total", b-a}'
echo -n "  cache rows (= union computed): "; rows_computed
for h in $HORIZONS; do psqlq "-v h=$h -f $DIR/03_task_table.sql" >/dev/null; done

echo "== NO-REUSE (cache OFF): reset before each task, recompute everything =="
t0=$(date +%s.%N)
NOREUSE_ROWS=0
for h in $HORIZONS; do
  psqlq "-f $DIR/tool_feat_cache_init.sql" >/dev/null          # wipe cache => full recompute
  timeit "features h=$h" "-v h=$h -f $DIR/02_features_adctr_pit.sql"
  n=$(rows_computed | tr -dc '0-9')
  NOREUSE_ROWS=$((NOREUSE_ROWS + n))
done
t1=$(date +%s.%N)
awk -v a="$t0" -v b="$t1" 'BEGIN{printf "  %-28s %8.2f s\n", "NO-REUSE feature-compute total", b-a}'
echo "  rows computed (sum over tasks): $NOREUSE_ROWS"
