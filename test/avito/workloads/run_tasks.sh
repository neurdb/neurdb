#!/usr/bin/env bash
# ============================================================================
# run_tasks.sh -- the FULL avito AdCTR horizon-sweep task set, end to end.
# ============================================================================
# NLQ: "I have a set of candidate ads to promote. For each candidate, predict
#       CTR over the next 1 / 3 / 7 days and give me one action list."
#
# Task DAG executed by this script (all inside the neurdb_dev container):
#
#   tool_cutoffs -> tool_rollups (shared daily rollups)
#   per horizon h in HORIZONS:
#     01 label  ->  02 features (PIT, vs cache)  ->  03 task table
#     08 PREDICT candidates (sched per MODES; "on" = qual pushed below the
#        AI operator, "off" = operator pinned at root)   -> w_pred_<h>
#   09 action list (join w_pred_1/3/7 -> promote_now/later/keep/reduce)
#
# Knobs (env):
#   DB         database to run against (default avito; see build_scaled_dbs.sh)
#   HORIZONS   default "1 3 7"
#   CAND       candidate predicate over input cols, default
#              "categoryid IN (60, 26, 27)"   (no single quotes allowed)
#   K          top-k for the action list, default 10
#   MODES      AI-operator scheduling mode(s) for 08, default "off on"
#              ("off on" = A/B both; the LAST one listed leaves its output in
#               w_pred_<h>, which 09 then consumes)
#   CACHE      on  (default) = build the shared data-prep artifacts (daily
#                  rollups + PIT feature cache) ONCE -> tasks reuse them,
#                  computing only their feature delta;
#              off = rebuild rollups + reset the cache BEFORE EVERY task ->
#                  each task prepares all of its data from the base tables
#                  (no cross-task reuse)
#   SKIP_PREP  =1 to skip cutoffs/labels/cache/features/task tables when the
#              w_task_<h> tables already exist
#
# Output: human-readable log + one "TIMING,<step>,<seconds>" line per step
# (machine-readable; consumed by run_matrix.sh).
#
# Usage:  bash run_tasks.sh
#         CACHE=off MODES=off bash run_tasks.sh     # fully-naive baseline
#         SKIP_PREP=1 CAND="categoryid = 60" bash run_tasks.sh
# ============================================================================
set -euo pipefail

CONTAINER=neurdb_dev
DB="${DB:-avito}"
PSQL="/code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -d $DB -v ON_ERROR_STOP=1"
DIR=/code/neurdb-dev/test/avito/workloads

HORIZONS="${HORIZONS:-1 3 7}"
CAND="${CAND:-categoryid IN (60, 26, 27)}"
K="${K:-10}"
MODES="${MODES:-off on}"
CACHE="${CACHE:-on}"
SKIP_PREP="${SKIP_PREP:-0}"

run() { docker exec "$CONTAINER" bash -lc "$PSQL $*"; }

TIMING_REPORT=""
timed() {  # timed <step-id> <psql-args...>
  local label=$1; shift
  local t0 t1 dt
  t0=$(date +%s.%N)
  run "$@"
  t1=$(date +%s.%N)
  dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')
  TIMING_REPORT+=$(printf '%-28s %8ss\n' "$label" "$dt")$'\n'
  echo "TIMING,$label,$dt"
}

echo "== task set: horizons=[$HORIZONS] cand=[$CAND] cache=$CACHE modes=[$MODES] k=$K =="

# ---- prep: cutoffs, rollups, labels, PIT feature cache, task tables ---------
# The REUSE axis (CACHE knob) governs the shared data-preparation artifacts:
# the daily rollups (tool_rollups) AND the PIT feature cache. CACHE=on builds
# them ONCE and every task reuses them; CACHE=off rebuilds both from the base
# tables before EVERY task (what an external per-task pipeline has to do).
if [ "$SKIP_PREP" != "1" ]; then
  echo ">> tool: cutoffs"
  timed "tool_cutoffs" "-f $DIR/tool_cutoffs.sql"
  if [ "$CACHE" = "on" ]; then
    echo ">> tool: daily rollups (ONCE = reuse across tasks)"
    timed "tool_rollups" "-f $DIR/tool_rollups.sql"
    echo ">> tool: feature cache init (ONCE = reuse across tasks)"
    timed "cache_init" "-f $DIR/tool_feat_cache_init.sql"
  fi
  for h in $HORIZONS; do
    if [ "$CACHE" != "on" ]; then
      echo ">> tool: daily rollups (per task = NO reuse) h=$h"
      timed "tool_rollups_h$h" "-f $DIR/tool_rollups.sql"
      echo ">> tool: feature cache init (per task = NO reuse) h=$h"
      timed "cache_init_h$h" "-f $DIR/tool_feat_cache_init.sql"
    fi
    echo ">> 01 label h=$h"
    timed "01_label_h$h" "-v h=$h -f $DIR/01_label_adctr.sql"
    echo ">> 02 features h=$h"
    timed "02_features_h$h" "-v h=$h -f $DIR/02_features_adctr_pit.sql"
    echo ">> 03 task table h=$h"
    timed "03_task_h$h" "-v h=$h -f $DIR/03_task_table.sql"
  done
else
  echo ">> prep skipped (SKIP_PREP=1), reusing existing w_task_<h>"
fi

# ---- per-horizon prediction task: AI operator scheduling per MODES ----------
for h in $HORIZONS; do
  for mode in $MODES; do
    echo ">> 08 predict candidates h=$h sched=$mode"
    timed "08_predict_h${h}_$mode" \
      "-v h=$h -v sched=$mode -v cand='$CAND' -v k=5 -f $DIR/08_predict_candidates.sql"
  done
done

# ---- aggregation task: one action list across horizons ----------------------
echo ">> 09 action list"
timed "09_action_list" "-v k=$K -f $DIR/09_action_list.sql"

echo
echo "== timing summary (wall clock) =="
printf '%s' "$TIMING_REPORT"
echo "== done =="
