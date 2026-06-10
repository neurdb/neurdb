#!/usr/bin/env bash
# ============================================================================
# run_matrix.sh -- the 2x2 ablation: feature-cache reuse x AI-operator sched.
# ============================================================================
# Runs the FULL task set (run_tasks.sh) under 4 settings:
#
#   setting              CACHE  MODES   meaning
#   cache_on__sched_on    on     on     co-designed: feature reuse + planner
#                                       pushes quals below the AI operator
#   cache_on__sched_off   on     off    feature reuse, operator pinned at root
#   cache_off__sched_on   off    on     no feature reuse, dynamic scheduling
#   cache_off__sched_off  off    off    fully naive baseline
#
# Each setting runs ONE scheduling mode per horizon (no A/B inside a run);
# the full log of every run is kept in logs/<setting>.log, and the
# "TIMING,<step>,<seconds>" lines are aggregated into logs/results.csv:
# one row per setting, one column per phase (label build, feature compute,
# task assembly, per-horizon predict, action list, total).
#
# Knobs (env): HORIZONS, CAND, K forwarded to run_tasks.sh.
# Usage:  bash run_matrix.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"
LOGDIR=logs
mkdir -p "$LOGDIR"

SETTINGS="on:on on:off off:on off:off"

for s in $SETTINGS; do
  cache=${s%%:*}
  sched=${s##*:}
  name="cache_${cache}__sched_${sched}"
  echo "=== setting $name ==="
  CACHE=$cache MODES=$sched bash run_tasks.sh 2>&1 | tee "$LOGDIR/$name.log"
done

# ---- aggregate the TIMING lines of the 4 logs into one CSV ------------------
CSV="$LOGDIR/results.csv"
echo "setting,cache,sched,cutoffs_s,label_s,cache_init_s,features_s,task_s,predict_h1_s,predict_h3_s,predict_h7_s,predict_total_s,action_list_s,total_s" > "$CSV"

for s in $SETTINGS; do
  cache=${s%%:*}
  sched=${s##*:}
  name="cache_${cache}__sched_${sched}"
  grep '^TIMING,' "$LOGDIR/$name.log" | awk -F, \
    -v setting="$name" -v cache="$cache" -v sched="$sched" '
    {
      step = $2; t = $3; total += t
      if      (step == "tool_cutoffs")        cutoffs    += t
      else if (step ~ /^01_label_/)           label      += t
      else if (step ~ /^cache_init/)          cacheinit  += t
      else if (step ~ /^02_features_/)        features   += t
      else if (step ~ /^03_task_/)            task       += t
      else if (step ~ /^08_predict_h1_/)      p1         += t
      else if (step ~ /^08_predict_h3_/)      p3         += t
      else if (step ~ /^08_predict_h7_/)      p7         += t
      else if (step == "09_action_list")      action     += t
      if      (step ~ /^08_predict_/)         predict    += t
    }
    END {
      printf "%s,%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
             setting, cache, sched, cutoffs, label, cacheinit, features, task,
             p1, p3, p7, predict, action, total
    }' >> "$CSV"
done

echo
echo "== results =="
column -s, -t "$CSV"
echo
echo "csv: $PWD/$CSV"
