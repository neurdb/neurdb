#!/usr/bin/env bash
# One-shot: PREDICT top-k ads by predicted CTR (auto batch count).
#
# Usage (host):
#   bash test/avito/workloads/predict_topk.sh
#   H=3 CAND="categoryid = 60" K=5 bash test/avito/workloads/predict_topk.sh
#   SCHED=off bash test/avito/workloads/predict_topk.sh   # no pushdown (slower)
#
# Knobs (env):
#   CONTAINER  default neurdb_dev
#   DB         default avito
#   H          horizon / w_task_<H>  (default 1)
#   SCHED      on|off  push candidate filter below PREDICT (default on)
#   CAND       SQL predicate on input cols (default categoryid IN (60, 26, 27))
#   K          top-k (default 10)
#   BATCH      batch size (default 4096)
set -euo pipefail

CONTAINER="${CONTAINER:-neurdb_dev}"
DB="${DB:-avito}"
H="${H:-1}"
SCHED="${SCHED:-on}"
CAND="${CAND:-categoryid IN (60, 26, 27)}"
K="${K:-10}"
BATCH="${BATCH:-4096}"

DIR=/code/neurdb-dev/test/avito/workloads
PSQL="/code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -d $DB -v ON_ERROR_STOP=1"

docker exec "$CONTAINER" bash -lc \
  "$PSQL -v h=$H -v sched=$SCHED -v cand=\"$CAND\" -v k=$K -v batch=$BATCH -f $DIR/predict_topk.sql"
