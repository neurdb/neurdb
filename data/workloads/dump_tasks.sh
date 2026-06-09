#!/usr/bin/env bash
# Export the avito task tables w_task_<h> from the (in-container) avito DB to CSV
# under data/workloads/dump/ (gitignored scratch), so host-side tooling (TabPFN
# env) can read them without a live DB connection (host psql access is blocked by
# pg_hba). The committed unit-test fixture is test/avito/w_task_1.csv; point the
# test at a fresh dump with NEURDB_TASK_CSV=data/workloads/dump/w_task_<h>.csv.
# Usage:  bash dump_tasks.sh            (horizons 1 3 7)
#         HORIZONS="1" bash dump_tasks.sh
set -euo pipefail

CONTAINER=neurdb_dev
PSQL="/code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -d avito -v ON_ERROR_STOP=1"
CONTAINER_OUT=/code/neurdb-dev/data/workloads/dump
HOST_OUT=/home/worker/r/neurdb/neurdb-dev/data/workloads/dump
HORIZONS="${HORIZONS:-1 3 7}"

mkdir -p "$HOST_OUT"
docker exec "$CONTAINER" bash -lc "mkdir -p $CONTAINER_OUT"
for h in $HORIZONS; do
  docker exec "$CONTAINER" bash -lc \
    "$PSQL -c \"\\copy (SELECT * FROM w_task_$h) TO '$CONTAINER_OUT/w_task_$h.csv' WITH (FORMAT csv, HEADER true)\""
  echo "dumped w_task_$h -> $HOST_OUT/w_task_$h.csv"
done
