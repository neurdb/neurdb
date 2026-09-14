#!/usr/bin/env bash
# End-to-end TabPFN PREDICT smoke test (run INSIDE the neurdb_dev container).
#
#   docker exec neurdb_dev bash /code/neurdb-dev/script/experiment/db-26/run_tabpfn_predict.sh
#
# Env knobs:
#   DB        target database          (default: avito)
#   LIMIT     rows pulled from w_task_1 (default: 1000)
#   BATCH     nr_task_batch_size        (default: 500)
#   NBATCH    nr_task_num_batches       (default: ceil(LIMIT/BATCH))
#   TARGET    regression target column  (default: label_ctr)
#   SRC       source table              (default: w_task_1)
set -euo pipefail

DB="${DB:-avito}"
LIMIT="${LIMIT:-1000}"
BATCH="${BATCH:-500}"
TARGET="${TARGET:-label_ctr}"
SRC="${SRC:-w_task_1}"
NBATCH="${NBATCH:-$(( (LIMIT + BATCH - 1) / BATCH ))}"

PSQL="/code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -d ${DB}"

echo "== ensure nr_pipeline extension + bookkeeping tables =="
$PSQL -v ON_ERROR_STOP=1 <<SQL
CREATE EXTENSION IF NOT EXISTS nr_pipeline;
-- model bookkeeping tables (the AI engine creates these in the 'neurdb' DB, but
-- PREDICT runs in this DB and the C extension queries router here). For tabpfn
-- the engine skips router registration, so this stays empty => always (re)train.
CREATE TABLE IF NOT EXISTS model (model_id SERIAL PRIMARY KEY, model_meta BYTEA);
CREATE TABLE IF NOT EXISTS layer (model_id INT, layer_id INT, create_time TIMESTAMP, layer_data BYTEA);
CREATE TABLE IF NOT EXISTS router (model_id INT, table_name TEXT, feature_columns TEXT, target_columns TEXT);
SQL

echo "== register AI engine (127.0.0.1:8090) =="
$PSQL -v ON_ERROR_STOP=1 <<SQL
DELETE FROM pg_catalog.nr_aiengine;
SELECT insert_ai_engine('127.0.0.1', 8090);
SELECT * FROM pg_catalog.nr_aiengine;
SQL

echo "== PREDICT VALUE OF ${TARGET} (rows=${LIMIT} batch=${BATCH} nbatch=${NBATCH}) =="
OUT="$($PSQL -v ON_ERROR_STOP=1 <<SQL 2>&1
SET nr_task_batch_size TO ${BATCH};
SET nr_task_num_batches TO ${NBATCH};
SET nr_task_epoch TO 1;
\timing on
PREDICT VALUE OF ${TARGET}
  FROM (SELECT * FROM ${SRC} LIMIT ${LIMIT}) AS t
  TRAIN tabpfn ON *;
SQL
)"
# show the head of the result, then a row count (avoids SIGPIPE from `head`)
echo "$OUT" | grep -vE '^[[:space:]]*-?[0-9.]+[[:space:]]*$' | head -20
echo "  ... predicted rows: $(echo "$OUT" | grep -cE '^[[:space:]]*-?[0-9]+\.[0-9]')"
echo "$OUT" | grep -iE 'ERROR|Time:' | head -5
echo "== done =="
