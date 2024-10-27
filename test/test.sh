#!/bin/bash

DB_NAME="neurdb"
DB_USER="neurdb"
SQL_SCRIPT="$(pwd)/test.sql"
DATASET_PATH="$(pwd)/frappe.csv"

DROP_TABLE_SQL="DROP TABLE IF EXISTS frappe_test;"
CREATE_TABLE_SQL="CREATE TABLE IF NOT EXISTS frappe_test (
  click_rate INT,
  feature1 INT,
  feature2 INT,
  feature3 INT,
  feature4 INT,
  feature5 INT,
  feature6 INT,
  feature7 INT,
  feature8 INT,
  feature9 INT,
  feature10 INT
);"
COPY_TABLE_SQL="COPY frappe_test FROM '$DATASET_PATH' DELIMITER ',' CSV HEADER;"


echo "[PREPARE] Checking if Dataset file exists..."
if [ ! -f $DATASET_PATH ]; then
  echo "[Fail] Dataset file does not exist"
  exit 1
fi

echo "[PREPARE] Checking if Postgres is ready..."
pg_isready -h localhost -U $DB_USER -d $DB_NAME -t 1
if [ $? -ne 0 ]; then
  echo "[Fail] Postgres is not ready"
  exit 1
fi

echo "[PREPARE] Creating test table..."
psql -h localhost -U $DB_USER -d $DB_NAME -c "$DROP_TABLE_SQL"
psql -h localhost -U $DB_USER -d $DB_NAME -c "$CREATE_TABLE_SQL"
if [ $? -ne 0 ]; then
  echo "[Fail] Failed to create table"
  exit 1
fi

psql -h localhost -U $DB_USER -d $DB_NAME -c "$COPY_TABLE_SQL"
if [ $? -ne 0 ]; then
  echo "[Fail] Failed to copy data into table"
  exit 1
fi

echo "[TEST] Running tests..."
psql -h localhost -U $DB_USER -d $DB_NAME -f $SQL_SCRIPT
if [ $? -ne 0 ]; then
  echo "[Fail] Test failed"
  exit 1
fi

psql -h localhost -U $DB_USER -d $DB_NAME -c "$DROP_TABLE_SQL"
echo "[Success] Test passed"
