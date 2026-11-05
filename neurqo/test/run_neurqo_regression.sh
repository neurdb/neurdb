#!/usr/bin/env bash
set -euo pipefail

PSQL_BIN="${PSQL_BIN:-/code/neurdb-dev/psql/bin/psql}"
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-neurdb}"
PGDATABASE="${PGDATABASE:-imdb_ori}"
QUERY_DIR="${QUERY_DIR:-/code/neurdb-dev/neurqo/test}"

if [ "$#" -eq 0 ]; then
  QUERIES=(job_1a job_2a job_6a job_8c job_17a job_33a)
else
  QUERIES=("$@")
fi

run_query() {
  local mode="$1"
  local sql_file="$2"

  "$PSQL_BIN" -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    -v ON_ERROR_STOP=1 -At <<SQL | sed '/^SET$/d'
SET neurqo = ${mode};
\\i ${sql_file}
SQL
}

for q in "${QUERIES[@]}"; do
  sql_file="${QUERY_DIR}/${q}.sql"
  if [ ! -f "$sql_file" ]; then
    echo "${q} MISSING ${sql_file}" >&2
    exit 1
  fi

  off="$(run_query off "$sql_file")"
  on="$(run_query on "$sql_file")"

  if [ "$off" = "$on" ]; then
    printf '%s OK %s\n' "$q" "$on"
  else
    printf '%s MISMATCH\nOFF=%s\nON=%s\n' "$q" "$off" "$on" >&2
    exit 1
  fi
done
