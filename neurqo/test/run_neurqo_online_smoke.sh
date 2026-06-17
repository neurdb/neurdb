#!/usr/bin/env bash
set -euo pipefail

PSQL_BIN="${PSQL_BIN:-/code/neurdb-dev/psql/bin/psql}"
PGHOST="${PGHOST:-127.0.0.1}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-neurdb}"
PGDATABASE="${PGDATABASE:-imdb_ori}"
QUERY_FILE="${QUERY_FILE:-/code/neurdb-dev/neurqo/test/job_2a.sql}"
TRAINER="${TRAINER:-/code/neurdb-dev/neurqo/server/online_trainer.py}"

tmpdir="$(mktemp -d /tmp/neurqo-online-smoke.XXXXXX)"
db_log="${tmpdir}/db_trajectory.jsonl"
transitions="${tmpdir}/transitions.jsonl"
query_out="${tmpdir}/query.out"

"${PSQL_BIN}" -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d "${PGDATABASE}" \
  -v ON_ERROR_STOP=1 -At >"${query_out}" <<SQL
SET neurqo.trajectory_log = '${db_log}';
SET neurqo.max_rounds = 1;
SET neurqo = on;
\\i ${QUERY_FILE}
SQL

python3 "${TRAINER}" --db-log "${db_log}" --out "${transitions}" --once >/dev/null

python3 - "${db_log}" "${transitions}" "${query_out}" <<'PY'
import json
import sys

db_log, transitions_path, query_out = sys.argv[1:4]
events = [json.loads(line) for line in open(db_log, encoding="utf-8") if line.strip()]
transitions = [
    json.loads(line)
    for line in open(transitions_path, encoding="utf-8")
    if line.strip()
]

if not events:
    raise SystemExit("no DB trajectory events were written")
first_state = events[0].get("state") or {}
if "plan_summary" not in first_state or "plan_json" not in first_state:
    raise SystemExit("trajectory state is missing plan_summary or plan_json")
if not transitions:
    raise SystemExit("online trainer wrote no transitions")
if len(transitions) > 1 and transitions[0].get("next_state") is None:
    raise SystemExit("first transition is missing next_state")

result_lines = [
    line for line in open(query_out, encoding="utf-8").read().splitlines()
    if line and line != "SET"
]
print(
    "online_smoke OK "
    f"events={len(events)} transitions={len(transitions)} "
    f"plan_nodes={first_state.get('plan_summary', {}).get('nodes')} "
    f"result={result_lines[-1] if result_lines else '<empty>'}"
)
PY
