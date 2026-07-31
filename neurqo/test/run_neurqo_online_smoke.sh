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
if first_state.get("request_type") != "high":
    raise SystemExit("trajectory does not start with a high-phase state")
split_events = [event for event in events if event.get("phase") == "split"]
if not split_events:
    raise SystemExit("smoke query did not exercise the split path")
for index, event in enumerate(events):
    states = event.get("decision_states") or {}
    if event.get("phase") == "split":
        for phase in ("select", "search", "low"):
            if not isinstance(states.get(phase), dict):
                raise SystemExit(f"split event is missing the {phase} state")
        action = event.get("action") or {}
        candidate_id = action.get("candidate_id")
        if candidate_id is None:
            raise SystemExit("split event is missing selected candidate_id")
        if not action.get("selection_strategy"):
            raise SystemExit("split event is missing selection_strategy")
        candidates = states["select"].get("candidates") or []
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("candidate_id") == candidate_id
            ),
            None,
        )
        if selected is None:
            raise SystemExit("selected candidate_id is absent from Select state")
        if selected.get("sql") != states["search"].get("sql"):
            raise SystemExit("Search did not receive the selected candidate SQL")
        if states["search"].get("sql") != states["low"].get("sql"):
            raise SystemExit("Low plan does not belong to the Search execution SQL")
        if index + 1 >= len(events):
            raise SystemExit("split event has no next residual round")
        result_name = event.get("result")
        next_high = (events[index + 1].get("decision_states") or {}).get("high") or {}
        if not result_name or result_name not in str(next_high.get("sql") or ""):
            raise SystemExit("next High state does not contain the materialized result")
final_event = next((event for event in events if event.get("phase") == "final"), None)
if final_event is None:
    raise SystemExit("trajectory is missing a final event")
final_states = final_event.get("decision_states") or {}
if final_states.get("select") is not None:
    raise SystemExit("final stop path must not invoke Select")
if not isinstance(final_states.get("search"), dict):
    raise SystemExit("final event is missing the Search state")
low_state = final_states.get("low") or {}
if "plan_summary" not in low_state or "plan_json" not in low_state:
    raise SystemExit("Low state is missing the Search-selected plan")
if not transitions:
    raise SystemExit("online trainer wrote no transitions")
if "decision_states" not in transitions[-1]:
    raise SystemExit("online transition is missing phase decision states")
if len(transitions) > 1 and transitions[0].get("next_state") is None:
    raise SystemExit("first transition is missing next_state")

result_lines = [
    line for line in open(query_out, encoding="utf-8").read().splitlines()
    if line and line != "SET"
]
print(
    "online_smoke OK "
    f"events={len(events)} transitions={len(transitions)} "
    f"plan_nodes={low_state.get('plan_summary', {}).get('nodes')} "
    f"result={result_lines[-1] if result_lines else '<empty>'}"
)
PY
