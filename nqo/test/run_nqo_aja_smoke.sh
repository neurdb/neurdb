#!/usr/bin/env bash
set -euo pipefail

PSQL_BIN="${PSQL_BIN:-/code/neurdb-dev/psql/bin/psql}"
PGHOST="${PGHOST:-127.0.0.1}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-neurdb}"
PGDATABASE="${PGDATABASE:-imdb_ori}"
NQO_RUNTIME_SRC="${NQO_RUNTIME_SRC:-/code/neurdb-dev/.nqo_runtime/nqo/src}"
TEST_POLICY="${TEST_POLICY:-/code/neurdb-dev/nqo/test/aja_test_policy.py:predict}"
AI_PORT="${AI_PORT:-18089}"

tmpdir="$(mktemp -d /tmp/nqo-aja-smoke.XXXXXX)"
server_log="${tmpdir}/ai_server.log"
query_log="${tmpdir}/query.log"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    rm -rf "${tmpdir}"
}
trap cleanup EXIT

PYTHONPATH="${NQO_RUNTIME_SRC}" python3 -m runtime.action_server \
    --host 127.0.0.1 \
    --port "${AI_PORT}" \
    --model-module "${TEST_POLICY}" \
    --require-model >"${server_log}" 2>&1 &
server_pid=$!

for _ in $(seq 1 50); do
    if curl -fsS "http://127.0.0.1:${AI_PORT}/" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
curl -fsS "http://127.0.0.1:${AI_PORT}/" >/dev/null

"${PSQL_BIN}" \
    -h "${PGHOST}" \
    -p "${PGPORT}" \
    -U "${PGUSER}" \
    -d "${PGDATABASE}" \
    -v ON_ERROR_STOP=1 >"${query_log}" 2>&1 <<SQL
SET client_min_messages = log;
CREATE TEMP TABLE nqo_aja_probe (k integer PRIMARY KEY, payload integer);
CREATE TEMP TABLE nqo_aja_probe_2 (k integer PRIMARY KEY, payload integer);
CREATE TEMP TABLE nqo_aja_build (k integer, payload integer);
INSERT INTO nqo_aja_probe
SELECT i, i FROM generate_series(1, 100000) AS g(i);
INSERT INTO nqo_aja_probe_2
SELECT i, i FROM generate_series(1, 100000) AS g(i);
INSERT INTO nqo_aja_build
SELECT i, i FROM generate_series(1, 10000) AS g(i);
ANALYZE nqo_aja_probe;
ANALYZE nqo_aja_probe_2;
ANALYZE nqo_aja_build;
DELETE FROM nqo_aja_build WHERE k > 10;

SET nqo.server_url = 'http://127.0.0.1:${AI_PORT}/action';
SET nqo.max_rounds = 0;
SET enable_mergejoin = off;
SET nqo.aja_max_nestloop_cost_ratio_pct = 0;
SET nqo.aja_aggressive_max_nestloop_cost_ratio_pct = 0;
SET nqo.aja_conservative_rows = 5;
SET nqo = on;
SELECT count(*)::text || ':' ||
       sum(build.payload + probe.payload)::text AS result
FROM nqo_aja_probe AS probe
JOIN nqo_aja_build AS build ON build.k = probe.k;

SET nqo.aja_conservative_rows = 100;
SELECT count(*)::text || ':' ||
       sum(build.payload + probe.payload)::text AS result
FROM nqo_aja_probe AS probe
JOIN nqo_aja_build AS build ON build.k = probe.k;

SET nqo.aja_conservative_rows = 5;
SET nqo.aja_aggressive_rows = 100;
SELECT count(*)::text || ':' ||
       sum(build_aggressive_case.payload + probe.payload)::text AS result
FROM nqo_aja_probe AS probe
JOIN nqo_aja_build AS build_aggressive_case
  ON build_aggressive_case.k = probe.k;

SET nqo.aja_conservative_rows = 100;
SELECT count(*)::text || ':' ||
       sum(build.payload + probe.payload + probe_2.payload)::text AS result
FROM nqo_aja_build AS build
JOIN nqo_aja_probe AS probe ON probe.k = build.k
JOIN nqo_aja_probe_2 AS probe_2 ON probe_2.k = build.k;
SQL

if [[ "$(grep -c '^ *10:110 *$' "${query_log}")" -ne 3 ]]; then
    cat "${query_log}"
    echo "AJA smoke returned an unexpected query result" >&2
    exit 1
fi
if [[ "$(grep -c '^ *10:165 *$' "${query_log}")" -ne 1 ]]; then
    cat "${query_log}"
    echo "AJA smoke returned an unexpected multi-join result" >&2
    exit 1
fi
if ! grep -q 'adaptive join planning.*wrapped=1' "${query_log}"; then
    cat "${query_log}"
    echo "AJA smoke did not create an adaptive join node" >&2
    exit 1
fi
if ! grep -q 'adaptive join planning.*wrapped=2' "${query_log}"; then
    cat "${query_log}"
    echo "AJA smoke did not wrap both eligible HashJoin nodes" >&2
    exit 1
fi
if ! grep -q 'actual_build_rows=10 .*threshold_rows=100 .*selected=NestLoop' \
    "${query_log}"; then
    cat "${query_log}"
    echo "AJA smoke did not select NestLoop below the threshold" >&2
    exit 1
fi
if ! grep -q 'level=aggressive .*actual_build_rows=10 .*threshold_rows=100 .*selected=NestLoop' \
    "${query_log}"; then
    cat "${query_log}"
    echo "AJA smoke did not apply the aggressive threshold" >&2
    exit 1
fi
if ! grep -q 'actual_build_rows=10 .*threshold_rows=5 .*selected=HashJoin' \
    "${query_log}"; then
    cat "${query_log}"
    echo "AJA smoke did not retain HashJoin above the threshold" >&2
    exit 1
fi

echo "aja_smoke OK actual_rows=10 single=10:110 multi=10:165 conservative=Hash/Nest aggressive=Nest"
