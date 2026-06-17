#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER="${CONTAINER:-neurdb_dev_opt}"
CONTAINER_REPO="${CONTAINER_REPO:-/code/neurdb-dev}"
RUNTIME_HOST="${RUNTIME_HOST:-${REPO_ROOT}/.neurqo_runtime}"
RUNTIME_CONTAINER="${RUNTIME_CONTAINER:-${CONTAINER_REPO}/.neurqo_runtime}"

NEURQO_SRC_HOST="${NEURQO_SRC_HOST:-/home/naili/neurqo}"
MODEL_PATH="${MODEL_PATH:-${NEURQO_SRC_HOST}/artifacts/models/JOB_standardmdp_rl_random_a_seed42_transfer_v2_job_random_a_latest.pt}"
MODEL_METHOD="${MODEL_METHOD:-standardmdp_rl}"
MODEL_HIDDEN="${MODEL_HIDDEN:-128}"
WORKLOAD="${WORKLOAD:-job}"
DEVICE="${DEVICE:-cpu}"

AI_HOST="${AI_HOST:-127.0.0.1}"
AI_PORT="${AI_PORT:-8088}"
SERVER_TIMEOUT_MS="${SERVER_TIMEOUT_MS:-5000}"
MAX_ROUNDS="${MAX_ROUNDS:-64}"
SEARCH_TOPK="${SEARCH_TOPK:-5}"
SEARCH_MAX_RELS="${SEARCH_MAX_RELS:-12}"
KEEP_SERVER="${KEEP_SERVER:-0}"
ONLINE_TRAIN="${ONLINE_TRAIN:-1}"
ONLINE_LEARNING_RATE="${ONLINE_LEARNING_RATE:-1e-5}"
ONLINE_EPOCHS="${ONLINE_EPOCHS:-1}"
RELOAD_UPDATED_MODEL="${RELOAD_UPDATED_MODEL:-1}"
POST_TRAIN_REPLAY="${POST_TRAIN_REPLAY:-0}"
ONLINE_EXPERIMENT_ROUNDS="${ONLINE_EXPERIMENT_ROUNDS:-1}"
PROMOTION_GUARD="${PROMOTION_GUARD:-1}"
PROMOTION_MIN_RELATIVE_SPEEDUP="${PROMOTION_MIN_RELATIVE_SPEEDUP:-1.0}"
PROMOTION_MIN_BASELINE_SPEEDUP="${PROMOTION_MIN_BASELINE_SPEEDUP:-0.0}"
FALLBACK_MIN_SPEEDUP="${FALLBACK_MIN_SPEEDUP:-1.0}"

PSQL_BIN="${PSQL_BIN:-${CONTAINER_REPO}/psql/bin/psql}"
PGHOST="${PGHOST:-127.0.0.1}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-neurdb}"
PGDATABASE="${PGDATABASE:-imdb_ori}"
QUERY_DIR_CONTAINER="${QUERY_DIR_CONTAINER:-${CONTAINER_REPO}/neurqo/test}"
QUERIES="${QUERIES:-job_2a}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_HOST="${RUNTIME_HOST}/runs/${RUN_ID}"
RUN_CONTAINER="${RUNTIME_CONTAINER}/runs/${RUN_ID}"
MIRROR_HOST="${RUNTIME_HOST}/neurqo"
MIRROR_CONTAINER="${RUNTIME_CONTAINER}/neurqo"
MODEL_BASENAME="$(basename "${MODEL_PATH}")"
MODEL_CONTAINER="${MIRROR_CONTAINER}/artifacts/models/${MODEL_BASENAME}"
UPDATED_MODEL_BASENAME="${UPDATED_MODEL_BASENAME:-${MODEL_BASENAME%.pt}.online.pt}"
ACTIVE_MODEL_CONTAINER="${MODEL_CONTAINER}"
UPDATED_MODEL_CONTAINER="${RUN_CONTAINER}/${UPDATED_MODEL_BASENAME}"

SERVER_STDOUT_CONTAINER="${RUN_CONTAINER}/ai_server.stdout.log"
SERVER_DECISIONS_CONTAINER="${RUN_CONTAINER}/ai_server_decisions.jsonl"
DB_TRAJECTORY_CONTAINER="${RUN_CONTAINER}/db_trajectory.jsonl"
TRANSITIONS_CONTAINER="${RUN_CONTAINER}/transitions.jsonl"
MODEL_UPDATE_METADATA_CONTAINER="${RUN_CONTAINER}/model_update.json"
MODEL_RELOAD_OUT_CONTAINER="${RUN_CONTAINER}/model_reload.out"
PROMOTION_DECISION_CONTAINER="${RUN_CONTAINER}/promotion.json"
SERVER_PID_FILE="/tmp/neurqo_ai_server_online.pid"

die() {
  echo "error: $*" >&2
  exit 1
}

docker_exec() {
  docker exec -u neurdb -w "${CONTAINER_REPO}" "${CONTAINER}" bash -lc "$*"
}

sync_neurqo_runtime() {
  [ -d "${NEURQO_SRC_HOST}/src/model/hrl" ] || die "missing NeurQO src: ${NEURQO_SRC_HOST}"
  [ -f "${NEURQO_SRC_HOST}/config/${WORKLOAD}/catalog.json" ] || die "missing catalog for workload=${WORKLOAD}"
  [ -f "${MODEL_PATH}" ] || die "missing model checkpoint: ${MODEL_PATH}"

  mkdir -p "${MIRROR_HOST}" "${MIRROR_HOST}/artifacts/models" "${RUN_HOST}"
  rm -rf "${MIRROR_HOST}/src" "${MIRROR_HOST}/config" "${MIRROR_HOST}/tools" "${MIRROR_HOST}/workloads"
  rm -rf "${MIRROR_HOST}/results"
  cp -a "${NEURQO_SRC_HOST}/src" "${MIRROR_HOST}/src"
  cp -a "${NEURQO_SRC_HOST}/config" "${MIRROR_HOST}/config"
  cp -a "${NEURQO_SRC_HOST}/tools" "${MIRROR_HOST}/tools"
  mkdir -p "${MIRROR_HOST}/workloads"
  cp -a "${NEURQO_SRC_HOST}/workloads/train_test.py" "${MIRROR_HOST}/workloads/train_test.py"
  touch "${MIRROR_HOST}/workloads/__init__.py"
  mkdir -p "${MIRROR_HOST}/results/raw/job" "${MIRROR_HOST}/results/raw/stack" "${MIRROR_HOST}/results/raw/tpch"
  cp -L "${NEURQO_SRC_HOST}/results/raw/job/job_step_action_times.csv" "${MIRROR_HOST}/results/raw/job/job_step_action_times.csv"
  cp -L "${NEURQO_SRC_HOST}/results/raw/stack/stack_step_action_times.csv" "${MIRROR_HOST}/results/raw/stack/stack_step_action_times.csv"
  cp -L "${NEURQO_SRC_HOST}/results/raw/tpch/tpch_step_action_times.csv" "${MIRROR_HOST}/results/raw/tpch/tpch_step_action_times.csv"
  cp -f "${MODEL_PATH}" "${MIRROR_HOST}/artifacts/models/${MODEL_BASENAME}"
  find "${MIRROR_HOST}" -type d -name __pycache__ -prune -exec rm -rf {} +
}

stop_server() {
  docker_exec "if [ -f '${SERVER_PID_FILE}' ]; then pid=\$(cat '${SERVER_PID_FILE}' 2>/dev/null || true); if [ -n \"\$pid\" ]; then kill \"\$pid\" 2>/dev/null || true; fi; rm -f '${SERVER_PID_FILE}'; fi; python3 - <<'PY'
import os
import signal

ancestors = set()
pid = os.getpid()
while pid > 1:
    ancestors.add(pid)
    try:
        with open(f'/proc/{pid}/stat', encoding='utf-8') as f:
            stat = f.read().split()
        pid = int(stat[3])
    except Exception:
        break

for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    proc_pid = int(name)
    if proc_pid in ancestors:
        continue
    try:
        with open(f'/proc/{proc_pid}/cmdline', 'rb') as f:
            cmd = f.read().replace(b'\\x00', b' ').decode('utf-8', 'ignore')
    except Exception:
        continue
    if 'ai_server.py' in cmd and '/neurqo/server/' in cmd:
        try:
            os.kill(proc_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
PY"
}

start_server() {
  docker_exec "mkdir -p '${RUN_CONTAINER}'"
  stop_server
  docker_exec "nohup python3 '${CONTAINER_REPO}/neurqo/server/ai_server.py' \
      --host '${AI_HOST}' \
      --port '${AI_PORT}' \
      --model-path '${MODEL_CONTAINER}' \
      --model-method '${MODEL_METHOD}' \
      --model-hidden '${MODEL_HIDDEN}' \
      --workload '${WORKLOAD}' \
      --device '${DEVICE}' \
      --neurqo-src '${MIRROR_CONTAINER}/src' \
      --trajectory-log '${SERVER_DECISIONS_CONTAINER}' \
      --require-model \
      > '${SERVER_STDOUT_CONTAINER}' 2>&1 & echo \$! > '${SERVER_PID_FILE}'"

  for _ in $(seq 1 40); do
    if docker_exec "pid=\$(cat '${SERVER_PID_FILE}' 2>/dev/null || true); [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null" &&
      docker_exec "python3 - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen('http://${AI_HOST}:${AI_PORT}/', timeout=1).read()
PY"; then
      return 0
    fi
    sleep 0.25
  done

  docker_exec "cat '${SERVER_STDOUT_CONTAINER}' 2>/dev/null || true" >&2
  die "AI server did not become healthy"
}

reload_server_model() {
  local model_path="$1"
  local out_path="$2"
  docker_exec "python3 - <<PY
import json
import urllib.request

payload = json.dumps({'model_path': '${model_path}'}).encode('utf-8')
req = urllib.request.Request(
    'http://${AI_HOST}:${AI_PORT}/reload',
    data=payload,
    headers={'Content-Type': 'application/json'},
    method='POST',
)
print(urllib.request.urlopen(req, timeout=30).read().decode('utf-8'), end='')
PY" >"${out_path}"
}

resolve_query_path() {
  local q="$1"
  if [[ "${q}" = /* ]]; then
    printf '%s\n' "${q}"
  elif [[ -f "${REPO_ROOT}/neurqo/test/${q}.sql" ]]; then
    printf '%s/%s.sql\n' "${QUERY_DIR_CONTAINER}" "${q}"
  elif [[ -f "${REPO_ROOT}/neurqo/test/job_${q}.sql" ]]; then
    printf '%s/job_%s.sql\n' "${QUERY_DIR_CONTAINER}" "${q}"
  else
    die "cannot resolve query ${q}; use a container SQL path or a file in neurqo/test"
  fi
}

run_one_query() {
  local q="$1"
  local query_path="$2"
  local mode="$3"
  local out_file="$4"
  local elapsed_file="$5"
  local status_file="$6"
  local start_ns
  local end_ns
  local rc=0

  start_ns="$(date +%s%N)"
  if ! docker exec -i -u neurdb -w "${CONTAINER_REPO}" "${CONTAINER}" bash -lc "'${PSQL_BIN}' -h '${PGHOST}' -p '${PGPORT}' -U '${PGUSER}' -d '${PGDATABASE}' -v ON_ERROR_STOP=1 -At" >"${out_file}" 2>&1 <<SQL
SET neurqo.server_url = 'http://${AI_HOST}:${AI_PORT}/action';
SET neurqo.server_timeout_ms = ${SERVER_TIMEOUT_MS};
SET neurqo.max_rounds = ${MAX_ROUNDS};
SET neurqo.search_topk = ${SEARCH_TOPK};
SET neurqo.search_max_rels = ${SEARCH_MAX_RELS};
SET neurqo.trajectory_log = '${DB_TRAJECTORY_CONTAINER}';
SET neurqo = ${mode};
\\i ${query_path}
SQL
  then
    rc=$?
  fi
  end_ns="$(date +%s%N)"
  printf '%s\n' "$(((end_ns - start_ns) / 1000000))" >"${elapsed_file}"
  printf '%s\n' "${rc}" >"${status_file}"
  sed -i '/^SET$/d' "${out_file}"
}

set_round_paths() {
  local round_no="$1"
  local prefix=""

  if [ "${ONLINE_EXPERIMENT_ROUNDS}" -gt 1 ]; then
    prefix="round_${round_no}."
    UPDATED_MODEL_CONTAINER="${RUN_CONTAINER}/${UPDATED_MODEL_BASENAME%.pt}.round_${round_no}.pt"
  else
    UPDATED_MODEL_CONTAINER="${RUN_CONTAINER}/${UPDATED_MODEL_BASENAME}"
  fi

  DB_TRAJECTORY_CONTAINER="${RUN_CONTAINER}/${prefix}db_trajectory.jsonl"
  TRANSITIONS_CONTAINER="${RUN_CONTAINER}/${prefix}transitions.jsonl"
  MODEL_UPDATE_METADATA_CONTAINER="${RUN_CONTAINER}/${prefix}model_update.json"
  MODEL_RELOAD_OUT_CONTAINER="${RUN_CONTAINER}/${prefix}model_reload.out"
  PROMOTION_DECISION_CONTAINER="${RUN_CONTAINER}/${prefix}promotion.json"
}

round_file_prefix() {
  local round_no="$1"
  if [ "${ONLINE_EXPERIMENT_ROUNDS}" -gt 1 ]; then
    printf 'round_%s.' "${round_no}"
  fi
}

write_promotion_decision() {
  local round_no="$1"
  local previous_model="$2"
  local candidate_model="$3"
  local decision_path="${RUN_HOST}/$(basename "${PROMOTION_DECISION_CONTAINER}")"

  python3 - \
    "${RUN_HOST}/timing.csv" \
    "${decision_path}" \
    "${round_no}" \
    "${QUERIES}" \
    "${PROMOTION_MIN_RELATIVE_SPEEDUP}" \
    "${PROMOTION_MIN_BASELINE_SPEEDUP}" \
    "${previous_model}" \
    "${candidate_model}" <<'PY'
import csv
import json
import sys
from pathlib import Path

timing_path = Path(sys.argv[1])
decision_path = Path(sys.argv[2])
round_no = sys.argv[3]
queries = sys.argv[4].split()
min_relative = float(sys.argv[5])
min_baseline = float(sys.argv[6])
previous_model = sys.argv[7]
candidate_model = sys.argv[8]

rows = list(csv.DictReader(timing_path.open())) if timing_path.exists() else []
round_rows = [r for r in rows if str(r.get("round", "1")) == str(round_no)]

def elapsed(row):
    try:
        return float(row["elapsed_ms"])
    except Exception:
        return None

def ok(row):
    return row is not None and str(row.get("status")) == "0"

def same_output(a, b):
    if not a or not b:
        return False
    try:
        return Path(a).read_bytes() == Path(b).read_bytes()
    except Exception:
        return False

by_query = {}
for row in round_rows:
    by_query.setdefault(row["query"], {})[row["mode"]] = row

per_query = {}
off_total = 0.0
on_total = 0.0
after_total = 0.0
candidate_complete = True
current_complete = True

for query in queries:
    modes = by_query.get(query, {})
    off = modes.get("off")
    on = modes.get("on")
    after = modes.get("on_after_update")
    off_ms = elapsed(off)
    on_ms = elapsed(on)
    after_ms = elapsed(after)
    off_total += off_ms or 0.0
    on_total += on_ms or 0.0
    after_total += after_ms or 0.0
    current_match = ok(off) and ok(on) and same_output(
        off.get("output_file") if off else None,
        on.get("output_file") if on else None,
    )
    candidate_match = ok(off) and ok(after) and same_output(
        off.get("output_file") if off else None,
        after.get("output_file") if after else None,
    )
    if not current_match:
        current_complete = False
    if not candidate_match:
        candidate_complete = False
    per_query[query] = {
        "off_ms": off_ms,
        "on_ms": on_ms,
        "on_after_update_ms": after_ms,
        "on_matches_off": current_match,
        "candidate_matches_off": candidate_match,
        "candidate_relative_speedup_vs_on": (
            on_ms / after_ms if on_ms and after_ms and after_ms > 0 else None
        ),
        "candidate_speedup_vs_off": (
            off_ms / after_ms if off_ms and after_ms and after_ms > 0 else None
        ),
    }

relative_speedup = on_total / after_total if on_total > 0 and after_total > 0 else 0.0
baseline_speedup = (
    off_total / after_total if off_total > 0 and after_total > 0 else 0.0
)

reasons = []
if not candidate_complete:
    reasons.append("candidate_output_mismatch_or_failed")
if relative_speedup < min_relative:
    reasons.append(
        f"candidate_relative_speedup {relative_speedup:.4f} < {min_relative:.4f}"
    )
if baseline_speedup < min_baseline:
    reasons.append(
        f"candidate_baseline_speedup {baseline_speedup:.4f} < {min_baseline:.4f}"
    )

accepted = not reasons
payload = {
    "round": int(round_no),
    "accepted": accepted,
    "reason": "accepted" if accepted else "; ".join(reasons),
    "previous_model": previous_model,
    "candidate_model": candidate_model,
    "metrics": {
        "off_total_ms": off_total,
        "on_total_ms": on_total,
        "on_after_update_total_ms": after_total,
        "current_outputs_match": current_complete,
        "candidate_outputs_match": candidate_complete,
        "candidate_relative_speedup_vs_on": relative_speedup,
        "candidate_speedup_vs_off": baseline_speedup,
        "min_relative_speedup": min_relative,
        "min_baseline_speedup": min_baseline,
    },
    "per_query": per_query,
}
decision_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print("1" if accepted else "0")
PY
}

write_summary() {
  python3 - "${RUN_HOST}" "${QUERIES}" "${FALLBACK_MIN_SPEEDUP}" <<'PY'
from collections import Counter
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
queries = sys.argv[2].split()
fallback_min_speedup = float(sys.argv[3])
timing_path = run_dir / "timing.csv"
rows = list(csv.DictReader(timing_path.open())) if timing_path.exists() else []

def count_jsonl(path):
    if not path.exists():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())

def count_jsonl_many(pattern):
    return sum(count_jsonl(path) for path in run_dir.glob(pattern))

def count_server_phase(path, phase):
    if not path.exists():
        return 0
    total = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        if json.loads(line).get("phase") == phase:
            total += 1
    return total

def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def load_text(path):
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()

def status_ok(row):
    return row and str(row.get("status")) == "0"

def elapsed(row):
    try:
        return float(row["elapsed_ms"])
    except Exception:
        return None

def same_output(a, b):
    if not a or not b:
        return False
    try:
        return Path(a).read_bytes() == Path(b).read_bytes()
    except Exception:
        return False

rounds = sorted({str(row.get("round", "1")) for row in rows}, key=lambda x: int(x))
per_round = {}
for row in rows:
    per_round.setdefault(str(row.get("round", "1")), {}).setdefault(
        "_rows", []
    ).append(row)

for round_no in rounds:
    round_rows = per_round[round_no].pop("_rows", [])
    by_query = {}
    for row in round_rows:
        by_query.setdefault(row["query"], {})[row["mode"]] = row

    off_total = 0.0
    on_total = 0.0
    effective_total = 0.0
    fallback_queries = []
    per_query = {}
    for query in queries:
        modes = by_query.get(query, {})
        off = modes.get("off")
        off_ms = elapsed(off) if status_ok(off) else None
        entry = {}
        for mode, row in sorted(modes.items()):
            ms = elapsed(row)
            speedup = off_ms / ms if off_ms and ms and ms > 0 else None
            match = same_output(
                off.get("output_file") if off else None,
                row.get("output_file") if row else None,
            )
            entry[mode] = {
                "status": int(row.get("status", -1)),
                "elapsed_ms": ms,
                "speedup_vs_off": speedup,
                "matches_off": match,
                "output_file": row.get("output_file"),
            }

        on = modes.get("on")
        on_ms = elapsed(on) if status_ok(on) else None
        on_speedup = off_ms / on_ms if off_ms and on_ms and on_ms > 0 else None
        on_match = same_output(
            off.get("output_file") if off else None,
            on.get("output_file") if on else None,
        )
        use_online = (
            off_ms is not None
            and on_ms is not None
            and on_match
            and on_speedup is not None
            and on_speedup >= fallback_min_speedup
        )
        if off_ms is not None:
            off_total += off_ms
        if on_ms is not None:
            on_total += on_ms
        if use_online:
            effective_total += on_ms
        elif off_ms is not None:
            effective_total += off_ms
            fallback_queries.append(query)

        per_query[query] = entry

    per_round[round_no] = {
        "per_query": per_query,
        "totals": {
            "off_ms": off_total,
            "on_ms": on_total,
            "effective_ms_with_fallback_guard": effective_total,
            "online_speedup_vs_off": (
                off_total / on_total if off_total > 0 and on_total > 0 else None
            ),
            "effective_speedup_vs_off": (
                off_total / effective_total
                if off_total > 0 and effective_total > 0
                else None
            ),
            "fallback_min_speedup": fallback_min_speedup,
            "fallback_queries": fallback_queries,
        },
    }

action_counts = {
    "model_source": Counter(),
    "high_action": Counter(),
    "round_action": Counter(),
    "search_label": Counter(),
    "search_strategy": Counter(),
    "low_label": Counter(),
    "lip_action": Counter(),
    "execution_action": Counter(),
}
decision_path = run_dir / "ai_server_decisions.jsonl"
if decision_path.exists():
    for line in decision_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("phase") != "policy_decision":
            continue
        action = event.get("action") or {}
        for key in action_counts:
            action_key = "action" if key == "round_action" else key
            value = action.get(action_key)
            if value is not None:
                action_counts[key][str(value)] += 1

promotion_decisions = []
for path in sorted(run_dir.glob("*promotion.json")):
    data = load_json(path)
    if data is not None:
        data["file"] = str(path)
        promotion_decisions.append(data)

model_updates = []
for path in sorted(run_dir.glob("*model_update.json")):
    data = load_json(path)
    if data is not None:
        data["file"] = str(path)
        model_updates.append(data)

model_reload_logs = {
    path.name: load_text(path)
    for path in sorted(list(run_dir.glob("*model_reload.out")) + list(run_dir.glob("*model_restore.out")))
}

summary = {
    "run_dir": str(run_dir),
    "queries": queries,
    "timing": rows,
    "rounds": rounds,
    "per_round": per_round,
    "per_query": per_round.get(rounds[0], {}).get("per_query", {}) if rounds else {},
    "db_trajectory_events": count_jsonl_many("*db_trajectory.jsonl"),
    "server_log_events": count_jsonl(run_dir / "ai_server_decisions.jsonl"),
    "server_decisions": count_server_phase(
        run_dir / "ai_server_decisions.jsonl", "policy_decision"
    ),
    "transitions": count_jsonl_many("*transitions.jsonl"),
    "action_counts": {k: dict(v) for k, v in action_counts.items()},
    "model_update": load_json(run_dir / "model_update.json"),
    "model_updates": model_updates,
    "model_reload": load_text(run_dir / "model_reload.out"),
    "model_reload_logs": model_reload_logs,
    "promotion_decisions": promotion_decisions,
    "compare": load_text(run_dir / "compare.txt"),
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
}

main() {
  [ "${ONLINE_EXPERIMENT_ROUNDS}" -ge 1 ] || die "ONLINE_EXPERIMENT_ROUNDS must be >= 1"

  mkdir -p "${RUN_HOST}"
  sync_neurqo_runtime
  start_server
  if [ "${KEEP_SERVER}" != "1" ]; then
    trap stop_server EXIT
  fi

  local timing_csv="${RUN_HOST}/timing.csv"
  printf 'round,query,mode,status,elapsed_ms,output_file\n' >"${timing_csv}"
  : >"${RUN_HOST}/compare.txt"

  local experiment_round
  for experiment_round in $(seq 1 "${ONLINE_EXPERIMENT_ROUNDS}"); do
    local prefix
    local previous_model
    local eval_candidate
    set_round_paths "${experiment_round}"
    prefix="$(round_file_prefix "${experiment_round}")"
    printf 'round %s active_model=%s\n' "${experiment_round}" "${ACTIVE_MODEL_CONTAINER}" | tee -a "${RUN_HOST}/rounds.log"

    for q in ${QUERIES}; do
      local query_path
      local off_out
      local on_out
      local off_ms
      local on_ms
      local off_status
      local on_status
      query_path="$(resolve_query_path "${q}")"
      off_out="${RUN_HOST}/${prefix}${q}.off.out"
      on_out="${RUN_HOST}/${prefix}${q}.on.out"
      off_ms="${RUN_HOST}/${prefix}${q}.off.ms"
      on_ms="${RUN_HOST}/${prefix}${q}.on.ms"
      off_status="${RUN_HOST}/${prefix}${q}.off.status"
      on_status="${RUN_HOST}/${prefix}${q}.on.status"

      run_one_query "${q}" "${query_path}" "off" "${off_out}" "${off_ms}" "${off_status}"
      run_one_query "${q}" "${query_path}" "on" "${on_out}" "${on_ms}" "${on_status}"

      printf '%s,%s,off,%s,%s,%s\n' "${experiment_round}" "${q}" "$(cat "${off_status}")" "$(cat "${off_ms}")" "${off_out}" >>"${timing_csv}"
      printf '%s,%s,on,%s,%s,%s\n' "${experiment_round}" "${q}" "$(cat "${on_status}")" "$(cat "${on_ms}")" "${on_out}" >>"${timing_csv}"
      if cmp -s "${off_out}" "${on_out}"; then
        printf 'round_%s %s MATCH\n' "${experiment_round}" "${q}" | tee -a "${RUN_HOST}/compare.txt"
      else
        printf 'round_%s %s MISMATCH\n' "${experiment_round}" "${q}" | tee -a "${RUN_HOST}/compare.txt"
        diff -u "${off_out}" "${on_out}" || true
      fi
    done

    if [ "${ONLINE_TRAIN}" = "1" ]; then
      previous_model="${ACTIVE_MODEL_CONTAINER}"
      docker_exec "python3 '${CONTAINER_REPO}/neurqo/server/online_trainer.py' \
        --db-log '${DB_TRAJECTORY_CONTAINER}' \
        --out '${TRANSITIONS_CONTAINER}' \
        --model-path '${ACTIVE_MODEL_CONTAINER}' \
        --updated-model-path '${UPDATED_MODEL_CONTAINER}' \
        --metadata-out '${MODEL_UPDATE_METADATA_CONTAINER}' \
        --model-method '${MODEL_METHOD}' \
        --model-hidden '${MODEL_HIDDEN}' \
        --workload '${WORKLOAD}' \
        --device '${DEVICE}' \
        --neurqo-src '${MIRROR_CONTAINER}/src' \
        --learning-rate '${ONLINE_LEARNING_RATE}' \
        --epochs '${ONLINE_EPOCHS}' \
        --once"

      eval_candidate=0
      if [ "${PROMOTION_GUARD}" = "1" ] || [ "${POST_TRAIN_REPLAY}" = "1" ]; then
        eval_candidate=1
      fi

      if [ "${RELOAD_UPDATED_MODEL}" = "1" ] || [ "${eval_candidate}" = "1" ]; then
        reload_server_model "${UPDATED_MODEL_CONTAINER}" "${RUN_HOST}/$(basename "${MODEL_RELOAD_OUT_CONTAINER}")"
      fi

      if [ "${eval_candidate}" = "1" ]; then
        for q in ${QUERIES}; do
          local query_path
          local out_file
          local ms_file
          local status_file
          query_path="$(resolve_query_path "${q}")"
          out_file="${RUN_HOST}/${prefix}${q}.on_after_update.out"
          ms_file="${RUN_HOST}/${prefix}${q}.on_after_update.ms"
          status_file="${RUN_HOST}/${prefix}${q}.on_after_update.status"
          run_one_query "${q}" "${query_path}" "on" "${out_file}" "${ms_file}" "${status_file}"
          printf '%s,%s,on_after_update,%s,%s,%s\n' "${experiment_round}" "${q}" "$(cat "${status_file}")" "$(cat "${ms_file}")" "${out_file}" >>"${timing_csv}"
          if cmp -s "${RUN_HOST}/${prefix}${q}.off.out" "${out_file}"; then
            printf 'round_%s %s ON_AFTER_UPDATE_MATCH\n' "${experiment_round}" "${q}" | tee -a "${RUN_HOST}/compare.txt"
          else
            printf 'round_%s %s ON_AFTER_UPDATE_MISMATCH\n' "${experiment_round}" "${q}" | tee -a "${RUN_HOST}/compare.txt"
            diff -u "${RUN_HOST}/${prefix}${q}.off.out" "${out_file}" || true
          fi
        done
      fi

      if [ "${PROMOTION_GUARD}" = "1" ]; then
        local accepted
        accepted="$(write_promotion_decision "${experiment_round}" "${previous_model}" "${UPDATED_MODEL_CONTAINER}")"
        if [ "${accepted}" = "1" ]; then
          ACTIVE_MODEL_CONTAINER="${UPDATED_MODEL_CONTAINER}"
          printf 'round %s promotion=accepted model=%s\n' "${experiment_round}" "${ACTIVE_MODEL_CONTAINER}" | tee -a "${RUN_HOST}/rounds.log"
        else
          reload_server_model "${previous_model}" "${RUN_HOST}/${prefix}model_restore.out"
          ACTIVE_MODEL_CONTAINER="${previous_model}"
          printf 'round %s promotion=rejected restored_model=%s\n' "${experiment_round}" "${ACTIVE_MODEL_CONTAINER}" | tee -a "${RUN_HOST}/rounds.log"
        fi
      elif [ "${RELOAD_UPDATED_MODEL}" = "1" ]; then
        ACTIVE_MODEL_CONTAINER="${UPDATED_MODEL_CONTAINER}"
        printf 'round %s promotion=unguarded model=%s\n' "${experiment_round}" "${ACTIVE_MODEL_CONTAINER}" | tee -a "${RUN_HOST}/rounds.log"
      fi
    else
      docker_exec "python3 '${CONTAINER_REPO}/neurqo/server/online_trainer.py' --db-log '${DB_TRAJECTORY_CONTAINER}' --out '${TRANSITIONS_CONTAINER}' --once"
    fi
  done

  write_summary
  echo "run_dir=${RUN_HOST}"
  echo "server_stdout=${RUN_HOST}/ai_server.stdout.log"
  if [ "${ONLINE_TRAIN}" = "1" ]; then
    echo "active_model_container=${ACTIVE_MODEL_CONTAINER}"
  fi
}

main "$@"
