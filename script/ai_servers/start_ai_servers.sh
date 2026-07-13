#!/bin/bash
# Start N AI engine servers (python server.py) in background.
# Usage: ./script/ai_servers/start_ai_servers.sh <N>
#   e.g. ./script/ai_servers/start_ai_servers.sh 3  → ports 8090,8091,8092; CUDA 0,1,2; logs in test/
# Ports start at 8090; CUDA_VISIBLE_DEVICES cycles from 0.

N="${1:-1}"
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
  echo "Usage: $0 <number>" >&2
  echo "  e.g. $0 3  to start 3 servers on 8090, 8091, 8092" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_DIR="${REPO_ROOT}/aiengine/runtime"
LOG_DIR="${REPO_ROOT}/test"
mkdir -p "$LOG_DIR"

if [[ ! -f "${RUNTIME_DIR}/server.py" ]]; then
  echo "Not found: ${RUNTIME_DIR}/server.py" >&2
  exit 1
fi

cd "$RUNTIME_DIR" || exit 1

BASE_PORT=8090
PIDS=()
for (( i=0; i<N; i++ )); do
  port=$(( BASE_PORT + i ))
  cuda=$i
  logfile="${LOG_DIR}/server_${port}.log"
  echo "Starting server $((i+1))/$N: port=$port CUDA_VISIBLE_DEVICES=$cuda log=$logfile"
  CUDA_VISIBLE_DEVICES=$cuda NR_LOG_LEVEL=INFO NR_PORT=$port nohup python -u server.py >> "$logfile" 2>&1 &
  PIDS+=($!)
  sleep 1
done

echo ""
echo "Started $N server(s). PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_DIR}/server_*.log"
echo ""
echo "--- Run these in psql to register the engines ---"
echo "DELETE FROM pg_catalog.nr_aiengine;"
for (( i=0; i<N; i++ )); do
  port=$(( BASE_PORT + i ))
  echo "select insert_ai_engine('127.0.0.1', $port);"
done
echo ""
