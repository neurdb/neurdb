#!/bin/bash
# Stop AI engine servers started by start_ai_servers.sh (python server.py).
# Finds and kills processes whose command line matches "python.*server.py"
# (same as start_ai_servers.sh), so no port range or N needed.
# Usage: ./script/stop_ai_servers.sh

# Match processes running python server.py (same as start_ai_servers.sh) (e.g. "python server.py" from aiengine/runtime)
if command -v pgrep &>/dev/null; then
  pids=$(pgrep -f "python.*server\.py" 2>/dev/null)
else
  pids=$(ps -eo pid,args 2>/dev/null | awk '/[p]ython.*server\.py/{print $1}')
fi

if [[ -z "$pids" ]]; then
  echo "No 'python server.py' processes found."
  exit 0
fi

echo "Stopping AI server process(es): $pids"
echo "$pids" | xargs kill 2>/dev/null
echo "Done."
