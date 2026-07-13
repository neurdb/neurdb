#!/bin/bash

# Release entrypoint — startup only, no compilation.
# All binaries are pre-built inside the Docker image.

set -e

NEURDB_DATA="${NEURDB_DATA:-/var/lib/neurdb/data}"
NEURDB_LOG_DIR="/var/log/neurdb"
PG_BIN="/opt/neurdb/bin"
VENV_DIR="/opt/neurdb-venv"
RUNTIME_DIR="${VENV_DIR}/runtime"

# 1. Run initdb if the cluster has not been initialised yet (first-run only).
# Check for PG_VERSION rather than the directory itself, because the directory
# may have been pre-created (e.g. by the Dockerfile) without initdb having run.
if [ ! -f "$NEURDB_DATA/PG_VERSION" ]; then
    mkdir -p "$NEURDB_DATA"
    "$PG_BIN/initdb" -D "$NEURDB_DATA" -U neurdb
fi

# 2. Patch postgresql.conf to load nr_kernel extensions
sed -i '/^#*shared_preload_libraries/d' "$NEURDB_DATA/postgresql.conf"
echo "shared_preload_libraries = 'pg_hint_plan, nr_molqo, nr_ext, nram, pg_neurstore'" >> "$NEURDB_DATA/postgresql.conf"

# Allow connections from any host (for Docker port forwarding)
if ! grep -q "listen_addresses = '\*'" "$NEURDB_DATA/postgresql.conf"; then
    echo "listen_addresses = '*'" >> "$NEURDB_DATA/postgresql.conf"
fi
if ! grep -q "0.0.0.0/0" "$NEURDB_DATA/pg_hba.conf"; then
    echo "host all all 0.0.0.0/0 trust" >> "$NEURDB_DATA/pg_hba.conf"
fi

# 3. Start PostgreSQL
"$PG_BIN/pg_ctl" -D "$NEURDB_DATA" -l /tmp/neurdb-postgres.log start

# 4. Wait for PostgreSQL to accept connections
echo -n 'Waiting for NeurDB to start '
until "$PG_BIN/psql" -h localhost -p 5432 -U neurdb -c '\q' 2>/dev/null; do
    printf '.'
    sleep 1
    # 5. Create the 'neurdb' database if it does not exist
    "$PG_BIN/createdb" -h localhost -p 5432 -U neurdb neurdb 2>/dev/null || true
done
echo ' OK'

# 6. Install nr_pipeline extension
"$PG_BIN/psql" -h localhost -p 5432 -U neurdb -c 'CREATE EXTENSION IF NOT EXISTS nr_pipeline;'
echo 'NR Pipeline extension installed'

# 7. Start the Python AI runtime server
cd "$RUNTIME_DIR"
export NR_LOG_LEVEL=INFO
nohup "$VENV_DIR/bin/python" server.py > /tmp/neurdb-ai-server.log 2>&1 &

# 8. Wait for the AI server to respond on :8090
echo -n 'Waiting for AI runtime server to start '
until curl --output /dev/null --silent --head --fail http://127.0.0.1:8090/; do
    printf '.'
    sleep 1
done
echo ' OK'

echo "NeurDB is ready."
echo "Connect with: psql -h localhost -p 5432 -U neurdb -d neurdb"

# 9. Keep the container alive
tail -f /dev/null
