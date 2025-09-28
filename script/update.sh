#!/bin/bash
set -e

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"

NEURDBPATH=${NEURDBPATH:-/code/neurdb-dev}
NR_PSQL_PATH=${NR_PSQL_PATH:-$NEURDBPATH/psql}
NR_DBDATA_PATH=${NR_DBDATA_PATH:-$NR_PSQL_PATH/data}
NR_DBENGINE_PATH=$NEURDBPATH/dbengine
NR_AIENGINE_PATH=$NEURDBPATH/aiengine
NR_API_PATH=$NEURDBPATH/api
NR_KERNEL_PATH=$NR_DBENGINE_PATH/nr_kernel
NR_CONF_FILE="$NR_DBDATA_PATH/postgresql.conf"

# Stop DB if running
if [ -d "$NR_DBDATA_PATH" ]; then
  $NR_PSQL_PATH/bin/pg_ctl -D $NR_DBDATA_PATH -l logfile stop || true
fi

# Clean previous database data
rm -rf "$NR_DBDATA_PATH"
rm -f "${NR_DBENGINE_PATH}/logfile"

# Rebuild PostgreSQL (release flags)
cd "$NR_DBENGINE_PATH"
make clean
./configure \
  --prefix="$NR_PSQL_PATH" \
  --disable-cassert \
  --disable-debug \
  CFLAGS="-O3 -DNDEBUG -march=native -fno-omit-frame-pointer"
make -j"$(nproc)"
sudo make install

# Init DB
mkdir -p "$NR_DBDATA_PATH"
$NR_PSQL_PATH/bin/initdb -D "$NR_DBDATA_PATH"

# Config: preload our ext but no debug spam
echo "shared_preload_libraries = 'nr_ext, nram'" >> "$NR_CONF_FILE"
echo "log_min_messages = warning" >> "$NR_CONF_FILE"
echo "max_prepared_transactions = 2" >> "$NR_CONF_FILE"

# Start DB
$NR_PSQL_PATH/bin/pg_ctl -D "$NR_DBDATA_PATH" -l logfile start

# Wait until ready & create DB
until $NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U neurdb -c '\q' >/dev/null 2>&1; do
  echo "Waiting for DB to be ready..."
  sleep 1
  $NR_PSQL_PATH/bin/createdb -h localhost -p 5432 neurdb || true
done
echo "Database rebuilt and started."

# Reinstall Python API (if needed)
cd "$NR_API_PATH/python"
touch setup.cfg
sudo pip install -e .
rm setup.cfg

# Recompile kernel extension
cd "$NR_KERNEL_PATH"
sudo make clean
sudo make install

# Keep container alive if used as dev image
tail -f /dev/null
