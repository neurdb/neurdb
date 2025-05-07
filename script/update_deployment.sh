#!/bin/bash

set -e
set -x

export LANG="en_US.UTF-8"
export LC_COLLATE="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
export LC_MESSAGES="en_US.UTF-8"
export LC_MONETARY="en_US.UTF-8"
export LC_NUMERIC="en_US.UTF-8"
export LC_TIME="en_US.UTF-8"
export DYLD_LIBRARY_PATH=$NEURDBPATH/psql/lib:$DYLD_LIBRARY_PATH

NEURDBPATH=${NEURDBPATH:-/code/neurdb-dev}
NR_PSQL_PATH=${NR_PSQL_PATH:-$NEURDBPATH/psql}
NR_DBDATA_PATH=${NR_DBDATA_PATH:-$NR_PSQL_PATH/data}
NR_DBENGINE_PATH=$NEURDBPATH/dbengine
NR_AIENGINE_PATH=$NEURDBPATH/aiengine
NR_API_PATH=$NEURDBPATH/api
NR_KERNEL_PATH=$NR_DBENGINE_PATH/nr_kernel

# Stop DB if running
if [ -d "$NR_DBDATA_PATH" ]; then
  $NR_PSQL_PATH/bin/pg_ctl -D $NR_DBDATA_PATH -l logfile stop || true
fi

# Clean previous database data
rm -rf $NR_DBDATA_PATH

# Rebuild PostgreSQL
cd $NR_DBENGINE_PATH
make clean
make -j
make install

# Re-init DB
mkdir -p $NR_DBDATA_PATH
$NR_PSQL_PATH/bin/initdb -D $NR_DBDATA_PATH

# Re-append kernel extension
echo "shared_preload_libraries = 'nr_ext'" >> $NR_DBDATA_PATH/postgresql.conf

# Start DB
$NR_PSQL_PATH/bin/pg_ctl -D $NR_DBDATA_PATH -l logfile start

# Wait for DB and create 'neurdb' database
until $NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U neurdb -c '\q'; do
  echo "Waiting for DB to be ready..."
  sleep 1
  $NR_PSQL_PATH/bin/createdb -h localhost -p 5432 neurdb || true
done
echo "Database rebuilt and started."

# (Optional) Load dataset
# $NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U neurdb -f $NEURDBPATH/dataset/iris/iris_psql.sql

# Reinstall Python API
cd $NR_API_PATH/python
touch setup.cfg
sudo pip install -e .
rm setup.cfg

# Recompile NR kernel
cd $NR_KERNEL_PATH
sudo make clean
sudo make install

# Start AI service
cd $NR_AIENGINE_PATH/runtime
export NR_LOG_LEVEL=INFO
nohup python -m hypercorn server:app -c app_config.toml &
echo "Python server restarted."

tail -f /dev/null
