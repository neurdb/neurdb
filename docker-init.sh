#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

# Variables (can be changed by user)
NEURDBPATH=${NEURDBPATH:-/code/neurdb-dev}
NR_BUILD_PATH=${NR_BUILD_PATH:-$NEURDBPATH/build}
NR_PSQL_PATH=${NR_PSQL_PATH:-$NR_BUILD_PATH/psql}
NR_DBDATA_PATH=${NR_DBDATA_PATH:-$NR_BUILD_PATH/data}

# Variables (cannot be changed by user)
NR_DBENGINE_PATH=$NEURDBPATH/dbengine
NR_AIENGINE_PATH=$NEURDBPATH/aiengine
NR_API_PATH=$NEURDBPATH/api
#NR_PIPELINE_PATH=$NR_DBENGINE_PATH/nr_kernel/nrext
NR_KERNEL_PATH=$NR_DBENGINE_PATH/nr_kernel

# Clean log file
# rm $NEURDBPATH/logfile || true

# Clean build directory based on CLEAN_BUILD env var
# CLEAN_BUILD=1        : clean everything (compile + data)
# CLEAN_BUILD=compile  : clean compile only, keep data
if [ "$CLEAN_BUILD" = "1" ]; then
  echo "Cleaning entire build directory..."
  rm -rf $NR_BUILD_PATH
elif [ "$CLEAN_BUILD" = "compile" ]; then
  echo "Cleaning compile artifacts, keeping data..."
  rm -rf $NR_BUILD_PATH/dbengine
  rm -rf $NR_BUILD_PATH/psql
  rm -rf $NR_BUILD_PATH/nr_kernel
  rm -rf $NR_BUILD_PATH/contrib
  rm -rf $NR_BUILD_PATH/api
  rm -f $NEURDBPATH/psql
fi

# # Install external lib.
# git clone https://github.com/facebook/rocksdb.git
# cd rocksdb
# make shared_lib -j
# sudo make install-shared

# Create build folder
mkdir -p $NR_BUILD_PATH

# Create psql folder
mkdir -p $NR_PSQL_PATH

# Create symlink for backward compatibility
if [ "$NR_PSQL_PATH" != "$NEURDBPATH/psql" ]; then
  rm -rf $NEURDBPATH/psql
  ln -sf $NR_PSQL_PATH $NEURDBPATH/psql
fi

# Compile PostgreSQL (out-of-source build)
mkdir -p $NR_BUILD_PATH/dbengine
cd $NR_BUILD_PATH/dbengine
$NR_DBENGINE_PATH/configure --prefix=$NR_PSQL_PATH --enable-debug
make -j
make install
echo 'Done! Now start the database'

# Compile and install pg_hint_plan (in build dir to avoid permission issues)
echo 'Installing pg_hint_plan...'
mkdir -p $NR_BUILD_PATH/contrib
cd $NR_BUILD_PATH/contrib
if [ ! -d "pg_hint_plan" ]; then
  git clone https://github.com/ossc-db/pg_hint_plan.git
fi
cd pg_hint_plan
git checkout PG16
make clean || true
make PG_CONFIG=$NR_PSQL_PATH/bin/pg_config
make PG_CONFIG=$NR_PSQL_PATH/bin/pg_config install
echo 'pg_hint_plan installed!'

# Crete DB engine if not exist
if [ ! -d "$NR_DBDATA_PATH" ]; then
  mkdir -p $NR_DBDATA_PATH
  $NR_PSQL_PATH/bin/initdb -D $NR_DBDATA_PATH
else
  # make sure DBDATA folder is under permission 0750 to avoid PG start failure
  sudo chmod 0750 $NR_DBDATA_PATH
fi

# Start DB engine
$NR_PSQL_PATH/bin/pg_ctl -D $NR_DBDATA_PATH -l $NR_BUILD_PATH/logfile start

# Wait a few seconds to ensure DB engine is up and running
until $NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U neurdb -c '\q'; do
  >&2 echo 'NeurDB is unavailable - sleeping'
  sleep 1
  # try to create `neurdb` database into the cluster
  $NR_PSQL_PATH/bin/createdb -h localhost -p 5432 neurdb
done
echo "DB Started!"

# Load iris dataset
# $NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U neurdb -f $NEURDBPATH/dataset/iris/iris_psql.sql

# Install neurdb package (copy to build dir to avoid permission issues)
mkdir -p $NR_BUILD_PATH/api/python
cp -r $NR_API_PATH/python/* $NR_BUILD_PATH/api/python/
cd $NR_BUILD_PATH/api/python
## For older setuptools
## Reference: https://stackoverflow.com/a/71946741
touch setup.cfg
sudo pip install -e .
rm setup.cfg
## For newer setuptools
# pip install -e . --config-settings editable_mode=compat

# Install socket.io-client-cpp (for nr_pipeline)
## Since we have migrated to websockets, we don't need this
# if [ ! -d "$NR_PIPELINE_PATH/lib" ]; then
#   mkdir -p $NR_PIPELINE_PATH/lib
# fi
# cd $NR_PIPELINE_PATH/lib

# if [ ! -d "socket.io-client-cpp" ]; then
#   git clone --recurse-submodules https://github.com/socketio/socket.io-client-cpp.git
# fi
# cd socket.io-client-cpp
# sudo chmod -R 777 ./
# cmake -DCMAKE_CXX_FLAGS="-fPIC" ./
# sudo make -j
# sudo make install

# Compile nr_pipeline
#cd $NR_PIPELINE_PATH
#sudo make clean
#sudo make install
#echo "Install NR Data Pipeline Extension & NR kernel extension Done"

# Compile nr_kernel (Makefile auto-syncs to build dir)
cd $NR_KERNEL_PATH
export PG_CONFIG=$NR_PSQL_PATH/bin/pg_config
make clean || true
make
make install

## Register nr_kernel as preloaded library
sed -i '/^#*shared_preload_libraries/d' $NR_DBDATA_PATH/postgresql.conf
echo 'shared_preload_libraries = '\''pg_hint_plan, nr_molqo, nr_ext, nram, pg_neurstore'\''' >> $NR_DBDATA_PATH/postgresql.conf

echo "Install NR Data Pipeline Extension & NR kernel extension Done"

# Restart DB engine to refresh the configuration
$NR_PSQL_PATH/bin/pg_ctl -D $NR_DBDATA_PATH -l $NR_BUILD_PATH/logfile restart

# Wait a few seconds to ensure DB engine is up and running
until $NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U neurdb -c '\q'; do
  >&2 echo 'NeurDB is unavailable - sleeping'
  sleep 1
  # try to create `neurdb` database into the cluster
  $NR_PSQL_PATH/bin/createdb -h localhost -p 5432 neurdb
done
echo "DB Restarted!"

# Run python server
cd $NR_AIENGINE_PATH/runtime
export NR_LOG_LEVEL=INFO  # Set log level
nohup python server.py &
sleep 10
echo -n 'Waiting for Python server to start '
until curl --output /dev/null --silent --head --fail http://127.0.0.1:8090/; do
  printf '.'
  sleep 1
done
echo 'Python Server started!'

echo "Please use 'control + c' to exit the logging print"

$NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U neurdb -c 'CREATE EXTENSION IF NOT EXISTS nr_pipeline;'
echo 'Install NR Data Pipeline Extension Done'

if [ "$GITHUB_ACTIONS" != "true" ]; then
  # If not in GitHub Actions, keep container running
  tail -f /dev/null
else
  # If in GitHub Actions
  # Wait for Python server to start
  echo -n 'Waiting for Python server to start '
  until curl --output /dev/null --silent --head --fail http://127.0.0.1:8090/; do
    printf '.'
    sleep 1
  done
  echo 'OK'

  # Do test
  cd $NEURDBPATH/test
  export PATH=$NR_PSQL_PATH/bin:$PATH
  bash test.sh

  # Exit normally
  exit 0
fi
