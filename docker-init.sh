#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

# Variables (can be changed by user)
NEURDBPATH=${NEURDBPATH:-/code/neurdb-dev}
NR_PSQL_PATH=${NR_PSQL_PATH:-$NEURDBPATH/psql}
NR_DBDATA_PATH=${NR_DBDATA_PATH:-$NR_PSQL_PATH/data}

# Variables (cannot be changed by user)
NR_DBENGINE_PATH=$NEURDBPATH/dbengine
NR_AIENGINE_PATH=$NEURDBPATH/aiengine
NR_API_PATH=$NEURDBPATH/api
#NR_PIPELINE_PATH=$NR_DBENGINE_PATH/nr_kernel/nrext
NR_KERNEL_PATH=$NR_DBENGINE_PATH/nr_kernel

# Clean log file
rm $NEURDBPATH/logfile || true

# Clean the psql folder (only for debugging purposes)
# rm -rf $NR_PSQL_PATH   || true

# # Install external lib.
# git clone https://github.com/facebook/rocksdb.git
# cd rocksdb
# make shared_lib -j
# sudo make install-shared

# Create psql folder
mkdir -p $NR_PSQL_PATH

# Compile PostgreSQL
cd $NR_DBENGINE_PATH
make distclean || true
./configure --prefix=$NR_PSQL_PATH
make -j
make install
echo 'Done! Now start the database'

# Crete DB engine if not exist
if [ ! -d "$NR_DBDATA_PATH" ]; then
  mkdir -p $NR_DBDATA_PATH
  $NR_PSQL_PATH/bin/initdb -D $NR_DBDATA_PATH
else
  # make sure DBDATA folder is under permission 0750 to avoid PG start failure
  sudo chmod 0750 $NR_DBDATA_PATH
fi

# Start DB engine
$NR_PSQL_PATH/bin/pg_ctl -D $NR_DBDATA_PATH -l logfile start

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

# Install neurdb package
cd $NR_API_PATH/python
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

# Compile nr_kernel
cd $NR_KERNEL_PATH
sudo make clean
sudo make install

## Register nr_kernel as preloaded library
echo 'shared_preload_libraries = '\''nr_ext, nram'\''' >> $NR_DBDATA_PATH/postgresql.conf

echo "Install NR Data Pipeline Extension & NR kernel extension Done"

# Run python server
cd $NR_AIENGINE_PATH/runtime
export NR_LOG_LEVEL=INFO  # Set log level
nohup python -m hypercorn server:app -c app_config.toml &
echo 'Python Server started!'

echo "Please use 'control + c' to exit the logging print"

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
