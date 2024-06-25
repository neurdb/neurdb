#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

# Ensure NEURDBPATH is set
NEURDBPATH=${NEURDBPATH:-/code/neurdb-dev}

# Clean files
rm -rf $NEURDBPATH/psql
rm $NEURDBPATH/logfile

# Print and execute commands
mkdir -p $NEURDBPATH/psql

cd $NEURDBPATH

make distclean || true

# Compile the PostgreSQL.
./configure --prefix=$NEURDBPATH/psql
make -j
make install

echo "Done! Now start the database"
mkdir -p $NEURDBPATH/psql/data
./psql/bin/initdb -D $NEURDBPATH/psql/data
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data -l logfile start

# Wait a few seconds to ensure the database is up and running
sleep 5

# Load dataset
$NEURDBPATH/psql/bin/psql -h localhost -U postgres -p 5432 -f $NEURDBPATH/dataset/iris_psql.sql

# Install packages
pip3 install -r $NEURDBPATH/contrib/nr/pysrc/requirement.txt

echo "DB started!"

# Continue
tail -f /dev/null

