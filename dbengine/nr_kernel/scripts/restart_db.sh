#!/bin/bash

NEURDBPATH=${NEURDBPATH:-/code/neurdb-dev}

# Stop the NEURDB PostgreSQL server
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data stop || {
    echo "Failed to stop the NEURDB PostgreSQL server (already stopped)"
}

# Start the NEURDB PostgreSQL server
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data -l logfile start || {
    echo "Failed to start the NEURDB PostgreSQL server"
    exit 1
}
