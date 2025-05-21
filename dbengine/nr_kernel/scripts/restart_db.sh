#!/bin/bash

# Navigate to the nr_kernel directory
cd /code/neurdb-dev/dbengine/nr_kernel || {
    echo "Failed to navigate to nr_kernel directory"
    exit 1
}

# Stop the NEURDB PostgreSQL server
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data stop || {
    echo "Failed to stop the NEURDB PostgreSQL server (already started)"
}

# Start the NEURDB PostgreSQL server
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data -l logfile start || {
    echo "Failed to start the NEURDB PostgreSQL server"
    exit 1
}

