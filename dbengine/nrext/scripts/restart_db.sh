#!/bin/bash

# Navigate to the nr_kernel directory
cd /code/neurdb-dev/dbengine/nrext/nr_kernel || {
    echo "Failed to navigate to nr_kernel directory"
    exit 1
}

# Stop the NEURDB PostgreSQL server
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data stop || {
    echo "Failed to stop the NEURDB PostgreSQL server"
    exit 1
}

# Build and install the kernel
make && make install || {
    echo "Failed to build and install the kernel"
    exit 1
}

# Start the NEURDB PostgreSQL server
$NEURDBPATH/psql/bin/pg_ctl -D $NEURDBPATH/psql/data -l logfile start || {
    echo "Failed to start the NEURDB PostgreSQL server"
    exit 1
}

# Launch psql
psql || {
    echo "Failed to launch psql"
    exit 1
}
