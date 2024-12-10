#!/bin/bash

# Find the PID of the target PostgreSQL process
pid=$(ps aux | grep 'postgres: postgres postgres' | grep -v grep | awk '{print $2}')

# Check if a PID was found
if [ -z "$pid" ]; then
    echo "No PostgreSQL process found."
    exit 1
fi

echo "Found PostgreSQL process with PID: $pid"

# Run gdbserver with the found PID
gdbserver 0.0.0.0:1234 --attach "$pid"
