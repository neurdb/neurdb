#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

# Set default mode to GPU
MODE="gpu"

# Default port mappings (empty means use mode defaults)
DB_PORT=""
DEBUG_PORT=""

# Check for the mode argument (CPU or GPU) and port options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cpu) MODE="cpu" ;;
        --gpu) MODE="gpu" ;;
        --db-port) DB_PORT="$2"; shift ;;
        --debug-port) DEBUG_PORT="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu           Build and run in CPU mode"
            echo "  --gpu           Build and run in GPU mode (default)"
            echo "  --db-port PORT  Specify the host port for database (default: 5432 for GPU, 15432 for CPU)"
            echo "  --debug-port PORT  Specify the host port for debug server (default: 1234 for GPU, 11234 for CPU)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set default ports based on mode if not specified
if [ "$MODE" == "cpu" ]; then
    DB_PORT=${DB_PORT:-15432}
    DEBUG_PORT=${DEBUG_PORT:-11234}
else
    DB_PORT=${DB_PORT:-5432}
    DEBUG_PORT=${DEBUG_PORT:-1234}
fi

# Remove the Docker container if it exists, suppressing errors if it doesn't
docker rm -f neurdb_dev || true

# Build the Docker image based on the selected mode
if [ "$MODE" == "cpu" ]; then
    docker build -t neurdbimg . -f Dockerfile.cpu --progress=plain --no-cache
else
    docker build -t neurdbimg . -f Dockerfile.cuda11 --progress=plain --no-cache
fi

# This to solve the permission problems
# chmod -R a+rwX .; \

# Run the Docker container
# You may replace or delete the port mapping

# Clean build directory based on CLEAN_BUILD env var
# CLEAN_BUILD=1        : clean everything (compile + data)
# CLEAN_BUILD=compile  : clean compile only, keep data

if [ "$MODE" == "cpu" ]; then
    docker run -d -e CLEAN_BUILD=1 --name neurdb_dev_opt \
      -v "$(pwd)":/code/neurdb-dev \
      -p ${DB_PORT}:5432 \
      -p ${DEBUG_PORT}:1234 \
      --cap-add=SYS_PTRACE \
      neurdbimg-opt
else
    docker run -d -e CLEAN_BUILD=1 --name neurdb_dev \
        -v $(pwd):/code/neurdb-dev \
        -p ${DB_PORT}:5432 \
        -p ${DEBUG_PORT}:1234 \
        --cap-add=SYS_PTRACE \
        --gpus all \
        neurdbimg
fi

# Follow the Docker container logs
docker logs -f neurdb_dev


# psql -h localhost -U neurdb -d neurdb
