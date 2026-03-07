#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

# Set default mode to GPU
MODE="gpu"
RELEASE=false

# Default port mappings (empty means use mode defaults)
DB_PORT=""
DEBUG_PORT=""

# Check for the mode argument (CPU or GPU) and port options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cpu) MODE="cpu" ;;
        --gpu) MODE="gpu" ;;
        --release) RELEASE=true ;;
        --db-port) DB_PORT="$2"; shift ;;
        --debug-port) DEBUG_PORT="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu           Build and run in CPU mode"
            echo "  --gpu           Build and run in GPU mode (default)"
            echo "  --release       Build and run the release image (Dockerfile.release)"
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
if [ "$RELEASE" = true ]; then
    DB_PORT=${DB_PORT:-5432}
else
    if [ "$MODE" == "cpu" ]; then
        DB_PORT=${DB_PORT:-15432}
        DEBUG_PORT=${DEBUG_PORT:-11234}
    else
        DB_PORT=${DB_PORT:-5432}
        DEBUG_PORT=${DEBUG_PORT:-1234}
    fi
fi

if [ "$RELEASE" = true ]; then
    # Release build: uses Dockerfile.release, selects variant via build arg
    VARIANT="cpu"
    [ "$MODE" == "gpu" ] && VARIANT="cuda11"
    IMAGE_NAME="neurdb:latest-${VARIANT}"
    docker build \
        -f Dockerfile.release \
        --target release \
        --build-arg VARIANT=${VARIANT} \
        --progress=plain \
        -t ${IMAGE_NAME} .

    # Release run: no source mount, no debug port, no CLEAN_BUILD
    CONTAINER_NAME="neurdb"
    docker rm -f ${CONTAINER_NAME} || true
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${DB_PORT}:5432 \
        --restart unless-stopped \
        ${IMAGE_NAME}
    docker logs -f ${CONTAINER_NAME}
else
    # Dev build: existing behaviour, unchanged
    docker rm -f neurdb_dev || true

    if [ "$MODE" == "cpu" ]; then
        docker build -t neurdbimg . -f Dockerfile.cpu --progress=plain --no-cache
    else
        docker build -t neurdbimg . -f Dockerfile.cuda11 --progress=plain --no-cache
    fi

    # Dev run: existing behaviour, unchanged
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
fi

# psql -h localhost -U neurdb -d neurdb
