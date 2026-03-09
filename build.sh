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
            echo "  --cpu           Build and run in CPU mode (dev)"
            echo "  --gpu           Build and run in GPU mode (dev, default)"
            echo "  --release       Build both CPU and GPU release images (Dockerfile.release)"
            echo "  --db-port PORT  Specify the host port for database (default: 5432 for GPU, 15432 for CPU)"
            echo "  --debug-port PORT  Specify the host port for debug server (default: 1234 for GPU, 11234 for CPU)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Returns an available host port: uses the given default if free, otherwise
# picks a random ephemeral port in 49152-65535.
find_available_port() {
    local preferred=$1
    if ! ss -tlnH "sport = :${preferred}" 2>/dev/null | grep -q .; then
        echo "$preferred"
        return
    fi
    local port
    while true; do
        port=$(( RANDOM % 16383 + 49152 ))
        if ! ss -tlnH "sport = :${port}" 2>/dev/null | grep -q .; then
            echo "$port"
            return
        fi
    done
}

# Set default ports based on mode if not specified (only needed for dev builds)
if [ "$RELEASE" != true ]; then
    if [ "$MODE" == "cpu" ]; then
        DB_PORT=$(find_available_port "${DB_PORT:-15432}")
        DEBUG_PORT=$(find_available_port "${DEBUG_PORT:-11234}")
    else
        DB_PORT=$(find_available_port "${DB_PORT:-5432}")
        DEBUG_PORT=$(find_available_port "${DEBUG_PORT:-1234}")
    fi
fi

if [ "$RELEASE" = true ]; then
    # Release build: build both CPU and GPU (cuda11) variants
    for VARIANT in cpu cuda11; do
        IMAGE_NAME="neurdb:latest-${VARIANT}"
        echo "==> Building release image: ${IMAGE_NAME}"
        docker build \
            -f Dockerfile.release \
            --target release \
            --build-arg VARIANT=${VARIANT} \
            --progress=plain \
            -t ${IMAGE_NAME} .
    done
    echo "Release images built successfully:"
    echo "  neurdb:latest-cpu"
    echo "  neurdb:latest-cuda11"
else
    # Container name based on mode
    CONTAINER_NAME="neurdb_dev"
    if [ "$MODE" == "cpu" ]; then
        CONTAINER_NAME="${CONTAINER_NAME}_cpu"
    fi

    # Dev build: existing behaviour, unchanged
    docker rm -f ${CONTAINER_NAME} || true

    # Select Dockerfile based on mode
    DOCKERFILE="Dockerfile.cuda11"
    if [ "$MODE" == "cpu" ]; then
        DOCKERFILE="Dockerfile.cpu"
    fi

    docker build -t neurdbimg . -f ${DOCKERFILE} --progress=plain --no-cache

    # Dev run: existing behaviour, unchanged
    # Clean build directory based on CLEAN_BUILD env var
    # CLEAN_BUILD=1        : clean everything (compile + data)
    # CLEAN_BUILD=compile  : clean compile only, keep data

    # Set GPU_FLAG for docker run if in GPU mode
    GPU_FLAG=""
    if [ "$MODE" != "cpu" ]; then
        GPU_FLAG="--gpus all"
    fi

    # Run the Docker container with appropriate port mappings and GPU access
    docker run -d -e CLEAN_BUILD=1 --name "${CONTAINER_NAME}" \
        -v "$(pwd)":/code/neurdb-dev \
        -p "${DB_PORT}:5432" \
        -p "${DEBUG_PORT}:1234" \
        --cap-add=SYS_PTRACE \
        ${GPU_FLAG} \
        neurdbimg

    # Follow the Docker container logs
    docker logs -f ${CONTAINER_NAME}
fi

# psql -h localhost -U neurdb -d neurdb
