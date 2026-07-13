#!/bin/bash
#
# NeurDB Installer for Linux (Docker mode)
# Usage: install.sh [OPTIONS]
#
# Options:
#   --version VERSION   NeurDB version to install (default: latest)
#   --cpu               Force CPU image
#   --gpu               Force GPU image (default: auto-detect)
#   --port PORT         Host port for PostgreSQL (default: 5432)
#   --data-dir PATH     Bind mount for persistent data
#   --uninstall         Stop and remove the NeurDB container and image
#   --native            Install natively (no Docker) — see Phase 3
#

set -e

# Defaults
VERSION="latest"
VARIANT=""
PORT=5432
DATA_DIR=""
UNINSTALL=false
NATIVE=false
CONTAINER_NAME="neurdb"
REGISTRY="ghcr.io/neurdb/neurdb"

usage() {
    echo "NeurDB Installer for Linux"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --version VERSION   NeurDB version to install (default: latest)"
    echo "  --cpu               Force CPU image"
    echo "  --gpu               Force GPU image (default: auto-detect)"
    echo "  --port PORT         Host port for PostgreSQL (default: 5432)"
    echo "  --data-dir PATH     Bind mount for persistent data outside the container"
    echo "  --uninstall         Stop and remove the NeurDB container and image"
    echo "  --native            Install natively without Docker (requires Ubuntu 20.04+)"
    echo "  -h, --help          Show this help message"
    exit 0
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version)  VERSION="$2"; shift ;;
        --cpu)      VARIANT="cpu" ;;
        --gpu)      VARIANT="cuda11" ;;
        --port)     PORT="$2"; shift ;;
        --data-dir) DATA_DIR="$2"; shift ;;
        --uninstall) UNINSTALL=true ;;
        --native)   NATIVE=true ;;
        -h|--help)  usage ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# --- Uninstall mode ---
if [ "$UNINSTALL" = true ]; then
    echo "Stopping and removing NeurDB..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    # Remove images
    for tag in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep "neurdb"); do
        docker rmi "$tag" 2>/dev/null || true
    done
    echo "NeurDB has been uninstalled."
    exit 0
fi

# --- Native mode placeholder (implemented in Phase 3, Step 3.3) ---
if [ "$NATIVE" = true ]; then
    # Source the native installation logic
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/install-native.sh" ]; then
        exec bash "$SCRIPT_DIR/install-native.sh" "$@"
    else
        echo "Error: Native installation is not yet available."
        echo "Please use Docker mode (remove --native flag)."
        exit 1
    fi
fi

# --- Docker mode ---

# 1. Check prerequisites
if ! command -v docker &>/dev/null; then
    echo "Error: Docker is not installed."
    echo "Install Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "Error: Docker daemon is not running."
    echo "Start Docker with: sudo systemctl start docker"
    exit 1
fi

# 2. Detect GPU availability if not explicitly set
if [ -z "$VARIANT" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "NVIDIA GPU detected — using GPU image."
        VARIANT="cuda11"
    else
        echo "No NVIDIA GPU detected — using CPU image."
        VARIANT="cpu"
    fi
fi

# 3. Determine image tag
if [ "$VERSION" = "latest" ]; then
    IMAGE_TAG="latest-${VARIANT}"
else
    IMAGE_TAG="${VERSION}-${VARIANT}"
fi
IMAGE="${REGISTRY}:${IMAGE_TAG}"

echo "Pulling NeurDB image: ${IMAGE}"
docker pull "$IMAGE"

# 4. Stop existing container if running
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# 5. Build run command
RUN_ARGS=(-d --name "$CONTAINER_NAME" -p "${PORT}:5432" --restart unless-stopped)

if [ -n "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
    RUN_ARGS+=(-v "${DATA_DIR}:/var/lib/neurdb/data")
fi

if [ "$VARIANT" = "cuda11" ]; then
    RUN_ARGS+=(--gpus all)
fi

RUN_ARGS+=("$IMAGE")

echo "Starting NeurDB container..."
docker run "${RUN_ARGS[@]}"

# 6. Wait for readiness
echo -n "Waiting for NeurDB to be ready "
MAX_WAIT=120
ELAPSED=0
while ! docker exec "$CONTAINER_NAME" /opt/neurdb/bin/psql -h localhost -p 5432 -U neurdb -c '\q' 2>/dev/null; do
    printf '.'
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo ""
        echo "Warning: NeurDB did not become ready within ${MAX_WAIT}s."
        echo "Check logs with: docker logs $CONTAINER_NAME"
        exit 1
    fi
done
echo " OK"

# 7. Print connection info
echo ""
echo "============================================"
echo "  NeurDB is running!"
echo "============================================"
echo ""
echo "  Connect with:"
echo "    psql -h localhost -p ${PORT} -U neurdb -d neurdb"
echo ""
echo "  Container name: ${CONTAINER_NAME}"
echo "  View logs:      docker logs -f ${CONTAINER_NAME}"
echo "  Stop:           docker stop ${CONTAINER_NAME}"
echo "  Uninstall:      $0 --uninstall"
echo ""
