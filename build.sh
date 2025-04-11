#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

# Set default mode to GPU
MODE="gpu"

# Check for the mode argument (CPU or GPU)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cpu) MODE="cpu" ;;
        --gpu) MODE="gpu" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Remove the Docker container if it exists, suppressing errors if it doesn't
docker rm -f neurdb_dev || true

# Build the Docker image based on the selected mode
if [ "$MODE" == "cpu" ]; then
    docker build -t neurdbimg . -f Dockerfile.cpu --progress=plain --no-cache
else
    docker build -t neurdbimg . -f Dockerfile.cuda11 --progress=plain --no-cache
fi

# Run the Docker container
# You may replace or delete the port mapping
if [ "$MODE" == "cpu" ]; then
    docker run -d --name neurdb_dev \
        -v $(pwd):/code/neurdb-dev \
        -p 5432:5432 \
        -p 1234:1234 \
        --cap-add=SYS_PTRACE \
        neurdbimg
else
    docker run -d --name neurdb_dev \
        -v $(pwd):/code/neurdb-dev \
        -p 5432:5432 \
        -p 1234:1234 \
        --cap-add=SYS_PTRACE \
        --gpus all \
        neurdbimg
fi

# Follow the Docker container logs
docker logs -f neurdb_dev
