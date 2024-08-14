#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

# Remove the Docker container if it exists, suppressing errors if it doesn't
docker rm -f neurdb_dev || true

# Build the Docker image
docker build -t neurdbimg .

# Run the Docker container
# You may replace or delete the port mapping
docker run -d --name neurdb_dev \
    -v $(pwd):/code/neurdb-dev \
    -p 5432:5432 \
    --cap-add=SYS_PTRACE \
    --gpus all \
    neurdbimg

# Follow the Docker container logs
docker logs -f neurdb_dev
