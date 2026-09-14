#!/bin/bash

# This script downloads the Eigen library to external/eigen.
# The Eigen library is header-only, so no compilation is needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EIGEN_DIR="$SCRIPT_DIR/../external/eigen"
TEMP_DIR="$EIGEN_DIR/eigen_temp"

# Skip if already downloaded
if [ -f "$EIGEN_DIR/Eigen/Core" ]; then
  echo "Eigen already exists at $EIGEN_DIR, skipping download"
  exit 0
fi

# Clean and recreate directory
rm -rf "$EIGEN_DIR"
mkdir -p "$EIGEN_DIR"
mkdir -p "$TEMP_DIR"

wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xvzf eigen-3.4.0.tar.gz -C "$TEMP_DIR"
mv ${TEMP_DIR}/eigen-3.4.0/* "$EIGEN_DIR"
rm -rf "$TEMP_DIR"
rm eigen-3.4.0.tar.gz
echo "Eigen downloaded to $EIGEN_DIR"
