#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

# This script downloads and build all required external libraries.
chmod +x ./download_onnx.sh \
  ./download_eigen.sh \
  ./download_tokenizer.sh

cd $SCRIPT_DIR
./download_onnx.sh
cd $SCRIPT_DIR
./download_eigen.sh
cd $SCRIPT_DIR
./download_tokenizer.sh

echo "All libraries downloaded and installed."
