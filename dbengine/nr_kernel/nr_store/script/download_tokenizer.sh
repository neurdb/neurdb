#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${SCRIPT_DIR}/.."
TOKENIZERS_SRC="${PROJECT_ROOT}/external/tokenizers-cpp"
TOKENIZERS_BUILD="${TOKENIZERS_SRC}/build"
TOKENIZERS_INSTALL="${TOKENIZERS_SRC}/install"

echo ">>> Cloning tokenizers-cpp ..."
rm -rf "${TOKENIZERS_SRC}"
git clone --depth 1 --recursive https://github.com/mlc-ai/tokenizers-cpp.git "${TOKENIZERS_SRC}"

echo ">>> Building tokenizers-cpp ..."
mkdir -p "${TOKENIZERS_BUILD}" && cd "${TOKENIZERS_BUILD}"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="${TOKENIZERS_INSTALL}"

make VERBOSE=1
make install

echo "tokenizers-cpp built & installed to ${TOKENIZERS_INSTALL}"
