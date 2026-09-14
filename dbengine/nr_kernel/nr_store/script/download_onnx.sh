#!/bin/bash

# This script downloads ONNX Runtime and Protobuf, and generates onnx_pb.h

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ONNX Runtime
ONNX_DIR="$SCRIPT_DIR/../external/onnx"
if [ -f "$ONNX_DIR/lib/libonnxruntime.so" ]; then
  echo "ONNX Runtime already exists at $ONNX_DIR, skipping download"
else
  rm -rf "$ONNX_DIR"
  mkdir -p "$ONNX_DIR"
  wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
  TEMP_DIR="$ONNX_DIR/onnxruntime_temp"
  mkdir -p "$TEMP_DIR"
  tar -xvzf onnxruntime-linux-x64-1.20.0.tgz -C "$TEMP_DIR"
  mv ${TEMP_DIR}/onnxruntime-linux-x64-1.20.0/* "$ONNX_DIR"
  rm -rf "$TEMP_DIR"
  rm onnxruntime-linux-x64-1.20.0.tgz
  echo "ONNX Runtime downloaded to $ONNX_DIR"
fi

# Protobuf
PROTOBUF_DIR="$SCRIPT_DIR/../external/protobuf"
if [ -f "$PROTOBUF_DIR/protobuf_install/bin/protoc" ]; then
  echo "Protobuf already exists at $PROTOBUF_DIR, skipping build"
else
  rm -rf "$PROTOBUF_DIR"
  mkdir -p "$PROTOBUF_DIR"
  git clone --recurse-submodules https://github.com/protocolbuffers/protobuf.git "$PROTOBUF_DIR"
  cd "$PROTOBUF_DIR"
  git checkout v21.1
  mkdir -p build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX="$PROTOBUF_DIR/protobuf_install" \
           -Dprotobuf_BUILD_TESTS=OFF \
           -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
           -DCMAKE_CXX_FLAGS=-fPIC \
           -DCMAKE_C_FLAGS=-fPIC
  make -j$(nproc)
  make install
  echo "Protobuf built and installed at $PROTOBUF_DIR/protobuf_install"
fi

# generate onnx_pb.h
ONNX_PROTO_DIR="$ONNX_DIR/onnx_protobuf"
if [ -f "$ONNX_PROTO_DIR/build/onnx.pb.h" ]; then
  echo "onnx_pb.h already exists, skipping generation"
else
  rm -rf "$ONNX_PROTO_DIR"
  mkdir -p "$ONNX_PROTO_DIR"
  git clone --recurse-submodules https://github.com/onnx/onnx.git "$ONNX_PROTO_DIR"
  cd "$ONNX_PROTO_DIR"
  git checkout v1.17.0

  mkdir -p "$ONNX_PROTO_DIR/build"
  $PROTOBUF_DIR/protobuf_install/bin/protoc onnx/onnx.proto --proto_path=onnx --cpp_out="$ONNX_PROTO_DIR/build"
  echo "Generated onnx_pb.h in $ONNX_PROTO_DIR/build"
fi
