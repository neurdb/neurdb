#!/usr/bin/env bash

PREFIX=${PREFIX:-$NEURDBPATH/external/ctxpipe}

# Make external folder
mkdir -p $PREFIX

# Download GTE-large
if [ ! -d "$PREFIX/gte-large" ]; then
  echo "Downloading GTE-large..."
  git lfs install
  status=$?
  if [ $status -ne 0 ]; then
    echo "Git LFS installation failed with status $status. Please install Git LFS and try again."
    exit 1
  fi
  git clone --depth 1 https://huggingface.co/thenlper/gte-large $PREFIX/gte-large
  rm -rf $PREFIX/gte-large/.git
  # The following two formats are not used by the module
  rm -rf $PREFIX/gte-large/onnx
  rm -rf $PREFIX/gte-large/openvino
  echo "Done."
else
  echo "GTE-large already exists. Skipping download."
fi

# Download CtxPipe agent weights
if [ ! -d "$PREFIX/model" ]; then
  echo "Downloading CtxPipe models..."
  temp_dir=`mktemp -d /tmp/neurdb-XXXXXXXX`
  git clone --depth 1 https://github.com/ctxpipe/ctxpipe $temp_dir
  mv $temp_dir/models/ctxpipe-3linear $PREFIX/model
  echo "Done."
else
  echo "CtxPipe models already exists. Skipping download."
fi

echo "Auto pipeline can now be used."
