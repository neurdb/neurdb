#!/usr/bin/env bash

for f in gram.c gram.h scan.c; do
    rm -v "$PWD/src/backend/parser/$f"
done
