#!/usr/bin/env bash

for f in gram.c gram.h scan.c; do
    rm -v "../dbengine/src/backend/parser/$f"
done
