#!/usr/bin/env bash

echo "You are about to remove:"

shopt -s nullglob
ds=(exp/TEST_*/ models/TEST_*/ logs/TEST_*/)
shopt -u nullglob

for d in "${ds[@]}"; do
  echo "${d}"
done

read -p "Confirm (y/N)? " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
  for d in "${ds[@]}"; do
    echo -n "Removing $d ..."
    rm -rf $d
    echo "OK"
  done
else
  # handle exits from shell or function but don't exit interactive shell
  [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
fi
