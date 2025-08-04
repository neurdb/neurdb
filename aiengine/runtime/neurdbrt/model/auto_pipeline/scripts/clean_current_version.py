import sys

sys.path.insert(0, ".")

import os
import shutil

from config import default_config as conf

version = conf.version

print(f"Current version: {conf.version}")

print(f"Cleaning directories: ")

ds = []
for dp in ["logs", "models", "exp"]:
    d = os.path.abspath(os.path.join(dp, version))
    print(d)
    ds.append(d)

answer = input("Confirm (y/N)? ")
if answer.lower() in ["y", "yes"]:
    print("Cleaning...", end="")

    for d in ds:
        shutil.rmtree(d, ignore_errors=True)

    print("OK")
else:
    print("Cancelled")
