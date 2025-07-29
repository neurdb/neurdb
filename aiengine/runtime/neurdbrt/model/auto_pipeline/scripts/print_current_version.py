import sys

sys.path.insert(0, ".")


from config import default_config as conf

version = conf.version

print(f"{conf.version}")
