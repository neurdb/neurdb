import os


DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "host": "127.0.0.1",
    "port": "5432",
}


def _try_get_cpu_count():
    nw = os.cpu_count()
    if nw is None:
        nw = 1
    return nw


DATALOADER_NUM_WORKERS = _try_get_cpu_count()
LOG_LEVEL = "DEBUG"