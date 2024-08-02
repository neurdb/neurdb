import os
import structlog
from neurdb.logger import configure_logging


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


DATALOADER_NUM_WORKERS = 0  # _try_get_cpu_count()
LOG_LEVEL = os.environ["NR_LOG_LEVEL"] if ("NR_LOG_LEVEL" in os.environ) else "DEBUG"


def _config_logger():
    configure_logging(None, LOG_LEVEL)

    logger_name = "cidr-baseline"
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    logger = logger.bind(logger=logger_name)
    logger.warning("Set logging level", level=LOG_LEVEL)

    return logger


logger = _config_logger()
