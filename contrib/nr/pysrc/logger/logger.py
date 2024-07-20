import logging
import os
from pathlib import Path
import sys
from typing import Optional
import structlog

processors = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.StackInfoRenderer(),
    structlog.dev.set_exc_info,
    structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
    # Add callsite parameters.
    # structlog.processors.CallsiteParameterAdder(
    #     {
    #         structlog.processors.CallsiteParameter.FILENAME,
    #         structlog.processors.CallsiteParameter.FUNC_NAME,
    #         structlog.processors.CallsiteParameter.LINENO,
    #     }
    # ),
]

log_level = os.environ["NR_LOG_LEVEL"] if ("NR_LOG_LEVEL" in os.environ) else "DEBUG"

structlog.configure(
    processors=processors + [structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging._nameToLevel[log_level]
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger("neurdb_algserver")
logger.warning("Set logging level", level=log_level)


def configure_logging(logfile: Optional[str]):
    if logfile is None:
        renderer = structlog.dev.ConsoleRenderer()
        logger_factory = structlog.PrintLoggerFactory()
    else:
        renderer = structlog.processors.JSONRenderer()

        # Create the log directory if it doesn't exist
        log_folder = "./"
        os.makedirs(log_folder, exist_ok=True)

        # Create a file handler
        log_file_path = os.path.join(log_folder, logfile)
        logger_factory = structlog.WriteLoggerFactory(
            file=Path(log_file_path).open("a")
        )

    structlog.configure(
        processors=processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
        context_class=dict,
        logger_factory=logger_factory,
        cache_logger_on_first_use=False,
    )
