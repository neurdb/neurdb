import logging
import os
from typing import Optional


# Define a function to configure the logger
def configure_logging(logfile: Optional[str]):
    # Create a formatter
    formatter = logging.Formatter(
        "[ %(levelname)s] %(asctime)s - %(message)s", datefmt="%m-%d-%y %H:%M:%S"
    )

    # Create a stream handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if logfile:
        # Create the log directory if it doesn't exist
        log_folder = "./"
        os.makedirs(log_folder, exist_ok=True)

        # Create a file handler
        log_file_path = os.path.join(log_folder, logfile)
        fh = logging.FileHandler(log_file_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


# Create a global logger
logger = logging.getLogger("neurdb_algserver")
logger.setLevel(logging.DEBUG)
