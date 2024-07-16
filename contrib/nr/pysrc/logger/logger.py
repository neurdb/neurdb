import logging
import os


# Define a function to configure the logger
def configure_logging():
    log_folder = './'
    log_file_name = "app.log"
    log_file_path = os.path.join(log_folder, log_file_name)

    # Create the log directory if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)

    # Create a file handler
    fh = logging.FileHandler(log_file_path, mode='a')
    fh.setLevel(logging.DEBUG)

    # Create a stream handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('[ %(levelname)s] %(asctime)s - %(message)s', datefmt='%m-%d-%y %H:%M:%S')

    # Add formatter to handlers
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)


# Create a global logger
logger = logging.getLogger('neurdb_algserver')
logger.setLevel(logging.DEBUG)
configure_logging()
