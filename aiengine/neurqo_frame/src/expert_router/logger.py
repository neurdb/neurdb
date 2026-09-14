# Standard library imports
import logging
import os
from datetime import datetime

# Local/project imports
from common.base_config import BaseConfig


class TrainingLogger:
    def __init__(self, log_dir="logs", log_filename=None):
        """
        Initialize the logger with a specified directory and optional filename.

        Args:
            log_dir (str): Directory where logs will be saved.
            log_filename (str, optional): Custom filename for the log. If None, uses timestamp.
        """
        # Create log directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Generate log filename with timestamp if not provided
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"training_{timestamp}.log"
        self.log_file = os.path.join(self.log_dir, log_filename)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),  # Save to file
                # logging.StreamHandler()  # Also print to console
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Logs will be saved to {self.log_file}")

    def info(self, message):
        """Log an info-level message."""
        self.logger.info(message)

    def get_log_file(self):
        """Return the path to the log file."""
        return self.log_file


plogger = TrainingLogger(log_dir=BaseConfig.LOG_DIR, log_filename=None)
