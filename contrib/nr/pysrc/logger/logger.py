import calendar
import time
import logging
import os


def configure_logging(args):
    ts = calendar.timegm(time.gmtime())
    log_file_name = f"{args.log_name}_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_folder, log_file_name)),
            logging.StreamHandler()
        ]
    )
