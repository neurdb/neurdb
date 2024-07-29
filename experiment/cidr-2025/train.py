"""Python script to train an ARMNet model."""

import os

import structlog
import torch
from python.dataloader import table_dataloader
import argparse

from apps import build_model
from config import DB_CONFIG, LOG_LEVEL
from shared_config.config import parse_config_arguments

from neurdb.logger import configure_logging

configure_logging(None)

logger_name = "cidr-baseline"
logger: structlog.stdlib.BoundLogger = structlog.get_logger()
logger = logger.bind(logger=logger_name)
logger.warning("Set logging level", level=LOG_LEVEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ARMNet model")
    parser.add_argument("--table", type=str, help="Name of the table to query")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--model_path", type=str, default="model.h5", help="Path to save the model"
    )
    args = parser.parse_args()

    table_name = args.table
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    model_path = args.model_path

    logger.debug(f"Loading data from table {table_name}...")
    train_loader, val_loader, test_loader, nfields, nfeat = table_dataloader(
        DB_CONFIG, table_name, batch_size
    )
    logger.debug(f"Data loaded from table {table_name}...")

    config_args = parse_config_arguments(os.path.join(os.environ["NEURDBPATH"], "contrib/nr/pysrc/config.ini"))
    config_args.epoch = args.num_epochs
    
    builder = build_model("armnet", config_args)
    builder.model_dimension = nfields, nfeat
    
    if os.path.exists(model_path):
        logger.info(f"Model file exists. Loading ...", path=model_path)
        builder.model.load_state_dict(torch.load(model_path))
        logger.info(f"Model loaded", path=model_path)
    else:
        logger.info(f"Model file does not exist. Training ...", path=model_path)
        builder.train(train_loader, val_loader, test_loader)
        logger.info(f"Model trained", path=model_path)
    
        logger.info(f"Saving model ...", path=model_path)
        torch.save(builder.model.state_dict(), model_path)
        logger.info(f"Model saved", path=model_path)
    
    builder.inference(test_loader)
