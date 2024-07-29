"""Python script to train an ARMNet model."""

import os

import torch
from python.dataloader import table_dataloader
import argparse

from models import build_model
from config import DB_CONFIG
from shared_config.config import parse_config_arguments
from config import logger
import time


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
    logger.debug(f"Data loaded from table {table_name}", nfields=nfields, nfeat=nfeat)

    config_args = parse_config_arguments(os.path.join(os.environ["NEURDBPATH"], "contrib/nr/pysrc/config.ini"))
    config_args.epoch = args.num_epochs
    config_args.state_dict_path = model_path
    config_args.nfield = nfields
    config_args.nfeat = nfeat
    
    builder = build_model("armnet", config_args)
    
    if os.path.exists(model_path):
        logger.info("Model file exists. Use external model params", path=model_path)
    else:
        logger.info("Model file does not exist. Training ...", path=model_path)
        start_time = time.time()
        builder.train(train_loader, val_loader, test_loader)
        time_usage = time.time() - start_time
        logger.info(f"Model trained, time usage = {time_usage}", path=model_path)
    
        logger.info("Saving model ...", path=model_path)
        torch.save(builder.model.state_dict(), model_path)
        logger.info("Model saved", path=model_path)

    start_time = time.time()
    y_pred = builder.inference(test_loader)
    time_usage = time.time() - start_time
    logger.info(f"Inference done, time={time_usage}", y_pred_head=y_pred[:10] if len(y_pred) >= 10 else y_pred)
