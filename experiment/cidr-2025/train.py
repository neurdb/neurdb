"""Python script to train an ARMNet model."""

from python.dataloader import table_dataloader
import argparse

from apps import build_model

from neurdb.logger import configure_logging
from neurdb.logger import logger

configure_logging(None)

CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "host": "127.0.0.1",
    "port": "5432",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ARMNet model")
    parser.add_argument("--table", type=str, help="Name of the table to query")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model_path", type=str, default="model.h5", help="Path to save the model")
    args = parser.parse_args()

    table_name = args.table
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    model_path = args.model_path

    logger.debug(f"Loading data from table {table_name}...")
    train_loader, val_loader, test_loader, nfields, nfeat = table_dataloader(CONFIG, table_name, batch_size)
    logger.debug(f"Data loaded from table {table_name}...")

    builder = build_model("armnet", ) # TODO: Add building arguments
