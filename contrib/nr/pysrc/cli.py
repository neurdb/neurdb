import argparse
import os
from typing import List, NewType, Tuple

import numpy as np
import torch
from logger.logger import logger, configure_logging
import traceback
from shared_config.config import parse_config_arguments
from apps import build_model
from connection import DatabaseModelHandler, NeurDBModelHandler
from utils.dataset import libsvm_dataloader, build_inference_loader

# from utils.io import save_model_weight, load_model_weight
from io import BytesIO

# from cache.model_cache import ModelCache

Error = NewType("Error", str)

# Load config and initialize once
# model_cache = ModelCache()

MODEL_HANDLER = NeurDBModelHandler

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _do_train(
    model_name: str,
    batch_size: int,
    libsvm_file: bytes,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
) -> Tuple[int, Error]:
    try:
        file_obj = BytesIO(libsvm_file)

        train_loader, val_loader, test_loader, nfields, nfeat = libsvm_dataloader(
            batch_size, args.data_loader_worker, file_obj
        )

        builder = build_model(model_name, args)
        builder.model_dimension = (nfeat, nfields)
        builder.train(train_loader, val_loader, test_loader)

        model_id = db.insert_model(builder.model)

        return model_id, None

    except Exception:
        return -1, str(traceback.format_exc())


def _do_inference(
    model_name: str,
    model_id: int,
    libsvm_file: bytes,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
) -> Tuple[List[np.ndarray], Error]:
    try:
        file_obj = BytesIO(libsvm_file)

        inference_loader, nfields, nfeat = build_inference_loader(
            args.data_loader_worker, file_obj
        )

        builder = build_model(model_name, args)

        try:
            builder.model = db.get_model(model_id).to(DEVICE)
        except FileNotFoundError:
            return [], f"model {model_name} not trained yet"

        # check if test data matching model dimension
        # model_nfeat, model_nfields = builder.model_dimension
        # if nfields > model_nfields or nfeat > model_nfeat:
        #     return (
        #         [],
        #         f"model {model_name} trained with nfields = {model_nfields} and nfeat = {model_nfeat}. "
        #         f"cannot handle input with nfields = {nfields} and nfeat = {nfeat}.",
        #     )

        infer_res = builder.inference(inference_loader)
        return infer_res, None

    except Exception:
        return [], str(traceback.format_exc())


def train(
    file_path: str,
    batch_size: int,
    model_name: str,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
) -> int:
    with open(file_path, "rb") as f:
        libsvm_file = f.read()

    model_id, err = _do_train(model_name, batch_size, libsvm_file, args, db)
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return

    print(f"train done. model_id: {model_id}")

    return model_id


def inference(
    file_path: str,
    model_name: str,
    model_id: int,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
) -> List[np.ndarray]:
    with open(file_path, "rb") as f:
        libsvm_file = f.read()

    response, err = _do_inference(model_name, model_id, libsvm_file, args, db)
    if err is not None:
        logger.error(f"inference failed with error: {err}")
        return

    logger.debug(f"inference done. response[0,:100]:")
    logger.debug(response[0][:100] if len(response[0]) >= 100 else response[0])

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.ini")
    parser.add_argument("-l", "--logfile", default=None, type=str)
    parser.add_argument("-t", "--train", action=argparse.BooleanOptionalAction)
    parser.add_argument("-i", "--inference", action=argparse.BooleanOptionalAction)
    parser.add_argument("-m", "--model-id", default=1, type=int)
    args = parser.parse_args()

    configure_logging(args.logfile)

    config_file_path = os.path.abspath(args.config)
    logger.debug(f"loading config file: {config_file_path}")
    config_args = parse_config_arguments(config_file_path)

    db = NeurDBModelHandler(
        {
            "db_name": config_args.db_name,
            "db_user": config_args.db_user,
            "db_host": config_args.db_host,
            "db_port": config_args.db_port,
            # "password": config_args.db_password,
        }
    )
    db.connect()
    logger.debug(f"connected to DB")

    file_path = os.path.abspath("../../../dataset/frappe/test.libsvm")
    model_name = "armnet"
    logger.info(f"file_path={file_path}, model_name={model_name}")

    if not args.train:
        model_id = args.model_id
    else:
        model_id = train(file_path, 32, model_name, config_args, db)

    if args.inference:
        inference(file_path, model_name, model_id, config_args, db)
