import argparse
import os
from typing import List, Optional, Tuple
from io import BytesIO

import numpy as np
import torch
from logger.logger import logger, configure_logging
import traceback
from shared_config.config import parse_config_arguments
from models import build_model
from connection import NeurDBModelHandler
from dataloader.libsvm import libsvm_dataloader, build_inference_loader

# from cache.model_cache import ModelCache

Error = Optional[str]

# Load config and initialize once
# model_cache = ModelCache()

MODEL_HANDLER = NeurDBModelHandler

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Setup:
    def __init__(
        self,
        model_name: str,
        libsvm_data: str,
        args: argparse.Namespace,
        db: MODEL_HANDLER,
    ) -> None:
        self._model_name = model_name
        self.libsvm_data = libsvm_data
        self._args = args
        self._db = db

    # def _load_dataset(self) -> bytes:
    #     with open(self._dataset_file_path, "rb") as f:
    #         return f.read()

    def train(self, batch_size: int) -> Tuple[int, Error]:
        try:
            train_loader, val_loader, test_loader, nfields, nfeat = libsvm_dataloader(
                batch_size, self._args.data_loader_worker, self.libsvm_data
            )

            builder = build_model(self._model_name, self._args)
            builder.model_dimension = (nfeat, nfields)
            builder.train(train_loader, val_loader, test_loader)

            model_id = self._db.insert_model(builder.model)
            return model_id, None

        except Exception:
            return -1, str(traceback.format_exc())

    def finetune(
        self, model_id: int, batch_size: int, start_layer_id: int
    ) -> Tuple[int, Error]:
        try:
            train_loader, val_loader, test_loader, nfields, nfeat = libsvm_dataloader(
                batch_size, self._args.data_loader_worker, self.libsvm_data
            )

            try:
                builder = build_model(self._model_name, self._args)
                model_storage = self._db.get_model(model_id)
                model = model_storage.to_model()
            except FileNotFoundError:
                return -1, f"model {self._model_name} not trained yet"

            for i, (_, layer) in enumerate(model.named_children()):
                if i < start_layer_id:
                    layer.requires_grad_(False)

            builder.model = model.to(DEVICE)
            builder.model_dimension = (nfeat, nfields)
            builder.train(train_loader, val_loader, test_loader)

            model_id = self._db.update_layers(model_id, model_storage, start_layer_id)

            return model_id, None

        except Exception:
            return -1, str(traceback.format_exc())

    def inference(
        self, model_id: int, batch_size: int
    ) -> Tuple[List[np.ndarray], Error]:
        try:
            inference_loader, nfields, nfeat = build_inference_loader(
                self._args.data_loader_worker, self.libsvm_data, batch_size
            )

            builder = build_model(self._model_name, self._args)
            try:
                builder.model = self._db.get_model(model_id).to_model().to(DEVICE)
            except FileNotFoundError:
                return [], f"model {self._model_name} not trained yet"

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
    model_name: str,
    training_libsvm: str,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
    batch_size: int,
) -> int:
    logger.info(f"start", task="train", model_name=model_name, batch_size=batch_size)

    s = Setup(model_name, training_libsvm, args, db)

    model_id, err = s.train(batch_size)
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    logger.info(f"done", task="train", model_id=model_id)
    return model_id


def finetune(
    model_name: str,
    finetune_libsvm: str,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
    model_id: int,
    batch_size: int,
) -> int:
    logger.info(
        f"start",
        task="finetune",
        model_id=model_id,
        model_name=model_name,
        batch_size=batch_size,
    )

    s = Setup(model_name, finetune_libsvm, args, db)

    model_id, err = s.finetune(model_id, batch_size, start_layer_id=5)
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    logger.info("done", task="finetune", model_id=model_id)
    return model_id


def inference(
    model_name: str,
    inference_libsvm: str,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
    model_id: int,
    batch_size: int,
) -> List[np.ndarray]:
    logger.info(
        f"start",
        task="finetune",
        model_id=model_id,
        model_name=model_name,
        batch_size=batch_size,
    )

    s = Setup(model_name, inference_libsvm, args, db)

    response, err = s.inference(model_id, batch_size)
    if err is not None:
        logger.error(f"inference failed with error: {err}")
        return []
    logger.debug(f"inference done. response[0,:100]:")
    logger.debug(response[0][:100] if len(response[0]) >= 100 else response[0])

    logger.info("done", task="inference", model_id=model_id)
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
    logger.debug(f"connected to DB")

    file_path = os.path.abspath("../../../dataset/frappe/test.libsvm")
    model_name = "armnet"
    logger.info(f"file_path={file_path}, model_name={model_name}")

    if not args.train:
        logger.info("mode: inference")
        model_id = args.model_id
    elif db.has_model(args.model_id):
        logger.info("model exists. mode: finetune")
        model_id = finetune(model_name, file_path, config_args, db, args.model_id, 32)
    else:
        logger.info("model does not exist. mode: train")
        model_id = train(model_name, file_path, config_args, db, 32)

    if args.inference:
        inference(model_name, file_path, config_args, db, model_id)
