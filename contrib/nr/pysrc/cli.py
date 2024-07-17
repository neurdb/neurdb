import argparse
import os
from typing import List, Optional, Tuple
from io import BytesIO

import numpy as np
import torch
from logger.logger import logger, configure_logging
import traceback
from shared_config.config import parse_config_arguments
from apps import build_model
from connection import DatabaseModelHandler, NeurDBModelHandler
from utils.dataset import libsvm_dataloader, build_inference_loader

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
        dataset_file_path: str,
        args: argparse.Namespace,
        db: MODEL_HANDLER,
    ) -> None:
        self._model_name = model_name
        self._dataset_file_path = dataset_file_path
        self._args = args
        self._db = db

        self._dataset_file = self._load_dataset()

    def _load_dataset(self) -> bytes:
        with open(self._dataset_file_path, "rb") as f:
            return f.read()

    def _make_model_args(self, nfields: int, nfeat: int) -> dict:
        # TODO: I think we should change NeurDB APIs here. It should infer the
        # model args automatically, or store the args into DB (model table),
        # instead of manually input
        args = self._args

        return {
            "model": {
                "nfield": nfields,
                "nfeat": nfeat,
                "nemb": args.nemb,
                "nhead": args.nattn_head,
                "alpha": args.alpha,
                "nhid": args.h,
                "mlp_nlayer": args.mlp_nlayer,
                "mlp_nhid": args.mlp_nhid,
                "dropout": args.dropout,
                "ensemble": args.ensemble,
                "deep_nlayer": args.dnn_nlayer,
                "deep_nhid": args.dnn_nhid,
            },
            "layers": {
                "embedding": {"nfeat": nfeat, "nemb": args.nemb},
                "attn_layer": {
                    "nhead": args.nattn_head,
                    "nfield": nfields,
                    "nemb": args.nemb,
                    "d_k": args.nemb,
                    "nhid": args.h,
                    "alpha": args.alpha,
                },
                "arm_bn": {"num_features": args.nattn_head * args.h},
                "mlp": {
                    "ninput": args.nattn_head * args.h * args.nemb,
                    "nlayers": args.mlp_nlayer,
                    "nhid": args.mlp_nhid,
                    "dropout": args.dropout,
                },
                "deep_embedding": {
                    "nfeat": nfeat,
                    "nemb": args.nemb,
                },
                "deep_mlp": {
                    "ninput": nfields * args.nemb,
                    "nlayers": args.dnn_nlayer,
                    "nhid": args.dnn_nhid,
                    "dropout": args.dropout,
                },
                "ensemble_layer": {"in_features": 2, "out_features": 1},
            },
        }

    def train(self, batch_size: int) -> Tuple[int, Error]:
        try:
            train_loader, val_loader, test_loader, nfields, nfeat = libsvm_dataloader(
                batch_size,
                self._args.data_loader_worker,
                BytesIO(self._dataset_file),
            )

            builder = build_model(self._model_name, self._args)
            builder.model_dimension = (nfeat, nfields)
            builder.train(train_loader, val_loader, test_loader)

            model_id = self._db.insert_model(builder.model)

            return model_id, None

        except Exception:
            return -1, str(traceback.format_exc())
        
    def finetune(self, model_id: int, batch_size: int, start_layer_id: int) -> Tuple[int, Error]:
        try:
            train_loader, val_loader, test_loader, nfields, nfeat = libsvm_dataloader(
                batch_size,
                self._args.data_loader_worker,
                BytesIO(self._dataset_file),
            )

            try:
                builder = build_model(model_name, self._args)
                model_args = self._make_model_args(nfields, nfeat)
                model_storage = self._db.get_model(model_id, model_args)
                model = model_storage.to_model()
            except FileNotFoundError:
                return -1, f"model {model_name} not trained yet"
            
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

    def inference(self, model_id: int) -> Tuple[List[np.ndarray], Error]:
        try:
            inference_loader, nfields, nfeat = build_inference_loader(
                self._args.data_loader_worker, BytesIO(self._dataset_file)
            )

            builder = build_model(model_name, self._args)
            model_args = self._make_model_args(nfields, nfeat)
            try:
                builder.model = self._db.get_model(model_id, model_args).to_model().to(DEVICE)
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
    model_name: str,
    file_path: str,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
    batch_size: int,
) -> int:
    s = Setup(model_name, file_path, args, db)

    model_id, err = s.train(batch_size)
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    print(f"train done. model_id: {model_id}")

    return model_id


def finetune(
    model_name: str,
    file_path: str,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
    model_id: int,
    batch_size: int,
) -> int:
    s = Setup(model_name, file_path, args, db)

    model_id, err = s.finetune(model_id, batch_size, start_layer_id=5)
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    print(f"finetune done. model_id: {model_id}")

    return model_id

def inference(
    model_name: str,
    file_path: str,
    args: argparse.Namespace,
    db: MODEL_HANDLER,
    model_id: int,
) -> List[np.ndarray]:
    s = Setup(model_name, file_path, args, db)

    response, err = s.inference(model_id)
    if err is not None:
        logger.error(f"inference failed with error: {err}")
        return []

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
