import argparse
from typing import List, Tuple, Optional
import numpy as np
import traceback
from dataloader.libsvm import libsvm_dataloader, build_inference_loader
from connection import NeurDBModelHandler
from models import build_model
from shared_config.config import DEVICE

Error = Optional[str]


class Setup:
    def __init__(
            self,
            model_name: str,
            libsvm_data: str,
            args: argparse.Namespace,
            db: NeurDBModelHandler,
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
                batch_size,
                self._args.data_loader_worker,
                self.libsvm_data
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
                self.libsvm_data
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

    def inference(self, model_id: int, batch_size: int) -> Tuple[List[np.ndarray], Error]:
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
