import argparse
from typing import List, Tuple, Optional
import numpy as np
import traceback
from dataloader.steam_libsvm_dataset import StreamingDataSet
from connection import NeurDBModelHandler
from models import build_model
from shared_config.config import DEVICE

Error = Optional[str]


class Setup:
    def __init__(
        self,
        model_name: str,
        libsvm_data: StreamingDataSet,
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

    async def train(
        self, epoch: int, train_batch_num: int, eva_batch_num: int, test_batch_num: int
    ) -> Tuple[int, Error]:
        try:
            nfields, nfeat = self.libsvm_data.setup_for_train_task(
                train_batch_num, eva_batch_num, test_batch_num
            )

            builder = build_model(self._model_name, self._args)
            builder.model_dimension = (nfeat, nfields)
            await builder.train(
                self.libsvm_data,
                self.libsvm_data,
                self.libsvm_data,
                epoch,
                train_batch_num,
                eva_batch_num,
                test_batch_num,
            )

            if self._args.run_model == "in_database":
                model_id = self._db.insert_model(builder.model)
            else:
                model_id = -1
            return model_id, None

        except Exception:
            print(traceback.format_exc())
            return -1, str(traceback.format_exc())

    async def finetune(
        self,
        model_id: int,
        start_layer_id: int,
        epoch: int,
        train_batch_num: int,
        eva_batch_num: int,
        test_batch_num: int,
    ) -> Tuple[int, Error]:
        try:
            nfields, nfeat = self.libsvm_data.setup_for_train_task(
                train_batch_num, eva_batch_num, test_batch_num
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
            await builder.train(
                self.libsvm_data,
                self.libsvm_data,
                self.libsvm_data,
                epoch,
                train_batch_num,
                eva_batch_num,
                test_batch_num,
            )

            model_id = self._db.update_layers(model_id, model_storage, start_layer_id)

            return model_id, None

        except Exception:
            return -1, str(traceback.format_exc())

    async def inference(
        self, model_id: int, inf_batch_num: int
    ) -> Tuple[List[np.ndarray], Error]:
        try:
            self.libsvm_data.setup_for_inference_task(inf_batch_num)

            builder = build_model(self._model_name, self._args)
            try:
                builder.model = self._db.get_model(model_id).to_model().to(DEVICE)
            except FileNotFoundError:
                return [], f"model {self._model_name} not trained yet"

            infer_res = await builder.inference(self.libsvm_data, inf_batch_num)
            return infer_res, None

        except Exception:
            return [], str(traceback.format_exc())

    def register_model(
        self, model_id: int, model_name: str, features: List[str], target: str
    ):
        try:
            self._db.register_model(model_id, model_name, features, target)
            return None
        except Exception:
            return str(traceback.format_exc())
