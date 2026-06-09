import argparse
import traceback
from typing import Any, List, Optional, Tuple

import numpy as np
from neurdbrt.config import DEVICE
from neurdbrt.dataloader import StreamingDataSet
from neurdbrt.model import build_model
from neurdbrt.repo import ModelRepository

Error = Optional[str]


class Setup:
    def __init__(
        self,
        model_name: str,
        libsvm_data: StreamingDataSet,
        args: argparse.Namespace,
        db: ModelRepository,
        tabpfn_cfg: Optional[dict] = None,
    ) -> None:
        self._model_name = model_name
        self.libsvm_data = libsvm_data
        self._args = args
        self._db = db
        # for the in-context "tabpfn" model: {task_type, stype_hints, id_cols}
        self._tabpfn_cfg = tabpfn_cfg or {}

    async def train(
        self,
        epoch: int,
        train_batch_num: int,
        eva_batch_num: int,
        test_batch_num: int,
        feature_names: List[str],
        target_name: str,
    ) -> Tuple[int, Error]:
        try:
            nfields, nfeat = self.libsvm_data.setup_for_train_task(
                train_batch_num, eva_batch_num, test_batch_num
            )
            self._args.nfields = nfields
            self._args.nfeat = nfeat

            builder = build_model(self._model_name, self._args)
            if self._model_name == "tabpfn":
                builder.configure(**self._tabpfn_cfg)
            await builder.train(
                self.libsvm_data,
                self.libsvm_data,
                self.libsvm_data,
                epoch,
                train_batch_num,
                eva_batch_num,
                test_batch_num,
                feature_names,
                target_name,
            )

            if self._model_name == "tabpfn":
                # in-context model: cache the fitted context in-process keyed by
                # a synthetic model_id (no gradient model to persist to the DB).
                from neurdbrt.model.tabpfn import session_store

                model_id = session_store.new_model_id()
                session_store.put(model_id, builder.model)
            elif self._args.run_model == "in_database":
                if self._model_name == "auto_pipeline":
                    model_id = self._db.insert_list(builder.model)
                else:
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
        feature_names: List[str],
        target_name: str,
    ) -> Tuple[int, Error]:
        try:
            nfields, nfeat = self.libsvm_data.setup_for_train_task(
                train_batch_num, eva_batch_num, test_batch_num
            )
            self._args.nfields = nfields
            self._args.nfeat = nfeat

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
            await builder.train(
                self.libsvm_data,
                self.libsvm_data,
                self.libsvm_data,
                epoch,
                train_batch_num,
                eva_batch_num,
                test_batch_num,
                feature_names,
                target_name,
            )

            model_id = self._db.update_layers(model_id, model_storage, start_layer_id)

            return model_id, None

        except Exception:
            return -1, str(traceback.format_exc())

    async def inference(
        self,
        model_id: int,
        inf_batch_num: int,
        feature_names: List[str],
        target_name: str,
        session_id: str,
    ) -> Tuple[List[List[Any]], Error]:
        try:
            self.libsvm_data.setup_for_inference_task(inf_batch_num)

            builder = build_model(self._model_name, self._args)
            if self._model_name == "tabpfn":
                from neurdbrt.model.tabpfn import session_store

                cached = session_store.get(model_id)
                if cached is None:
                    return [], (
                        f"tabpfn context for model_id={model_id} not found "
                        f"(train phase did not run on this engine process)"
                    )
                builder.model = cached
            else:
                try:
                    if self._model_name == "auto_pipeline":
                        builder.model = self._db.get_list(model_id)
                    else:
                        builder.model = (
                            self._db.get_model(model_id).to_model().to(DEVICE)
                        )
                except FileNotFoundError:
                    return [], f"model {self._model_name} not trained yet"

            infer_res = await builder.inference(
                self.libsvm_data, inf_batch_num, feature_names, target_name, session_id
            )
            return infer_res, None

        except Exception:
            return [], str(traceback.format_exc())
