from typing import Optional, Tuple, TypedDict
from neurdb.storeman.common import ModelStorage
from torch import nn
import neurdb
from logger.logger import logger


class DBParams(TypedDict):
    db_name: str
    db_user: str
    db_host: str
    db_port: str


class NeurDBModelHandler:
    def __init__(self, db_params: DBParams):
        self._db_params = db_params
        self._conn = neurdb.NeurDB(**self._db_params)

    def close(self):
        if self._conn:
            self._conn.close()
            del self._conn

    def insert_model(self, model: nn.Module) -> int:
        serialized_model = neurdb.ModelSerializer.serialize_model(model)
        return self._conn.save_model(serialized_model)

    def update_layers(
        self, model_id: int, model: ModelStorage, start_layer_id: int
    ) -> int:
        for i, l in enumerate(model.layer_sequence):
            if i < start_layer_id:
                continue

            self._conn.update_model(model_id, i, l.get_pickled())
            logger.debug(f"layer updated", model_id=model_id, layer_id=i)

        return model_id

    def get_model(self, model_id: int) -> ModelStorage:
        storage = self._conn.load_model(model_id).unpack()
        return storage

    def has_model(self, model_id: int) -> bool:
        try:
            self._conn.load_model(model_id)
        except FileNotFoundError:
            return False
        return True
