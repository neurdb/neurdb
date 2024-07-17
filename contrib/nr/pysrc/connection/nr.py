from typing import Tuple, TypedDict
from torch import nn
import neurdb


class DBParams(TypedDict):
    db_name: str
    db_user: str
    db_host: str
    db_port: str


class NeurDBModelHandler:
    def __init__(self, db_params: DBParams):
        self._db_params = db_params
        self._conn = None

    def connect(self):
        self._conn = neurdb.NeurDB(**self._db_params)

    def close(self):
        if self._conn:
            self._conn.close()
            del self._conn
            self._conn = None

    def insert_model(self, model: nn.Module) -> int:
        serialized_model = neurdb.ModelSerializer.serialize_model(model)
        return self._conn.save_model(serialized_model)

    def get_model(self, model_id: int) -> nn.Module:
        storage = self._conn.load_model(model_id).unpack()
        return storage.to_model()
