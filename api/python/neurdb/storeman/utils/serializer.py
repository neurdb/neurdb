import pickle
from io import BytesIO

import torch

from ..common import ModelStorage


class ModelSerializer:
    """
    Class to serialize and deserialize PyTorch models
    """

    # ---------------- Public APIs ----------------
    @staticmethod
    def serialize_model(model: torch.nn.Module) -> ModelStorage.Pickled:
        """
        Serialize a PyTorch model to a pickled representation
        :param model: PyTorch model in nn.Module format
        :return: ModelStorage.Pickled, the pickled representation of the model in storage
        """
        return ModelStorage.from_model(model).get_pickled()

    @staticmethod
    def deserialize_model(model_pickled: ModelStorage.Pickled) -> torch.nn.Module:
        """
        Deserialize a pickled representation of a PyTorch model to a PyTorch model
        :param model_pickled: The pickled representation of the model in storage
        :return: PyTorch model in nn.Module format
        """
        return model_pickled.unpack().to_model()

    # ---------------- Private APIs ----------------
    @staticmethod
    def _deserialize_layer(layer_bytes: bytes) -> torch.nn.Module:
        """
        Deserialize a layer from its pickled representation, this is not exposed, only model-level
        apis are exposed from storage package
        :param layer_bytes: The pickled representation of the layer
        :return: PyTorch layer in nn.Module format
        """
        buffer = BytesIO(layer_bytes)
        meta = pickle.load(buffer)
        layer = meta["class"](**meta["init_params"])
        layer.load_state_dict(meta["state_dict"])
        return layer
