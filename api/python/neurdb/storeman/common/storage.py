import pickle
from abc import ABCMeta
from io import BytesIO
from typing import List, Optional, Type, Union

from torch import nn


class Pickled(metaclass=ABCMeta):
    """
    Abstract base class for the pickled representation of the model
    """

    @property
    def model_meta_pickled(self) -> bytes:
        """
        Get the pickled representation of metadata
        @return: pickled representation of metadata
        """
        raise NotImplementedError

    @model_meta_pickled.setter
    def model_meta_pickled(self, value: bytes):
        raise NotImplementedError

    @property
    def layer_sequence_pickled(self) -> List[bytes]:
        """
        Get the pickled representation of each layer
        @return: list of pickled representation of each layer
        """
        raise NotImplementedError

    @layer_sequence_pickled.setter
    def layer_sequence_pickled(self, value: List[bytes]):
        raise NotImplementedError


class LayerStorage:
    """
    Class to store the PyTorch layer
    """

    def __init__(
        self,
        layer_class: Type[nn.Module],
        init_params: dict,
        state_dict: dict,
        name: Optional[str] = None,
    ):
        """
        @param layer_class: Class of the layer, e.g. nn.Linear
        @param init_params: The initial parameters of the layer, e.g. {'in_features': 10, 'out_features': 2} for nn.Linear
        @param state_dict: The state dictionary of the layer, e.g. layer.state_dict()
        @param name: The name of the layer, this is optional, but must be provided if the layer is part of a model
        """
        self.layer_class = layer_class
        # self.init_params = Utils.clean_init_params(init_params)
        self.init_params = init_params
        self.state_dict = state_dict
        self.name = name

    def __str__(self):
        return f"LayerStorage({self.layer_class})"

    def get_pickled(self) -> bytes:
        """
        Get the pickled representation of the layer
        @return: pickled representation of the layer
        """
        buffer = BytesIO()
        pickle.dump(self, buffer)
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def unpickle(pickled: bytes) -> "LayerStorage":
        """
        Unpickle the pickled representation of the layer, this api is not recommended to use
        @param pickled: pickled representation of the layer
        @return: LayerStorage object
        """
        buffer = BytesIO(pickled)
        return pickle.load(buffer)

    @staticmethod
    def from_layer(layer: nn.Module, name: Optional[str] = None):
        """
        Create a LayerStorage object from the PyTorch layer
        @param layer: The PyTorch layer in nn.Module format
        @param name: The name of the layer, this is optional, but must be provided if the layer is part of a model
        @return: LayerStorage object
        """
        return LayerStorage(
            layer_class=layer.__class__,
            init_params=layer.__dict__,
            state_dict=layer.state_dict(),
            name=name,
        )

    def to_layer(self) -> nn.Module:
        """
        Convert the LayerStorage object to the PyTorch layer
        @return: A nn.Module object
        """
        # params = self.init_params["layers"][self.name]
        # layer = self.layer_class(**filter_args(params, self.layer_class))
        layer = self.layer_class.__new__(self.layer_class)
        layer.__dict__.update(self.init_params)
        layer.load_state_dict(self.state_dict)
        return layer


class LayerSequenceStorage(List[LayerStorage]):
    """
    Class to store a sequence of layers
    This class can be considered as a helper class to store the sequence of layers in a model
    """

    def __init__(self, layers: List[LayerStorage]):
        super().__init__()
        self._layers = layers

    def __str__(self):
        return f"LayerSequenceStorage({len(self._layers)} layers)"

    def __iter__(self):
        return iter(self._layers)

    def append(self, layer: LayerStorage):
        self._layers.append(layer)

    def insert(self, index: int, layer: LayerStorage):
        self._layers.insert(index, layer)

    def get_pickled(self) -> List[bytes]:
        """
        Get the pickled representation of each layer in the sequence
        @return: list of pickled representation of each layer
        """
        return [layer.get_pickled() for layer in self._layers]

    @staticmethod
    def from_list(layers: list):
        """
        Create a LayerSequenceStorage object from a list of layers
        @param layers: list of layers, each layer can be a nn.Module (if name is not provided) or
        a tuple <name, nn.Module> (if name is provided)

        PS: I implemented this way because python does not support method overloading

        @return: A LayerSequenceStorage object
        """
        if all(isinstance(layer, nn.Module) for layer in layers):
            return LayerSequenceStorage(
                [LayerStorage.from_layer(layer) for layer in layers]
            )
        elif all(isinstance(layer, tuple) for layer in layers):
            return LayerSequenceStorage(
                [LayerStorage.from_layer(layer, name) for name, layer in layers]
            )
        else:
            raise ValueError("Invalid list of layers")

    @staticmethod
    def from_dict(layers: dict):
        """
        Create a LayerSequenceStorage object from a dictionary of layers
        @param layers: dictionary of layers, each layer can be a nn.Module (if name is not provided) or
        a tuple <name, nn.Module> (if name is provided)
        @return: A LayerSequenceStorage object
        """
        # check if the keys are from 0 to n-1
        if set(layers.keys()) != set(range(len(layers))):
            raise ValueError(
                "Keys of the layers should be from 0 to n-1, no missing or duplicated."
            )
        # sort the layers by key
        if all(isinstance(layer, nn.Module) for layer in layers.values()):
            sorted_layers = [
                LayerStorage.from_layer(layer) for layer in layers.values()
            ]
        elif all(isinstance(layer, tuple) for layer in layers.values()):
            sorted_layers = [
                LayerStorage.from_layer(layer, name)
                for _, (name, layer) in sorted(layers.items())
            ]
        else:
            raise ValueError("Invalid dictionary of layers")
        return LayerSequenceStorage.from_list(sorted_layers)


class ModelStorage:
    """
    Class to store the PyTorch model
    It has an inner class Pickled to store the pickled representation of the model
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        init_params: dict,
        layers: LayerSequenceStorage = LayerSequenceStorage([]),
    ):
        if LayerSequenceStorage is None:
            layers = LayerSequenceStorage([])
        self.model_class = model_class
        # self.init_params = Utils.clean_init_params(init_params)
        self.init_params = init_params
        self.layer_sequence = layers

    def __str__(self):
        return f"ModelStorage({self.model_class})"

    @staticmethod
    def from_model(model: nn.Module):
        """
        Create a ModelStorage object from a nn.Module object
        @param model: PyTorch model in nn.Module format
        @return: ModelStorage object
        """
        return ModelStorage(
            model_class=model.__class__,
            init_params=model.__dict__,
            layers=LayerSequenceStorage.from_list(list(model.named_children())),
        )

    def append_layer(self, layer: LayerStorage):
        self.layer_sequence.append(layer)

    def insert_layer(self, index: int, layer: LayerStorage):
        self.layer_sequence.insert(index, layer)

    def to_model(self) -> nn.Module:
        """
        Convert the ModelStorage object to the PyTorch model
        @return: A PyTorch model in nn.Module format
        """
        model = self.model_class.__new__(self.model_class)
        model.__dict__.update(self.init_params)
        for layer in self.layer_sequence:
            if layer.name is None:
                raise ValueError("Layer name should not be None")
            model.add_module(layer.name, layer.to_layer())
        return model

    def get_pickled(self) -> "Pickled":
        """
        Get the pickled representation of the model
        :@return: Pickled object
        """
        return ModelStorage.PickledModel(self)

    class PickledModel(Pickled):
        """
        Inner class to support the pickling of the ModelStorage object
        """

        def __init__(
            self,
            model: Union["ModelStorage", bytes],
            layer_sequence_pickled: Optional[List[bytes]] = None,
        ):
            """
            Constructor for the Pickled class
            @param model: ModelStorage object of the model
            or
            @param model: pickled representation of the model in bytes
            @param layer_sequence_pickled: list of pickled representation of each layer in the model

            @note: This implementation is a roundabout way to support constructor overloading in Python
            """
            if isinstance(model, bytes) and layer_sequence_pickled is not None:
                self.model_meta_pickled = model
                self.layer_sequence_pickled = layer_sequence_pickled
            elif isinstance(model, ModelStorage):
                self.model_meta_pickled = pickle.dumps(
                    {"class": model.model_class, "init_params": model.init_params}
                )
                self.layer_sequence_pickled = model.layer_sequence.get_pickled()
            else:
                raise ValueError(
                    "Invalid arguments for ModelStorage.Pickled constructor"
                )

        def unpack(self) -> "ModelStorage":
            """
            Unpack the pickled representation of the model
            @return: ModelStorage object
            """
            return ModelStorage(
                model_class=pickle.loads(self.model_meta_pickled)["class"],
                init_params=pickle.loads(self.model_meta_pickled)["init_params"],
                layers=LayerSequenceStorage(
                    [
                        pickle.loads(layer_pickled)
                        for layer_pickled in self.layer_sequence_pickled
                    ]
                ),
            )


class PickledList(Pickled):
    def __init__(self, model_meta_pickled: bytes, layer_sequence_pickled: List[bytes]):
        self._model_meta_pickled = model_meta_pickled
        self._layer_sequence_pickled = layer_sequence_pickled
        super().__init__()

    @property
    def model_meta_pickled(self) -> bytes:
        return self._model_meta_pickled

    @model_meta_pickled.setter
    def model_meta_pickled(self, value: bytes):
        self._model_meta_pickled = value

    @property
    def layer_sequence_pickled(self) -> List[bytes]:
        return self._layer_sequence_pickled

    @layer_sequence_pickled.setter
    def layer_sequence_pickled(self, value: List[bytes]):
        self._layer_sequence_pickled = value


class Utils:
    """
    Utility class for the storage module
    """

    @staticmethod
    def clean_init_params(params: dict) -> dict:
        """
        Clean the dictionary of the parameters, removing the private attributes and functions
        @param params: parameters dictionary
        @return: cleaned parameters dictionary
        """
        return {
            k: v
            for k, v in params.items()
            if not k.startswith("_") and not callable(v) and k != "training"
        }
