from datetime import datetime

from multipledispatch import dispatch


class ModelEntity:
    """
    Model relation entity in the database
    @note primary key: model_id
    """

    @dispatch(int, bytes)
    def __init__(self, model_id: int, model_meta: bytes):
        """
        @param model_id: model id, serial primary key
        @param model_meta: model meta data
        """
        self.model_id = model_id
        self.model_meta = model_meta

    @dispatch(bytes)
    def __init__(self, model_meta: bytes):
        """
        @param model_meta: model meta data
        """
        self.model_id = None
        self.model_meta = model_meta


class LayerEntity:
    """
    Layer relation entity in the database
    @note primary key: model_id, layer_id, create_time
    """

    @dispatch(int, int, datetime, bytes)
    def __init__(self, model_id: int, layer_id: int, create_time: datetime, layer_data: bytes):
        """
        @param model_id: model id (foreign key)
        @param layer_id: layer id
        @param create_time: creation time
        @param layer_data: layer data
        """
        self.model_id = model_id
        self.layer_id = layer_id
        self.create_time = create_time
        self.layer_data = layer_data

    # This __init__ is commented out because it is dangerous to set model_id to None for a LayerEntity
    # @dispatch(int, datetime, bytes)
    # def __init__(self, layer_id: int, create_time: datetime, layer_data: bytes):
    #     """
    #     @param model_id: model id (foreign key)
    #     @param create_time: creation time
    #     @param layer_data: layer data
    #     """
    #     self.model_id = None
    #     self.layer_id = layer_id
    #     self.create_time = create_time
    #     self.layer_data = layer_data
