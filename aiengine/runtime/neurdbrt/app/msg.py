import json
import uuid
from abc import ABCMeta, abstractmethod

from neurdbrt import utils


class Request(metaclass=ABCMeta):
    def __init__(self, data_json: dict) -> None:
        self._data = data_json
        if not self._is_legal_request(data_json):
            raise ValueError("bad request")

    @abstractmethod
    def _is_legal_request(self, data_json) -> bool:
        raise NotImplementedError


class Response(metaclass=ABCMeta):
    @abstractmethod
    def to_json(self) -> str:
        raise NotImplementedError


class SetupRequest(Request):
    def __init__(self, data_json: dict) -> None:
        super().__init__(data_json)
        self._session_id = str(uuid.uuid4())

    def _is_legal_request(self, data_json) -> bool:
        return True

    @property
    def session_id(self):
        return self._session_id


class SetupResponse(Response):
    def __init__(self, session_id) -> None:
        self._session_id = session_id

    def to_json(self):
        return json.dumps(
            {"version": 1, "event": "ack_setup", "sessionId": self._session_id}
        )


class DisconnectRequest(Request):
    def __init__(self, data_json: dict) -> None:
        super().__init__(data_json)
        self._session_id = data_json.get("sessionId")

    def _is_legal_request(self, data_json) -> bool:
        if not isinstance(data_json.get("sessionId"), str):
            return False
        return True

    @property
    def session_id(self):
        return self._session_id


class DisconnectResponse(Response):
    def __init__(self, session_id) -> None:
        self._session_id = session_id

    def to_json(self) -> str:
        return json.dumps(
            {"version": 1, "event": "ack_disconnect", "sessionId": self._session_id}
        )


class ModelResultResponse(Response):
    def __init__(self, session_id, model_id) -> None:
        self._session_id = session_id
        self._model_id = model_id

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "event": "result",
                "sessionId": self._session_id,
                "payload": "Task completed",
                "model_id": self._model_id,
            }
        )


class InferenceResultResponse(Response):
    def __init__(self, session_id, result) -> None:
        self._session_id = session_id
        self._result = result

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "event": "result",
                "sessionId": self._session_id,
                "payload": "Inference completed",
                "byte": utils.flatten_2d_array(self._result),
            }
        )


class TaskRequest(Request):
    """Value object"""

    def __init__(self, data_json: dict, is_inference: bool) -> None:
        self._is_inference = is_inference
        super().__init__(data_json)

        self._session_id = data_json["sessionId"]
        self._n_feat = data_json["nFeat"]
        self._n_field = data_json["nField"]
        self._cache_size = data_json["cacheSize"]
        if is_inference:
            self._total_batch_num = data_json["spec"]["nBatch"]
        else:
            self._total_batch_num = (
                data_json["spec"]["nBatchTrain"]
                + data_json["spec"]["nBatchEval"]
                + data_json["spec"]["nBatchTest"]
            )

    @property
    def session_id(self):
        return self._session_id

    @property
    def n_feat(self):
        return self._n_feat

    @property
    def n_field(self):
        return self._n_field

    @property
    def total_batch_num(self):
        return self._total_batch_num

    @property
    def cache_size(self):
        return self._cache_size

    def _is_legal_request(self, data_json) -> bool:
        if not isinstance(data_json["sessionId"], str):
            return False
        if not isinstance(data_json["nFeat"], int):
            return False
        if not isinstance(data_json["nField"], int):
            return False
        if not isinstance(data_json["cacheSize"], int):
            return False
        if not isinstance(data_json["spec"], dict):
            return False

        if self._is_inference:
            if not isinstance(data_json["spec"]["nBatch"], int):
                return False
        else:
            if not isinstance(data_json["spec"]["nBatchTrain"], int):
                return False
            if not isinstance(data_json["spec"]["nBatchEval"], int):
                return False
            if not isinstance(data_json["spec"]["nBatchTest"], int):
                return False

        return True


class AckTaskResponse(Response):
    def __init__(self, session_id) -> None:
        self._session_id = session_id

    def to_json(self) -> str:
        return json.dumps(
            {"version": 1, "event": "ack_task", "sessionId": self._session_id}
        )
