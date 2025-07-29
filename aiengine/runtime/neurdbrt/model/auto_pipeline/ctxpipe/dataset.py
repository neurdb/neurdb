from typing import Dict, List, TypedDict

import pandas as pd


class TaskInfo(TypedDict):
    dataset: str
    csv_file: str
    label: str
    model: str
    task_name: str


class DatasetInfo(TypedDict):
    dataset: str
    column_index: Dict[str, int]
    label: str
    index: List[int]


class Dataset:
    def __init__(
        self, name: str, path: str, label_column_id: int, label_column_name: str
    ) -> None:
        self._name = name
        self._path = path
        self._label_column_id = label_column_id
        self._label_column_name = label_column_name

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    @property
    def label_column_id(self):
        return self._label_column_id

    @property
    def label_column_name(self):
        return self._label_column_name

    def make_task_info(self, predictor_name: str, task_name: str) -> TaskInfo:
        return {
            "dataset": self.path.split("/")[-2],
            "csv_file": self.path.split("/")[-1],
            "label": str(self.label_column_id),
            "model": predictor_name,
            "task_name": task_name,
        }

    @property
    def info(self) -> DatasetInfo:
        data = pd.read_csv(self.path)

        columns = data.columns
        column_index = {}
        for index, col in enumerate(columns):
            column_index[col] = index

        return {
            "dataset": self.path.split("/")[-2],
            "column_index": column_index,
            "label": str(self.label_column_name),
            "index": [self.label_column_id],
        }
