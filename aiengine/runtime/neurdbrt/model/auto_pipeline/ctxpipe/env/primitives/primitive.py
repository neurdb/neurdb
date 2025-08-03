from typing import Tuple

import pandas as pd


class Primitive:
    def __init__(self, name="blank"):
        self.id = 0
        self.gid = 25
        self.name = name
        self.description = "No-op"
        self.hyperparams = []
        self.type = "blank"

    def fit(self, data):
        pass

    def transform(
        self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_x, test_x

    def transform_x(self, test_x: pd.DataFrame) -> pd.DataFrame:
        return test_x

    def can_accept(self, data):
        return True

    def can_accept_a(self, data):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        num_cols = data._get_numeric_data().columns
        if not len(num_cols) == 0:
            return True
        return False

    def can_accept_b(self, data):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        return True

    def can_accept_c(self, data, task=None, larpack=False):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        with pd.option_context("mode.use_inf_as_na", True):
            if data.isna().any().any():
                return False
        if not len(cat_cols) == 0:
            return False
        return True

    def can_accept_c1(self, data, task=None, larpack=False):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        if not len(cat_cols) == 0:
            return False
        return True

    def can_accept_c2(self, data, task=None, larpack=False):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        if not len(num_cols) == 0:
            return False
        return True

    def can_accept_d(self, data, task):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if not len(cat_cols) == 0:
            return False

        with pd.option_context("mode.use_inf_as_na", True):
            if data.isna().any().any():
                return False
            return True

    def is_needed(self, data):
        return True

    def __repr__(self) -> str:
        return f"<{self.name}>"
