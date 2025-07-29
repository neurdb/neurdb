import os
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

import comp
import deterministic
from config import EnvConfig

from .pipeline import Pipeline
from .primitives.primitive import Primitive


class Environment:

    def __init__(self, config: EnvConfig, train=True):
        self._config = config
        self.column_num = config.column_num
        self.state = None
        self.reward = None
        self.action = None
        self.next_state = None
        self.done = False
        self.train = train
        self.lpip_state = None
        self._pipeline = None

    @property
    def pipeline(self) -> Pipeline:
        if not self._pipeline:
            raise ValueError("self.pipeline not initialized")

        return self._pipeline

    def reset(self, taskid=None, predictor=None, metric=None, default=True):
        print(
            f"taskid: {taskid}, predictor: {predictor}, metric: {metric}, default: {default}"
        )

        old_taskid = taskid
        old_predictor = predictor
        old_metric = metric

        is_pipeline_determined = False
        while not is_pipeline_determined:
            if default == True or taskid is None or predictor is None or metric is None:
                taskid = deterministic.task_rng.choice(self._config.train_index)
                logger.debug(f"taskid: {taskid}")

                predictor_id = deterministic.predictor_rng.integers(
                    1, len(comp.predictors) + 1
                )
                try:
                    predictor = [i for i in comp.predictors if i.id == predictor_id][0]
                except IndexError:
                    logger.warning(
                        f"predictor_id {predictor_id} not exist in self._config.predictors. retrying"
                    )
                    taskid = old_taskid
                    continue

                logger.debug(f"predictor_id: {predictor_id}")

                metric = [
                    i
                    for i in comp.metrics
                    if i.id == self._config.classification_metric_id
                ][0]

            try:
                self.reset_pipeline(taskid, predictor, metric, train=self.train)
            except FileNotFoundError:
                dataset_path = os.path.join(
                    self._config.dataset_path,
                    self._config.classification_task_dic[taskid]["dataset"],
                    self._config.classification_task_dic[taskid]["csv_file"],
                )
                logger.warning(f"dataset {dataset_path} not found. sample again")
                taskid = old_taskid
                predictor = old_predictor
                metric = old_metric
                continue

            is_pipeline_determined = True

            self.column_num = self._config.column_num
            self.lpip_state = self.get_lpip_state()
            self.reward = None
            self.action = None
            self.next_state = None
            self.done = False

    def reset_pipeline(self, taskid, predictor, metric, train=True):
        if self._pipeline:
            self._pipeline.reset_data()
            del self._pipeline
            self._pipeline = None

        self._pipeline = Pipeline(taskid, predictor, metric, self._config, train=train)

    def step(
        self, step: Primitive, has_timeout=True
    ) -> Tuple[Optional[list], Optional[int]]:
        logger.debug(f"adding step {step.name}...")

        if self._pipeline is None:
            self.reset()

        self.prim_state = self.get_state()
        step_result = self.pipeline.add_step(
            step, has_timeout=has_timeout
        )  # type:ignore
        if step_result <= 0:
            return None, step_result

        self.next_prim_state = self.get_state()
        self.get_reward(has_timeout=has_timeout)
        self.set_done()
        self.action = step

        # logger.debug(f"added step {step.name}. reward: {self.reward} done: {self.done}")

        return [self.prim_state, self.reward, self.next_prim_state, self.done], None

    def get_data_feature(self) -> np.ndarray:
        def _test_value(value) -> float:
            if np.isnan(value) or abs(value) == np.inf:
                return 0.0
            else:
                return value

        def _test_frexp(value) -> Tuple[float, int]:
            if np.isnan(value) or abs(value) == np.inf:
                return 0.0, 0
            else:
                return np.frexp(value)

        inp_data = pd.DataFrame(self.pipeline.train_x)

        # if len(self.pipeline.train_y.shape) > 1:
        #     train_y = self.pipeline.train_y[0]
        # else:
        #     train_y = self.pipeline.train_y
        # categorical = list(self.pipeline.train_x.dtypes == object)

        column_info = {}

        for i in range(len(inp_data.columns)):
            col = inp_data.iloc[:, i]
            if i >= self.column_num:
                break
            s_s = col

            column_info[i] = {}
            column_info[i]["col_name"] = "unknown_" + str(i)
            column_info[i]["dtype"] = str(s_s.dtypes)  # 1
            column_info[i]["length"], column_info[i]["length_exp"] = _test_frexp(
                len(s_s.values)
            )  # 2
            column_info[i]["null_ratio"] = s_s.isnull().sum() / len(s_s.values)  # 3
            column_info[i]["ctype"] = (
                1 if inp_data.columns[i] in self.pipeline.num_cols else 2
            )
            column_info[i]["nunique"], column_info[i]["nunique_exp"] = _test_frexp(
                s_s.nunique()
            )  # 5
            column_info[i]["nunique_ratio"] = s_s.nunique() / len(s_s.values)  # 6

            d = s_s.describe()

            if "mean" not in d:
                column_info[i]["ctype"] = 2

            if column_info[i]["ctype"] == 1:  # numeric
                column_info[i]["mean"], column_info[i]["mean_exp"] = _test_frexp(
                    d["mean"]
                )  # 7
                column_info[i]["std"], column_info[i]["std_exp"] = _test_frexp(
                    d["std"]
                )  # 8
                column_info[i]["min"], column_info[i]["min_exp"] = _test_frexp(
                    d["min"]
                )  # 9
                column_info[i]["25%"], column_info[i]["25%_exp"] = _test_frexp(d["25%"])
                column_info[i]["50%"], column_info[i]["50%_exp"] = _test_frexp(d["50%"])
                column_info[i]["75%"], column_info[i]["75%_exp"] = _test_frexp(d["75%"])
                column_info[i]["max"], column_info[i]["max_exp"] = _test_frexp(d["max"])
                column_info[i]["median"], column_info[i]["median_exp"] = _test_frexp(
                    s_s.median()
                )

                if len(s_s.mode()) != 0:
                    column_info[i]["mode"], column_info[i]["mode_exp"] = _test_frexp(
                        s_s.mode().iloc[0]
                    )
                else:
                    column_info[i]["mode"], column_info[i]["mode_exp"] = 0.0, 0

                mr = s_s.astype("category").describe().iloc[3] / len(s_s.values)
                column_info[i]["mode_ratio"] = _test_value(mr)

                column_info[i]["sum"], column_info[i]["sum_exp"] = _test_frexp(
                    s_s.sum()
                )
                column_info[i]["skew"], column_info[i]["skew_exp"] = _test_frexp(
                    s_s.skew()
                )
                column_info[i]["kurt"], column_info[i]["kurt_exp"] = _test_frexp(
                    s_s.kurt()
                )

                # print(f"column_info[{i}]: {column_info[i]}")

            elif column_info[i]["ctype"] == 2:  # category
                column_info[i]["mean"], column_info[i]["mean_exp"] = 0.0, 0
                column_info[i]["std"], column_info[i]["std_exp"] = 0.0, 0
                column_info[i]["min"], column_info[i]["min_exp"] = 0.0, 0
                column_info[i]["25%"], column_info[i]["25%_exp"] = 0.0, 0
                column_info[i]["50%"], column_info[i]["50%_exp"] = 0.0, 0
                column_info[i]["75%"], column_info[i]["75%_exp"] = 0.0, 0
                column_info[i]["max"], column_info[i]["max_exp"] = 0.0, 0
                column_info[i]["median"], column_info[i]["median_exp"] = 0.0, 0

                column_info[i]["mode"], column_info[i]["mode_exp"] = 0.0, 0
                column_info[i]["mode_ratio"] = 0.0
                column_info[i]["sum"], column_info[i]["sum_exp"] = 0.0, 0
                column_info[i]["skew"], column_info[i]["skew_exp"] = 0.0, 0
                column_info[i]["kurt"], column_info[i]["kurt_exp"] = 0.0, 0

        data_feature = []
        for index in column_info.keys():
            one_column_feature = []
            column_dic = column_info[index]
            for kw in column_dic.keys():
                if kw == "col_name" or kw == "content":
                    continue
                elif kw == "dtype":
                    content = comp.dtype_id_map[column_dic[kw]]
                else:
                    content = column_dic[kw]
                one_column_feature.append(content)
            data_feature.append(one_column_feature)

        if len(column_info) < self.column_num:
            for index in range(len(column_info), self.column_num):
                one_column_feature = np.zeros(self._config.column_feature_dim)
                data_feature.append(one_column_feature)

        data_feature = np.ravel(np.array(data_feature))

        del inp_data
        del column_info

        return data_feature

    def get_lpip_state(self) -> np.ndarray:
        data_feature = self.get_data_feature()
        predictor = np.array([self.pipeline.predictor.id])
        state = np.concatenate((data_feature, predictor))
        return state

    def get_state(self) -> np.ndarray:
        data_feature = self.get_data_feature()
        sequence = np.array(self.pipeline.gsequence)
        predictor = np.array([self.pipeline.predictor.id - 1])
        logic_pipeline_id = np.array([self.pipeline.logic_pipeline_id])
        state = np.concatenate((data_feature, sequence, predictor, logic_pipeline_id))
        return state

    def get_reward(self, has_timeout):
        if len(self.pipeline.sequence) < 6:
            self.reward = (
                0.0 if self.pipeline.sequence[-1].id == 0 else self._config.blank_reward
            )
        else:
            self.reward = self.pipeline.evaluate(has_timeout=has_timeout)
            self.end_time = time.time()

    def set_done(self):
        if len(self.pipeline.sequence) < 6:
            self.done = False
        elif len(self.pipeline.sequence) == 6:
            self.done = True
        else:
            return

    def has_nan(self):
        has_num_nan = False
        has_cat_nan = False

        def catch_num(data):
            num_cols = [
                col for col in data.columns if str(data[col].dtypes) != "object"
            ]
            cat_cols = [col for col in data.columns if col not in num_cols]
            cat_train_x = data[cat_cols]
            num_train_x = data[num_cols]
            return cat_train_x, num_train_x

        with pd.option_context("mode.use_inf_as_na", True):
            cat_train_x, num_train_x = catch_num(self.pipeline.train_x)
            cat_test_x, num_test_x = catch_num(self.pipeline.test_x)
            if len(self.pipeline.cat_cols) != 0:
                if cat_train_x.isna().any().any():
                    has_cat_nan = True
                if cat_test_x.isna().any().any():
                    has_cat_nan = True
            if len(self.pipeline.num_cols) != 0:
                if num_train_x.isna().any().any():
                    has_num_nan = True
                if num_test_x.isna().any().any():
                    has_num_nan = True

        return has_num_nan, has_cat_nan

    def has_cat_cols(self):
        if not len(self.pipeline.cat_cols) == 0:
            return True
        else:
            return False
