import multiprocessing
import os
import signal
import time
from multiprocessing import Process
from typing import List

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import comp
import util
from config import GlobalConfig

from .primitives.predictor import *


class FunctionTimedOut(Exception):
    pass


def _do_add_step(
    train_x, test_x, train_y, step: Primitive, queue: multiprocessing.Queue
):
    os.setsid()
    print(f"executing function _do_add_step")

    try:
        train_x, test_x = step.transform(train_x, test_x, train_y)
        num_cols = list(train_x._get_numeric_data().columns)
        cat_cols = list(set(train_x.columns) - set(num_cols))
    except:
        queue.put(FunctionTimedOut())
        queue.close()
        return

    queue.put([train_x, test_x, num_cols, cat_cols])
    queue.close()
    return train_x, test_x, num_cols, cat_cols


def _do_evaluate(
    train_x, test_x, train_y, test_y, predictor, metric, queue: multiprocessing.Queue
):
    os.setsid()
    print(f"executing function _do_evaluate")

    try:
        pred_y = predictor.transform(train_x, train_y, test_x)
        result = metric.evaluate(pred_y, test_y)
    except:
        queue.put(FunctionTimedOut())
        queue.close()
        return

    queue.put([pred_y, result])
    queue.close()
    return pred_y, result


class Pipeline:

    def __init__(
        self, taskid, predictor: Primitive, metric, config: GlobalConfig, train=True
    ):
        self._config = config
        self.taskid = taskid
        self.metric = metric
        self.predictor = predictor
        self.train = train

        self.code = ""
        self.result = 0
        self.sequence: List[Primitive] = []
        self.index = 0

        self.data_x: pd.DataFrame = None
        self.data_y: pd.DataFrame = None
        self.train_x: pd.DataFrame = None
        self.test_x: pd.DataFrame = None
        self.train_y: pd.DataFrame = None
        self.test_y: pd.DataFrame = None
        self.pred_y: pd.DataFrame = None

        self.num_cols: list = []
        self.cat_cols: list = []

        self.load_data(taskid)
        self._logic_pipeline_id = None
        self.gsequence = [26, 26, 26, 26, 26, 26]

    @property
    def logic_pipeline_id(self) -> int:
        if self._logic_pipeline_id is None:
            raise ValueError("self.logic_pipeline_id not initialized")

        return self._logic_pipeline_id

    @logic_pipeline_id.setter
    def logic_pipeline_id(self, value) -> None:
        self._logic_pipeline_id = value

    def load_data(self, taskid, ratio=0.8, split_random_state=0):
        if self._config.single_dataset_mode:
            data = pd.read_csv(
                os.path.join(
                    self._config.dataset_path,
                    self._config.classification_task_dic[taskid]["csv_file"],
                )
            ).infer_objects()
        else:
            data = pd.read_csv(
                os.path.join(
                    self._config.dataset_path,
                    self._config.classification_task_dic[taskid]["dataset"],
                    self._config.classification_task_dic[taskid]["csv_file"],
                )
            ).infer_objects()

        label_index = int(self._config.classification_task_dic[taskid]["label"])

        data = data.replace([np.inf, -np.inf], np.nan)
        data.dropna(subset=[data.iloc[:, label_index].name])
        if data.shape[0] > 1500 and self.train:
            data = data.iloc[:1500, :]

        column = str(data.columns[label_index])
        # logger.debug(f"column={column}")
        self.data_x = data.drop(columns=[column], axis=1)
        self.data_y = data.iloc[:, label_index].values
        del data

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x,
            self.data_y,
            train_size=ratio,
            test_size=1 - ratio,
            random_state=split_random_state,
        )

        if str(self.data_y.dtype) == "Object":
            le = LabelEncoder()
            self.data_y = le.fit_transform(self.data_y)

        self.num_cols = list(self.train_x._get_numeric_data().columns)
        self.cat_cols = list(set(self.train_x) - set(self.num_cols))

    def reset_data(self):
        self.data_x = None
        self.data_y = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.pred_y = None

        del self.data_x
        del self.data_y
        del self.train_x
        del self.test_x
        del self.train_y
        del self.test_y
        del self.pred_y
        del self._config
        del self.taskid
        del self.metric
        del self.predictor
        del self.train
        del self.num_cols
        del self.cat_cols

        del self.code
        del self.result
        del self.sequence
        del self.index

    def get_index(self):
        return self.index

    def _subprocess(self, func, args, has_timeout):
        q = multiprocessing.Queue()
        args.append(q)
        process = Process(target=func, args=args)

        # logger.debug(f"process {func.__name__} created")

        func_return = None

        process.start()

        if has_timeout:
            timed_out = False
            finished = False
            passed_time = 0.0
            while not timed_out and not finished:
                if not process.is_alive():
                    finished = True
                    break

                if not q.empty():
                    finished = True
                    break

                time.sleep(0.1)
                passed_time += 0.1
                if passed_time % 10 < 0.001:
                    logger.debug(f"Passed {passed_time} secs")

                if passed_time > self._config.step_timeout:
                    timed_out = True

            if timed_out:
                logger.warning(f"Timed out: {func.__name__}")
                func_return = None
            else:
                try:
                    func_return = q.get_nowait()
                    if isinstance(func_return, BaseException):
                        raise func_return
                except:
                    logger.warning(f"Error: {func.__name__}")
                    func_return = None
        else:
            func_return = q.get()

        try:
            if process.pid:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                logger.warning(f"Use process.terminate()")
                process.terminate()
        except:
            pass

        os.system(f"pkill -f '.*joblib'")

        q.close()

        util.clean_mem()

        return func_return

    def add_step(self, step: Primitive, has_timeout=True):  # step is a Primitive
        if self.index >= len(comp.lpipelines[self.logic_pipeline_id]):
            return -1

        pre_pipeline = []
        if self.index > 0:
            for ind in range(self.index):
                pre_pipeline.append(ind)
        if (
            step.type in pre_pipeline
            or not step.can_accept(self.train_x)
            or not step.can_accept(self.test_x)
            or (not step.is_needed(self.train_x) and not step.is_needed(self.test_x))
        ):
            return 0

        try:
            func_return = self._subprocess(
                _do_add_step,
                [self.train_x, self.test_x, self.train_y, step],
                has_timeout=has_timeout,
            )
            if func_return is None:
                logger.error(f"adding step {step} timed out")
                return -1

            [self.train_x, self.test_x, self.num_cols, self.cat_cols] = func_return
        except FunctionTimedOut:
            logger.error(f"adding step {step} timed out")
            return -1

        self.sequence.append(step)
        self.gsequence[self.index] = step.gid

        self.index += 1
        return 1

    def evaluate(self, has_timeout=True):
        if len(self.sequence) < 6:
            return

        logger.info(
            f"evaluating {self.sequence} using predictor {self.predictor.name}..."
        )

        try:
            func_return = self._subprocess(
                _do_evaluate,
                [
                    self.train_x,
                    self.test_x,
                    self.train_y,
                    self.test_y,
                    self.predictor,
                    self.metric,
                ],
                has_timeout=has_timeout,
            )
            if func_return is None:
                logger.error(
                    f"evaluating {self.sequence} using predictor {self.predictor.name} error"
                )
                self.result = -1
            else:
                [self.pred_y, self.result] = func_return
        except FunctionTimedOut:
            logger.error(
                f"evaluating {self.sequence} using predictor {self.predictor.name} timed out"
            )
            self.result = -1
        except Exception as e:
            logger.error(
                f"evaluating {self.sequence} using predictor {self.predictor.name} failed with error {str(e)}"
            )
            self.result = -1

        return self.result
