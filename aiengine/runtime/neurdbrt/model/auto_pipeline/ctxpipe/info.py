import json
import os

from loguru import logger

import util

DEFAULT_AIPIPE_PREFIX = "exp/diffprep/aipipe"
DEFAULT_RESULT_PREFIX = "exp/diffprep/result"
DEFAULT_DATASET_PREFIX = "data/dataset"


class Info:
    def __init__(
        self,
        aipipe_core_prefix=DEFAULT_AIPIPE_PREFIX,
        result_prefix=DEFAULT_RESULT_PREFIX,
        dataset_prefix=DEFAULT_DATASET_PREFIX,
    ) -> None:
        self.aipipe_core_prefix = aipipe_core_prefix
        self.result_prefix = result_prefix
        self.dataset_prefix = dataset_prefix

        self._makedirs()
        self._resolve_file_paths()
        self._check_and_create_empty_files()

    def _makedirs(self):
        os.makedirs(self.aipipe_core_prefix, exist_ok=True)
        os.makedirs(self.result_prefix, exist_ok=True)

    def _resolve_file_paths(self):
        ### "datasetinfo.json" saves (datasetname, column names, label index) of each dataset
        self.dataset_info_path = util.abspath(
            self.aipipe_core_prefix, "datasetinfo.json"
        )
        ### "classification_task_dic.json" saves (dataset, data file, label index, model, task) of each task for AIPipe
        self.task_info_path = util.abspath(
            self.aipipe_core_prefix, "classification_task_dic.json"
        )
        ### "test_index.json" saves online tasks for AIPipe
        self.task_index_path = util.abspath(self.aipipe_core_prefix, "test_index.json")

        self._resolve_result_files_paths()

    def _resolve_result_files_paths(self):
        self.failed_file_path = util.abspath(self.result_prefix, "failed.tsv")
        self.done_file_path = util.abspath(self.result_prefix, "done.csv")
        self.stats_db_file_path = util.abspath(self.result_prefix, "stats.sqlite")

    def _check_and_create_empty_files(self):
        for fpath in [self.dataset_info_path, self.task_info_path]:
            if not os.path.exists(fpath):
                logger.warning("creating empty info file at {}", fpath)

                with open(fpath, "w") as f:
                    json.dump({}, f)

        for fpath in [self.task_index_path]:
            if not os.path.exists(fpath):
                logger.warning("creating empty info file at {}", fpath)

                with open(fpath, "w") as f:
                    json.dump([], f)
