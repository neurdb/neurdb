import env

env.init()

import sys

import config as conf
from ctxpipe.agentman import AgentManager
from ctxpipe.info import Info

# from ctxpipe.stats import init_stats_db


class Setup:
    def __init__(self, aipipe_core_prefix, result_prefix, dataset_prefix) -> None:
        self._info = Info(aipipe_core_prefix, result_prefix, dataset_prefix)
        # init_stats_db(self._info.stats_db_file_path)

    def _init_info_path(self):
        conf.set_info(self._info)
        conf.init()

    def train(self, resume_from=0):
        self._init_info_path()

        am = AgentManager()
        am.train(resume_from=resume_from)


def train_on_haipipe_dataset():
    setup = Setup(
        aipipe_core_prefix="data/meta",
        result_prefix="data/train_result",
        dataset_prefix="data/dataset",
    )

    resume_from = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    print(f"RESUME FROM {resume_from}")

    setup.train(resume_from=resume_from)


if __name__ == "__main__":
    # evaluate_on_diffprep_dataset()
    train_on_haipipe_dataset()
