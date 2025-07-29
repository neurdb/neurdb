import json
import math
import os

from loguru import logger

import deterministic
import env
import util
from ctxpipe.env.metric import *
from ctxpipe.env.primitives import *
from ctxpipe.info import Info

_info = None


def set_info(info: Info):
    global _info
    _info = info


def init():
    # TODO: Extract initializing configuration into object
    global _info
    if not _info:
        raise ValueError("info path not set")

    GlobalConfig.dataset_path = _info.dataset_prefix

    if os.path.exists(_info.aipipe_core_prefix):
        with open(_info.task_info_path) as f:
            GlobalConfig.classification_task_dic = json.load(f)

        """load information files"""
        GlobalConfig.fold_length = math.ceil(
            len(GlobalConfig.classification_task_dic) / Config.k_fold
        )

        with open(_info.task_index_path) as f:
            GlobalConfig.test_index = json.load(f)

        GlobalConfig.train_index = list(
            set(GlobalConfig.classification_task_dic.keys())
            - set(GlobalConfig.test_index)
        )
        GlobalConfig.train_index.sort()


class Config:
    k_fold = 3
    checkpoint: bool = True
    record: bool = False


class GlobalConfig:
    device = env.DEVICE
    enable_context_plugin = True
    enable_ocg_experience_replay = True

    ### hyperparameters for DQN
    gamma = 0.0
    learning_rate = 1e-5  # 1e-5
    frames = 50000
    max_buff = 5000

    column_num = 100

    step_timeout = 60

    eps_decay = 2000

    blank_reward = 0.0
    blank_rewards_str = (
        f"{blank_reward}" if blank_reward >= 0.0 else f"m{-blank_reward}"
    )

    batch_size = 200 if not env.IS_TEST else 10
    logic_batch_size = batch_size // 5

    ctxpipe_setup_name = ""
    if not enable_context_plugin:
        ctxpipe_setup_name = "-noctx"
    elif not enable_ocg_experience_replay:
        ctxpipe_setup_name = "-noocg"
    else:
        ctxpipe_setup_name = "-3linear"

    version = f"{'TEST_' if env.IS_TEST else ''}ctxpipe{ctxpipe_setup_name}"

    exp_dir = util.abspath(env.exp_prefix, f"{version}")
    log_dir = util.abspath("logs", f"{version}")
    # model_dir: str = util.abspath("models", f"{version}")
    model_dir: str = os.path.abspath(
        os.path.join(os.environ["NEURDBPATH"], "external", "ctxpipe", "model")
    )

    result_log_file_name: str = util.abspath(log_dir, "result_log.npy")
    loss_log_file_name: str = util.abspath(log_dir, "loss_log.pkl")
    lp_loss_log_file_name: str = util.abspath(log_dir, "lp_loss_log.npy")
    test_reward_dic_file_name: str = util.abspath(log_dir, "test_reward_dict.npy")

    pipelines_file_name: str = util.abspath(exp_dir, "pipelines.tsv")

    def makedirs(self):
        for d in [self.log_dir, self.exp_dir, self.model_dir]:
            logger.debug("making dir: {}", d)
            os.makedirs(d, exist_ok=True)

    backpropagate_interval: int = 50 if not env.IS_TEST else 10
    checkpoint_interval: int = 200

    column_feature_dim = 19 + 14
    data_dim: int = column_num * column_feature_dim

    classification_task_dic = {}

    dataset_path = ""
    fold_length = -1
    train_index = []
    test_index = []

    single_dataset_mode = False


default_config = GlobalConfig()


class AgentConfig(GlobalConfig):
    epsilon_start = 1.0 if not env.IS_TEST else 0.2
    epsilon_min = 0.2 if not env.IS_TEST else 0.1

    # prim_state_dim: int = GlobalConfig.data_dim + 6 + 1 + 1
    lpip_state_dim: int = GlobalConfig.data_dim + 1


class DQNConfig(GlobalConfig):
    ### RNN param
    seq_embedding_dim: int = 96
    seq_hidden_size: int = 96
    seq_num_layers: int = 1
    predictor_embedding_dim: int = 16
    lpipeline_embedding_dim: int = 8


class EnvConfig(GlobalConfig):
    classification_metric_id: int = 1  # Default: F1; Current: Acc
    regression_metric_id: int = 4


default_dqn_config = DQNConfig()
default_agent_config = AgentConfig()
default_env_config = EnvConfig()
