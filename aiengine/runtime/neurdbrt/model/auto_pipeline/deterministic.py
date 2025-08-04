import os
import random

import numpy as np
import torch

RANDOM_SEED = 1145


def seed_everything(seed=RANDOM_SEED):
    """ "
    Seed everything.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type:ignore
    torch.backends.cudnn.benchmark = False  # type:ignore
    # torch.use_deterministic_algorithms(True)


seed_everything()

task_rng = np.random.default_rng(RANDOM_SEED)
predictor_rng = np.random.default_rng(RANDOM_SEED)
buffer_rng = np.random.default_rng(RANDOM_SEED)
epsilon_greedy_rng = np.random.default_rng(RANDOM_SEED)
random_action_rng = np.random.default_rng(RANDOM_SEED)
