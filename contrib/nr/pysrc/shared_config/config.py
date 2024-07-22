import argparse
import configparser
import os
import torch
import random
import numpy as np


def seed_everything(seed=2022):
    """
    [reference]
    https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Function to parse configuration arguments
def parse_config_arguments(config_path: str) -> argparse.Namespace:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    parser = configparser.ConfigParser()
    parser.read(config_path)

    args = argparse.Namespace()

    # Default config
    args.base_dir = parser.get("DEFAULT", "base_dir", fallback=".")
    args.model_repo = parser.get("DEFAULT", "model_repo", fallback="./models")

    # DB config
    args.db_name = parser.get("DB_CONFIG", "db_name", fallback="postgres")
    args.db_user = parser.get("DB_CONFIG", "db_user", fallback="postgres")
    args.db_host = parser.get("DB_CONFIG", "db_host", fallback="localhost")
    args.db_port = parser.get("DB_CONFIG", "db_port", fallback="5432")
    # this is set via environment variable in psql
    args.db_password = parser.get("DB_CONFIG", "db_password", fallback="123")

    # server config
    args.server_port = parser.get("SERVER_CONFIG", "server_port", fallback="8090")

    # server config
    args.data_loader_worker = parser.getint("DATALOADER", "worker", fallback="1")

    # model config
    args.nfield = parser.getint("MODEL_CONFIG", "nfield", fallback=10)
    args.nfeat = parser.getint("MODEL_CONFIG", "nfeat", fallback=1000)
    args.nemb = parser.getint("MODEL_CONFIG", "nemb", fallback=128)
    args.nattn_head = parser.getint("MODEL_CONFIG", "nattn_head", fallback=4)
    args.alpha = parser.getfloat("MODEL_CONFIG", "alpha", fallback=0.2)
    args.h = parser.getint("MODEL_CONFIG", "h", fallback=8)
    args.mlp_nlayer = parser.getint("MODEL_CONFIG", "mlp_nlayer", fallback=2)
    args.mlp_nhid = parser.getint("MODEL_CONFIG", "mlp_nhid", fallback=256)
    args.dropout = parser.getfloat("MODEL_CONFIG", "dropout", fallback=0.5)
    args.ensemble = parser.getboolean("MODEL_CONFIG", "ensemble", fallback=True)
    args.dnn_nlayer = parser.getint("MODEL_CONFIG", "dnn_nlayer", fallback=3)
    args.dnn_nhid = parser.getint("MODEL_CONFIG", "dnn_nhid", fallback=512)

    # training config
    args.epoch = parser.get("TRAIN_MODEL", "epoch", fallback="10")
    args.lr = parser.getfloat("TRAIN_MODEL", "lr", fallback=0.001)
    args.epoch = parser.getint("TRAIN_MODEL", "epoch", fallback=10)
    args.report_freq = parser.getint("TRAIN_MODEL", "report_freq", fallback=10)
    args.patience = parser.getint("TRAIN_MODEL", "patience", fallback=5)
    args.eval_freq = parser.getint("TRAIN_MODEL", "eval_freq", fallback=100)

    return args
