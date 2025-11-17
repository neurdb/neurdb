import gc
import random
import torch
import os
import pickle
import numpy as np
import pandas as pd

from relbench.base import Table
from torch_frame import stype
from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer

from typing import Dict, Union, Any

DEFAULT_COL_DICT_NAME = "col_types.pkl"

ColTypeDictHint = Union[Dict[str, Dict[str, stype]], Dict[str, stype]]


def get_default_text_embedder_cfg():
    pass


def load_col_types(
    cache_path: str,
    file_name: str = DEFAULT_COL_DICT_NAME
) -> ColTypeDictHint:
    """
    Load the column types from the cache.
    :param cache_path: The path to the cache file.
    :return: A dictionary of column types.
    """
    path = os.path.join(cache_path, file_name)
    col_type = pickle.load(open(path, "rb"))
    return col_type


def save_col_types(
        cache_path: str,
        col_type_dict: ColTypeDictHint,
        file_name: str = DEFAULT_COL_DICT_NAME
) -> None:
    """
    Save the column types to the cache.
    :param cache_path: The path to the cache file.
    :param col_type: A dictionary of column types.
    """
    path = os.path.join(cache_path, file_name)
    pickle.dump(col_type_dict, open(path, "wb"))


def save_np_dict(edge_dict: Dict[str, np.array], path: str):
    """_summary_

    Parameters
    ----------
    edge_dict : Dict[str, np.array]
        {"src_table-des_table": np.array} 2-D array
    path : str
        file path to save the edges
    """
    np.savez(path, **edge_dict)


def load_np_dict(path: str) -> Dict[str, np.array]:
    """_summary_

    Parameters
    ----------
    path : str
        file path to load the edges

    Returns
    -------
    Dict[str, np.array]
        {"src_table-des_table": np.array} 2-D array
    """
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def to_unix_time(ser: pd.Series) -> np.ndarray:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp (in seconds)."""
    assert ser.dtype in [np.dtype("datetime64[s]"), np.dtype("datetime64[ns]")]
    unix_time = ser.astype("int64").values
    if ser.dtype == np.dtype("datetime64[ns]"):
        unix_time //= 10**9
    return unix_time


def remove_pkey_fkey(col_to_stype: Dict[str, Any], table: Table) -> dict:
    r"""Remove pkey, fkey columns since they will not be used as input feature."""
    if table.pkey_col is not None:
        if table.pkey_col in col_to_stype:
            col_to_stype.pop(table.pkey_col)
    for fkey in table.fkey_col_to_pkey_table.keys():
        if fkey in col_to_stype:
            col_to_stype.pop(fkey)


def setup_torch(seed: int = 1998):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
