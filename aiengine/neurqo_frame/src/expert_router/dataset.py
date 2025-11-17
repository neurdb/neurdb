# Standard library imports
import copy
from typing import Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch

# Local/project imports
from common import BaseConfig
from torch.utils.data import DataLoader, Dataset, random_split


def convert_to_features(
    df: pd.DataFrame, fq_instance, threshold: float, cfg: BaseConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
    if threshold is None:
        threshold = 0
    X = []
    query_idents = df["query_ident"].tolist()
    for idx, sql in enumerate(df["query_sql"]):
        try:
            sql_feature = fq_instance.encode_query(query_idents[idx], sql)
        except Exception as e:
            print(
                f"Error {e} encoding query at index {idx}, query_ident: {query_idents[idx]}, sql: {sql}"
            )
            raise e
        X.append(sql_feature)

    # Parse execution times
    execution_times = df["execution_time_ms"].tolist()
    y_execution_times = np.zeros((len(execution_times), len(cfg.ALL_METHODS)))
    for query_id_, exec_time_dict in enumerate(execution_times):
        for method, time in exec_time_dict.items():
            if method in cfg.FIXED_LABEL_MAPPING:
                y_execution_times[query_id_, cfg.FIXED_LABEL_MAPPING[method]] = time

    # Generate multi-label targets
    y_multi = np.zeros((len(df), len(cfg.ALL_METHODS)))
    for i, exec_time_dict in enumerate(execution_times):
        times = np.array([exec_time_dict.get(method) for method in cfg.ALL_METHODS])
        best_time = np.min(times)  # Best total_time for this query
        acceptable_time = best_time * (
            1 + threshold
        )  # Threshold for acceptable performance
        for j, time in enumerate(times):
            if time <= acceptable_time:
                y_multi[i, j] = 1

    # Compute class weights based on frequency of being acceptable
    class_counts = np.sum(
        y_multi, axis=0
    )  # Number of times each optimizer is acceptable
    class_weights = np.zeros(len(cfg.ALL_METHODS))
    for i, count in enumerate(class_counts):
        if count > 0:
            class_weights[i] = len(df) / (
                len(cfg.ALL_METHODS) * count
            )  # Inverse frequency
        else:
            class_weights[i] = (
                1.0  # Default weight if never acceptable (avoid division by zero)
            )

    return X, y_multi, y_execution_times, query_idents, class_weights


def dynamic_padding_collate(batch):
    X_batch = [item[0] for item in batch]  # List of feature dicts
    y_multi_batch = torch.stack([item[1] for item in batch])  # Multi-label targets
    y_execution_times_batch = torch.stack(
        [item[2] for item in batch]
    )  # Execution times
    query_idents_batch = [item[3] for item in batch]  # Query identifiers

    # Find maximum lengths for joins and filters in the batch
    max_joins = max(len(x["join_conditions"]) for x in X_batch)
    max_filters = max(len(x["filter_conditions"]) for x in X_batch)

    # Pad join_conditions and filter_conditions
    padded_join_conditions = []
    padded_filter_conditions = []

    for x in X_batch:
        joins = torch.from_numpy(
            x["join_conditions"]
        )  # Shape: [num_joins, feature_dim]
        filters = torch.from_numpy(
            x["filter_conditions"]
        )  # Shape: [num_filters, feature_dim]

        # Pad joins to max_joins
        pad_joins = max_joins - len(joins)
        if pad_joins > 0:
            pad_tensor = torch.full((pad_joins, joins.shape[1]), 0.0)  # Pad with 0
            joins_padded = torch.cat([joins, pad_tensor], dim=0)
        else:
            joins_padded = joins

        # Pad filters to max_filters
        pad_filters = max_filters - len(filters)
        if pad_filters > 0:
            pad_tensor = torch.full((pad_filters, filters.shape[1]), 0.0)  # Pad with 0
            filters_padded = torch.cat([filters, pad_tensor], dim=0)
        else:
            filters_padded = filters

        padded_join_conditions.append(joins_padded)
        padded_filter_conditions.append(filters_padded)

    # Stack padded tensors into batch tensors
    join_conditions_batch = torch.stack(
        padded_join_conditions
    )  # Shape: [batch_size, max_joins, 4]
    filter_conditions_batch = torch.stack(
        padded_filter_conditions
    )  # Shape: [batch_size, max_filters, 3]

    # Reconstruct X_batch with padded tensors
    X_padded = {
        "join_conditions": join_conditions_batch,
        "filter_conditions": filter_conditions_batch,
        "table_sizes": torch.stack(
            [torch.from_numpy(x["table_sizes"]) for x in X_batch]
        ),
    }

    return X_padded, y_multi_batch, y_execution_times_batch, query_idents_batch


class QueryFeatureDataset(Dataset):
    def __init__(
        self,
        X,
        y_multi,
        y_execution_times,
        query_idents,
        class_weights=None,
        workload_name=" ",
    ):
        X = copy.deepcopy(X)
        # Apply global shift once
        for x in X:
            x["join_conditions"][:, 0] += 1  # table1_id
            x["join_conditions"][:, 1] += 1  # col1_id
            x["join_conditions"][:, 2] += 1  # table2_id
            x["join_conditions"][:, 3] += 1  # col2_id

            x["filter_conditions"][:, 0] += 1  # table_id
            x["filter_conditions"][:, 1] += 1  # col_id

        self.X = X
        self.y_multi = torch.tensor(y_multi, dtype=torch.float32)  # Multi-label targets
        self.y_execution_times = torch.tensor(y_execution_times, dtype=torch.float32)
        self.query_idents = query_idents  # Store string labels directly

        self._workload_name = workload_name  # Store string labels directly
        self._class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

    def get_current_worklaod(self):
        return self._workload_name

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y_multi[index],
            self.y_execution_times[index],
            self.query_idents[index],
        )

    def get_class_weights(self):
        return self._class_weights


def get_data_loader_split(
    cfg: BaseConfig,
    workload_name: str,
    datasets: Dict[str, pd.DataFrame],
    batch_size: int,
    fq_instance,
    threshold: float,
    val_split: float = 0.2,
) -> Tuple[Dict[str, DataLoader], int, int]:
    data_loaders = {}
    input_dim = 0
    output_dim = len(cfg.ALL_METHODS)

    for name, df in datasets.items():
        X, y_multi, y_execution_times, query_idents, class_weights = (
            convert_to_features(df, fq_instance, threshold, cfg=cfg)
        )
        dataset = QueryFeatureDataset(
            X,
            y_multi,
            y_execution_times,
            query_idents,
            class_weights if name == "train" else None,
            workload_name=workload_name,
        )

        if name == "train":
            total_size = len(dataset)
            val_size = int(val_split * total_size)
            train_size = total_size - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # Use custom collate function
            data_loaders["train"] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=dynamic_padding_collate,
            )
            data_loaders["val"] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dynamic_padding_collate,
            )
        else:
            data_loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dynamic_padding_collate,
            )

    return data_loaders, input_dim, output_dim


def get_data_loader(
    cfg: BaseConfig,
    datasets: Dict[str, pd.DataFrame],
    batch_size: int,
    fq_instance,
    threshold: float,
    workload_name: str = " ",
) -> Tuple[Dict[str, DataLoader], int, int]:
    """
    Args:
        workload_name:
        datasets: Dictionary of dataframes (e.g., {'train': df, 'test': df})
        batch_size: Batch size for DataLoader
        fq_instance: SQL feature encoder
        threshold: Performance threshold for multi-labeling

    Returns:
        data_loaders: Dictionary of DataLoaders
        input_dim: Input feature dimension
        output_dim: Number of optimizers (classes)
    """
    data_loaders = {}
    input_dim = 0
    output_dim = len(cfg.ALL_METHODS)

    for name, df in datasets.items():
        X, y_multi, y_execution_times, query_idents, class_weights = (
            convert_to_features(df, fq_instance, threshold, cfg=cfg)
        )

        dataset = QueryFeatureDataset(
            X,
            y_multi,
            y_execution_times,
            query_idents,
            class_weights if name == "train" else None,
            workload_name=workload_name,
        )
        shuffle = name == "train"
        data_loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dynamic_padding_collate,
        )

    return data_loaders, input_dim, output_dim


def get_data_loader_online(
    cfg: BaseConfig,
    datasets: Dict[str, pd.DataFrame],
    batch_size: int,
    fq_instance,
    threshold: float,
    workload_name: str = " ",
) -> Tuple[Dict[str, DataLoader], int, int]:
    """
    Args:
        workload_name:
        datasets: Dictionary of dataframes (e.g., {'train': df, 'test': df})
        batch_size: Batch size for DataLoader
        fq_instance: SQL feature encoder
        threshold: Performance threshold for multi-labeling

    Returns:
        data_loaders: Dictionary of DataLoaders
        input_dim: Input feature dimension
        output_dim: Number of optimizers (classes)
    """
    data_loaders = {}
    input_dim = 0
    output_dim = len(cfg.ALL_METHODS)

    for name, df in datasets.items():
        X, y_multi, y_execution_times, query_idents, class_weights = (
            convert_to_features(df, fq_instance, threshold, cfg=cfg)
        )

        dataset = QueryFeatureDataset(
            X,
            y_multi,
            y_execution_times,
            query_idents,
            class_weights if name == "train" else None,
            workload_name=workload_name,
        )
        data_loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dynamic_padding_collate,
        )

    return data_loaders, input_dim, output_dim
