# Standard library imports
import glob
import json
import logging
import os
import random
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

# Third-party imports
import pandas as pd

# Local/project imports
from common import BaseConfig, get_config

warnings.filterwarnings("ignore", category=FutureWarning)


def assign_base_experiment(experiment: str) -> str:
    """Assign base_experiment based on experiment name."""
    if 'leave_one_out' in experiment:
        return 'leave_one_out'
    elif 'base_query_split' in experiment:
        return 'base_query'
    elif 'random_split' in experiment:
        return 'random'
    return None


def load_and_process_data(cfg: BaseConfig, with_leon=False) -> pd.DataFrame:
    # Load raw data
    filepath = cfg.CSV
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath)

    df.loc[df[df['timed_out']].index, 'execution_time'] = cfg.EXECUTION_TIME_OUT
    df.loc[df[df['planning_time'] >= cfg.EXECUTION_TIME_OUT].index, 'planning_time'] = 0.0
    df['prepare_time'] = df['inference_time'] + df['planning_time']

    print(f"Loaded raw data from {filepath}")

    aggregated_dfs = []
    split_df = \
        df.groupby(['experiment', 'query_ident', 'split']).count().reset_index().sort_values(
            ['experiment', 'query_ident'])[
            ['experiment', 'query_ident', 'split']]

    for (method, run_type, experiment), df_group in df.groupby(['method', 'run_type', 'experiment'], dropna=False):
        if run_type != 'regular':
            continue

        df_group.loc[df_group.index, 'timed_out'] = df_group.loc[df_group.index, 'timed_out'].astype(int)

        x = df_group.groupby(['method', 'run_type', 'experiment', 'query_ident', 'split'], dropna=False)
        aggregated = x.agg(
            {'inference_time': ['mean', 'std'], 'planning_time': ['mean', 'std'], 'execution_time': ['mean', 'std'],
             'total_time': ['mean', 'std'], 'prepare_time': ['mean', 'std'], 'timed_out': 'max'}).reset_index()

        # Fix nested names from double aggregations (mean+std)
        new_col_names = []
        for c in aggregated.columns:
            if c[1] == 'std':
                new_col_names.append(".".join(c))
            else:
                new_col_names.append(c[0])
        aggregated.columns = new_col_names

        aggregated = aggregated.sort_values(['query_ident'])

        aggregated.loc[aggregated['timed_out'] > 0, 'timed_out'] = 1.0
        aggregated.loc[aggregated['timed_out'] > 0, 'execution_time'] = cfg.EXECUTION_TIME_OUT
        # Add correction to total_time
        aggregated['total_time'] = aggregated['inference_time'] + aggregated['planning_time'] + aggregated[
            'execution_time']

        # Add number of runs per query
        aggregated['n'] = x.count()['run_id'].tolist()

        if method != 'PostgreSQL':
            if 'leave_one_out' in experiment:
                aggregated['base_experiment'] = 'leave_one_out'
            elif 'base_query_split' in experiment:
                aggregated['base_experiment'] = 'base_query'
            elif 'random_split' in experiment:
                aggregated['base_experiment'] = 'random'

            aggregated_dfs.append(aggregated.copy())

        else:
            for new_experiment in df['experiment'].dropna().unique():
                tmp = aggregated.copy()
                tmp = tmp.sort_values(['query_ident'])

                tmp['experiment'] = new_experiment
                tmp['split'] = split_df[split_df['experiment'] == new_experiment]['split'].tolist()

                if 'leave_one_out' in new_experiment:
                    tmp['base_experiment'] = 'leave_one_out'
                elif 'base_query_split' in new_experiment:
                    tmp['base_experiment'] = 'base_query'
                elif 'random_split' in new_experiment:
                    tmp['base_experiment'] = 'random'

                aggregated_dfs.append(tmp)

    return pd.concat(aggregated_dfs)


def split_train_test_queries(test_queries_dict: Dict[str, List[str]], cfg: BaseConfig, test_size: int = 20,
                             seed: int = None) -> Tuple[List[str], List[str]]:
    """
    Split queries into train and test sets with a random strategy.
    """
    all_queries = cfg.get_all_queries()
    total_queries = len(all_queries)

    test_queries_candidates = set()
    for queries in test_queries_dict.values():
        test_queries_candidates.update(queries)

    all_queries_list = list(set(all_queries))
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(42)

    test_queries = set(random.sample(test_queries_candidates, min(test_size, len(test_queries_candidates))))
    train_queries = [q for q in all_queries_list if q not in test_queries]

    test_queries = sorted(list(test_queries))
    train_queries = sorted(train_queries)

    overlap = set(train_queries) & set(test_queries)
    if overlap:
        logging.warning(f"Overlap between train and test queries: {overlap}")
    if not test_queries:
        logging.warning("No test queries identified")
    if not train_queries:
        logging.warning("No train queries identified")

    print(f"Total queries: {total_queries}, Train: {len(train_queries)}, Test: {len(test_queries)}")
    return train_queries, test_queries


def compute_avg_performance(df: pd.DataFrame, queries: List[str], split_type: str = 'train') -> pd.DataFrame:
    """Compute average total time per query-method pair."""
    df_split = df[df['split'] == split_type].copy()
    df_queries = df_split[df_split['query_ident'].isin(queries)].copy()
    if df_queries.empty:
        logging.warning(f"No data found for {split_type} split with queries: {queries[:5]}...")
    return df_queries.groupby(['query_ident', 'method'])['total_time'].mean().reset_index()


def assign_labels(avg_performance: pd.DataFrame) -> pd.DataFrame:
    """Assign labels: top-1 method and top-2 methods based on smallest average total time."""
    labeled_data = []
    for query in avg_performance['query_ident'].unique():
        query_data = avg_performance[avg_performance['query_ident'] == query]
        sorted_data = query_data.sort_values('total_time')
        top_methods = sorted_data['method'].values

        execution_time_dict = dict(zip(query_data['method'], query_data['total_time']))
        labels = {
            'query_ident': query,
            'top_1_method': top_methods[0],
            'top_2_methods': list(top_methods[:2]) if len(top_methods) >= 2 else list(top_methods),
            'execution_time_ms': json.dumps(execution_time_dict)
        }

        labeled_data.append(labels)
    return pd.DataFrame(labeled_data)


def prepare_workload_test_datasets(
        save_dir: str,
        df: pd.DataFrame, test_queries_dict: Dict[str, List[str]],
        cfg: BaseConfig) -> \
        Dict[str, pd.DataFrame]:
    workload_datasets = {}
    os.makedirs(save_dir, exist_ok=True)

    for workload, query_idents in test_queries_dict.items():
        # Filter df for the specific experiment/workload
        df_workload = df[df['experiment'] == workload].copy()
        df_workload = df_workload[df_workload['query_ident'].isin(query_idents)].copy()
        df_workload = df_workload[['query_ident', 'method', 'total_time']].copy()
        labeled_data = []
        for query in df_workload['query_ident'].unique():
            query_data = df_workload[df_workload['query_ident'] == query]
            sorted_data = query_data.sort_values('total_time')  # Sort by total_time ascending
            methods = sorted_data['method'].values

            execution_time_dict = dict(zip(query_data['method'], query_data['total_time']))
            labels = {
                'query_ident': query,
                'top_1_method': methods[0],
                'top_2_methods': list(methods[:2]) if len(methods) >= 2 else list(methods),  # Top 2 methods
                'execution_time_ms': json.dumps(execution_time_dict)
            }
            labeled_data.append(labels)

        # Create DataFrame for this workload
        df_labeled = pd.DataFrame(labeled_data)

        # Add raw SQL strings
        df_labeled['query_sql'] = df_labeled['query_ident'].apply(lambda q: cfg.load_sql_query(q))

        # Sort by query_ident for consistency
        df_labeled = df_labeled.sort_values('query_ident')

        # Save to disk
        output_file = os.path.join(save_dir, f"workload_{workload}_test_data.csv")
        df_labeled.to_csv(output_file, index=False)
        print(f"Saved dataset for '{workload}' to '{output_file}' with {len(df_labeled)} rows")

        # Store in dictionary
        workload_datasets[workload] = df_labeled

        # Log a sample for verification
        print(f"Sample data for '{workload}':")
        print(df_labeled.head())

    return workload_datasets


def prepare_workload_train_test_datasets(
        save_dir: str, df: pd.DataFrame, test_queries_dict: Dict[str, List[str]],
        cfg: BaseConfig) -> \
        Dict[str, Dict[str, pd.DataFrame]]:

    workload_datasets = {}
    os.makedirs(save_dir, exist_ok=True)

    for workload, _ in test_queries_dict.items():
        # Filter df for the specific experiment/workload
        df_workload = df[df['experiment'] == workload].copy()
        # df_workload = df_workload[df_workload['query_ident'].isin(query_idents)].copy()

        # Split into train and test based on 'split' column
        df_train = df_workload[df_workload['split'] == 'train'][['query_ident', 'method', 'total_time']].copy()
        df_test = df_workload[df_workload['split'] == 'test'][['query_ident', 'method', 'total_time']].copy()

        # Compute labels for train and test separately
        train_labeled_data = []
        test_labeled_data = []

        # Train labels
        for query in df_train['query_ident'].unique():
            query_data = df_train[df_train['query_ident'] == query]
            sorted_data = query_data.sort_values('total_time')
            methods = sorted_data['method'].values
            execution_time_dict = dict(zip(query_data['method'], query_data['total_time']))
            labels = {
                'query_ident': query,
                'top_1_method': methods[0],
                'top_2_methods': list(methods[:2]) if len(methods) >= 2 else list(methods),
                'execution_time_ms': json.dumps(execution_time_dict)
            }
            train_labeled_data.append(labels)

        # Test labels
        for query in df_test['query_ident'].unique():
            query_data = df_test[df_test['query_ident'] == query]
            sorted_data = query_data.sort_values('total_time')
            methods = sorted_data['method'].values
            execution_time_dict = dict(zip(query_data['method'], query_data['total_time']))
            labels = {
                'query_ident': query,
                'top_1_method': methods[0],
                'top_2_methods': list(methods[:2]) if len(methods) >= 2 else list(methods),
                'execution_time_ms': json.dumps(execution_time_dict)
            }
            test_labeled_data.append(labels)

        # Create DataFrames
        df_train_labeled = pd.DataFrame(train_labeled_data).sort_values('query_ident')
        df_test_labeled = pd.DataFrame(test_labeled_data).sort_values('query_ident')

        # Add raw SQL strings
        df_train_labeled['query_sql'] = df_train_labeled['query_ident'].apply(lambda q: cfg.load_sql_query(q))
        df_test_labeled['query_sql'] = df_test_labeled['query_ident'].apply(lambda q: cfg.load_sql_query(q))

        # Save to disk
        train_output_file = os.path.join(save_dir, f"workload_{workload}_train_data.csv")
        test_output_file = os.path.join(save_dir, f"workload_{workload}_test_data.csv")

        df_train_labeled.to_csv(train_output_file, index=False)
        df_test_labeled.to_csv(test_output_file, index=False)

        print(f"Saved train dataset for '{workload}' to '{train_output_file}' with {len(df_train_labeled)} rows")
        print(f"Saved test dataset for '{workload}' to '{test_output_file}' with {len(df_test_labeled)} rows")

        # Store in dictionary
        workload_datasets[workload] = {
            'train': df_train_labeled,
            'test': df_test_labeled
        }

        # Log samples
        print(f"Sample train data for '{workload}':")
        print(df_train_labeled.head())
        print(f"Sample test data for '{workload}':")
        print(df_test_labeled.head())

    return workload_datasets


def prepare_router_dataset(save_dir: str,
                           df: pd.DataFrame, train_queries: List[str], test_queries: List[str],
                           seed,
                           cfg: BaseConfig) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare train and test datasets for the router with raw SQL strings."""
    train_perf = compute_avg_performance(df, train_queries, 'train')
    test_perf = compute_avg_performance(df, test_queries, 'test')

    train_labeled = assign_labels(train_perf)
    test_labeled = assign_labels(test_perf)

    # Add raw SQL strings instead of query features
    for dataset in [train_labeled, test_labeled]:
        dataset['query_sql'] = dataset['query_ident'].apply(lambda q: cfg.load_sql_query(q))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create folder named with seed
    folder_name = f"{save_dir}/data_stack_seed_{seed}"
    os.makedirs(folder_name, exist_ok=True)

    # Save datasets into the folder
    train_labeled.to_csv(os.path.join(folder_name, f"train_data_{timestamp}.csv"), index=False)
    test_labeled.to_csv(os.path.join(folder_name, f"test_data_{timestamp}.csv"), index=False)
    print(f"Router datasets saved: {folder_name}/train_data_{timestamp}.csv, {folder_name}/test_data_{timestamp}.csv")

    return train_labeled, test_labeled


def load_merged_datasets(folder_name: str) -> Dict[str, pd.DataFrame]:
    # Load train and test datasets
    train_files = sorted(glob.glob(os.path.join(folder_name, 'train_data_*.csv')))
    test_files = sorted(glob.glob(os.path.join(folder_name, 'test_data_*.csv')))

    if not train_files or not test_files:
        raise FileNotFoundError(f"No train or test files found in {folder_name}")

    latest_train_file = train_files[-1]
    latest_test_file = test_files[-1]

    datasets = {
        'train': pd.read_csv(latest_train_file),
        'test': pd.read_csv(latest_test_file)
    }

    # Convert JSON strings back to dictionaries
    for split in ['train', 'test']:
        datasets[split]['execution_time_ms'] = datasets[split]['execution_time_ms'].apply(json.loads)

    return datasets


def load_workload_test_datasets(base_dir) -> Dict[str, pd.DataFrame]:
    datasets = {}
    workload_files = sorted(glob.glob(os.path.join(base_dir, 'workload_*_test_data.csv')))  # Fixed pattern
    for file_path in workload_files:
        filename = os.path.basename(file_path)
        workload_name = filename.replace('workload_', '').replace('_test_data.csv', '')
        datasets[workload_name] = pd.read_csv(file_path)
        print(f"Loaded dataset for workload '{workload_name}' from {file_path}")
        datasets[workload_name]['execution_time_ms'] = datasets[workload_name]['execution_time_ms'].apply(json.loads)

    return datasets


def load_workload_train_test_datasets(folder_name: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    datasets = {}

    # Find all workload train/test files
    train_files = glob.glob(os.path.join(folder_name, "workload_*_train_data.csv"))
    if len(train_files) == 0:
        raise
    for train_file in train_files:
        # Extract workload name from filename
        workload = os.path.basename(train_file).replace("workload_", "").replace("_train_data.csv", "")
        test_file = os.path.join(folder_name, f"workload_{workload}_test_data.csv")

        datasets[workload] = {
            'train': pd.read_csv(train_file),
            'test': pd.read_csv(test_file)
        }
        for split in ['train', 'test']:
            datasets[workload][split]['execution_time_ms'] = datasets[workload][split]['execution_time_ms'].apply(
                json.loads)

    return datasets


def validate(df_agg):
    import numpy as np
    df_sum = df_agg \
        .groupby(['base_experiment', 'experiment', 'method', 'split']) \
        .agg({
        'total_time': 'sum', 'execution_time': 'sum', 'planning_time': 'sum', 'inference_time': 'sum',
        'prepare_time': 'sum',
        'total_time.std': 'sum', 'execution_time.std': 'sum', 'planning_time.std': 'sum', 'inference_time.std': 'sum',
        'prepare_time.std': 'sum',
        'n': 'mean'
    }) \
        .reset_index()

    df_sum['execution_time.err'] = 1.96 * df_sum['execution_time.std'] / np.sqrt(df_sum['n'])
    df_sum['planning_time.err'] = 1.96 * df_sum['planning_time.std'] / np.sqrt(df_sum['n'])
    df_sum['inference_time.err'] = 1.96 * df_sum['inference_time.std'] / np.sqrt(df_sum['n'])
    df_sum['total_time.err'] = 1.96 * df_sum['total_time.std'] / np.sqrt(df_sum['n'])
    df_sum['prepare_time.err'] = 1.96 * df_sum['prepare_time.std'] / np.sqrt(df_sum['n'])
    return df_sum


if __name__ == "__main__":
    cfg = get_config("stack")
    df_processed = load_and_process_data(cfg=cfg)
    df_sum = validate(df_processed)

    _ = prepare_workload_test_datasets(
        save_dir=cfg.TESTPATH,
        df=df_processed,
        test_queries_dict=cfg.TEST_QUERIES,
        cfg=cfg
    )

    _ = prepare_workload_train_test_datasets(
        save_dir=cfg.TRAIN_TEST,
        df=df_processed,
        test_queries_dict=cfg.TEST_QUERIES,
        cfg=cfg
    )
