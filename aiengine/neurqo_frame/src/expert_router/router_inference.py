# Standard library imports
import json
import time
from collections import Counter
from parser.table_parser import load_db_info_json
from typing import Tuple

# Local/project imports
from common import BaseConfig, get_config
from utils.io import set_global_seed

from .controller_offline import ModelBuilder
from .dataset import get_data_loader
from .encoder import Sql2VecEmbeddingV2
from .logger import plogger
from .workloads import load_workload_test_datasets


def map_back_to_method(predictions: dict, cfg: BaseConfig) -> dict:
    id_to_method = {idx: method for method, idx in cfg.FIXED_LABEL_MAPPING.items()}
    predicted_methods = {
        query: id_to_method[pred] for query, pred in predictions.items()
    }
    return predicted_methods


def compute_total_time_for_predictions(
    df_processed,
    query_method: dict,
    workload_name: str,
    avg_inference_time_per_query_cpu,
) -> Tuple[float, dict]:
    df_workload = df_processed[df_processed["experiment"] == workload_name].copy()
    total_time_sum = 0.0
    per_query_time = {}
    for query_id, predicted_method in query_method.items():
        # Find the row in df_workload for this query_id and predicted_method
        query_method_row = df_workload[
            (df_workload["query_ident"] == query_id)
            & (df_workload["method"] == predicted_method)
        ]

        total_time = query_method_row["total_time"].iloc[0]
        inference_time = query_method_row["inference_time"].iloc[0]
        planning_time = query_method_row["planning_time"].iloc[0]
        execution_time = query_method_row["execution_time"].iloc[0]

        total_time_sum += total_time
        per_query_time[query_id] = {
            "prepare_time": inference_time
            + planning_time
            + avg_inference_time_per_query_cpu,
            "execution_time": execution_time,
        }
        # print(f"Query: {query_id}, Method: {predicted_method}, Total Time: {total_time}")

    print(f"Total time for workload '{workload_name}': {total_time_sum}")
    return total_time_sum, per_query_time


def inference_single_workload(
    df_processed,
    model_path,
    test_data_path: str,
    batch_size: int,
    dataset: str,
    cfg: BaseConfig,
):
    SysOnworkloadCollector = {}
    query_per_workload = {}

    # from arg_utils import shared_args
    # from dataset.data_process import DataProcessor
    # _args = shared_args.get_stats_arg_parser()
    # _, all_paths = shared_args.init_system_dirs(_args)
    # data_processor = DataProcessor(_args)
    # data_processor.dataset_profiling(all_paths)

    # feature instance
    db_profile_res = load_db_info_json(cfg.DB_INFO_DICT)
    sql_vec = Sql2VecEmbeddingV2(
        config=cfg.CONFIG, db_profile_res=db_profile_res, checkpoint_file=cfg.EMBED_FILE
    )

    fq_instance = sql_vec

    datasets = load_workload_test_datasets(test_data_path)
    sorted_exps = sorted(datasets.keys())
    print(sorted_exps)

    for workload_name in sorted_exps:
        test_df = datasets[workload_name]
        data_loaders, input_dim, output_dim = get_data_loader(
            cfg=cfg,
            workload_name=workload_name,
            datasets={"test": test_df},
            batch_size=batch_size,
            fq_instance=fq_instance,
            threshold=None,
        )

        builder = ModelBuilder(
            num_tables=len(set(db_profile_res.table_no_map.values())),
            num_columns=108,
            output_dim=output_dim,
            model_path_prefix=f"{model_path}/{workload_name}_model",
            embedding_path=None,  # load the whole model
            num_heads=4,
            embedding_dim=256,
            is_fix_emb=None,
            num_layers=2,
            dataset=dataset,
            cfg=cfg,
        )

        builder.load_model("hypered_2")
        start_time = time.time()  # Start timing
        predictions_dict, predictions_time = builder.inference(
            data_loaders["test"], save_embedding=True
        )
        end_time = time.time()  # End timing

        total_inference_time = end_time - start_time
        avg_inference_time_per_query_cpu = total_inference_time / len(test_df)

        print("avg_inference_time_per_query", avg_inference_time_per_query_cpu)
        print(predictions_time)
        print(predictions_dict)
        query_method = map_back_to_method(predictions_dict, cfg=cfg)
        total_time_sum, per_query_time = compute_total_time_for_predictions(
            df_processed, query_method, workload_name, avg_inference_time_per_query_cpu
        )

        # single query
        query_per_workload[workload_name] = per_query_time

        # sum query
        SysOnworkloadCollector[workload_name] = total_time_sum + total_inference_time

        plogger.info(f"{workload_name}: {dict(Counter(query_method.values()))}")
        plogger.info(f"{workload_name}: {total_time_sum}")
        plogger.info(f"\n")

    with open(f"{cfg.RESULT_DATA_BASE}/inference_res_{dataset}_for_figure", "w") as f:
        json.dump(query_per_workload, f)
    print("inference_single_workload", SysOnworkloadCollector)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Figure")
    parser.add_argument("--dataset", help="imdb or stack")
    args = parser.parse_args()

    cfg = get_config(args.dataset)

    if args.dataset == "imdb":
        from .workloads import load_and_process_data

        used_load_and_process_data = load_and_process_data

    elif args.dataset == "stack":
        from .workloads_stack import load_and_process_data

        used_load_and_process_data = load_and_process_data

    else:
        raise 0

    set_global_seed(2550)
    df_processed = used_load_and_process_data(cfg=cfg)

    inference_single_workload(
        df_processed=df_processed,
        model_path=f"./experiment_result/models/",
        test_data_path=cfg.TESTPATH,
        batch_size=160,
        dataset=args.dataset,
        cfg=cfg,
    )
