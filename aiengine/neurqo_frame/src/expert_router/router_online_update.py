# Standard library imports
import argparse
import json
import os
import time
from parser.table_parser import load_db_info_json
from typing import Dict

# Third-party imports
import pandas as pd
import torch

# Local/project imports
from common import IMDBConfig, StackConfig
from utils.io import set_global_seed

from .controller_offline import ModelBuilder
from .controller_online import OnlineRouter
from .dataset import get_data_loader_online
from .encoder import Sql2VecEmbeddingV2
from .logger import plogger


def get_single_exp_dataset(online_data_path, experiment_str) -> Dict[str, pd.DataFrame]:
    online_files = os.listdir(online_data_path)
    res = {}
    for online_csv_file in online_files:
        if experiment_str not in online_csv_file:
            continue
        online_csv = pd.read_csv(os.path.join(online_data_path, online_csv_file))
        print(f"loading dataset from {online_csv_file}")
        online_csv["execution_time_ms"] = online_csv["execution_time_ms"].apply(
            json.loads
        )
        res[online_csv_file] = online_csv
    return res


def train_online_workload(args, cfg):
    db_profile_res = load_db_info_json(cfg.DB_INFO_DICT)
    sql_vec = Sql2VecEmbeddingV2(
        config=cfg.CONFIG, db_profile_res=db_profile_res, checkpoint_file=cfg.EMBED_FILE
    )
    fq_instance = sql_vec

    workload_name = args.single_exp
    dataframe_dict = get_single_exp_dataset(
        online_data_path=cfg.TRAIN_TEST_ONLINE_CONVARIATE, experiment_str=workload_name
    )

    for shift_degree, df_data in dataframe_dict.items():
        print(f"Begin {shift_degree} ======================================= ")
        if workload_name not in shift_degree:
            continue
        data_loaders, input_dim, output_dim = get_data_loader_online(
            cfg=cfg,
            datasets={"train": df_data},
            batch_size=args.batch_size,
            fq_instance=fq_instance,
            threshold=args.threshold,
            workload_name=workload_name,
        )
        data_loader = data_loaders["train"]

        pretrained_builder = ModelBuilder(
            num_tables=len(set(db_profile_res.table_no_map.values())),
            num_columns=108,
            output_dim=output_dim,
            model_path_prefix=f"{args.model_path}/{workload_name}_model",
            embedding_path=args.model_path if args.load_embed else None,
            num_heads=4,
            embedding_dim=args.embedding_dim,
            is_fix_emb=True,  # Fix embedding during online tuning
            num_layers=args.num_self_attn_layers,
            dataset=args.dataset,
            cfg=cfg,
        )

        pretrained_builder.load_model(evl_method="hypered_2")

        pretrained_model = pretrained_builder.model

        online_router = OnlineRouter(
            cfg=cfg,
            pretrained_model=pretrained_model,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            update_frequency=args.update_frequency,
            lr=args.online_lr,
            decay_rate=args.decay_rate,
            regret_window=args.regret_window,
            dropout_samples=args.dropout_samples,
            alpha=args.alpha,
            loss_gamma=args.loss_gamma,
            class_weights=data_loader.dataset.get_class_weights(),
        )

        (
            offlline_cur_cumulated_latency,
            offline_cumulated_time,
            offline_query_selection,
            per_query_offline,
        ) = online_router.inference(data_loader)
        assert offlline_cur_cumulated_latency == offline_cumulated_time[-1]
        # === Online learning started ===
        cumulated_time = []  # Without overhead
        cumulated_time_with_overhead = []  # With inference and update overhead
        cur_cumulated_latency = 0
        cur_cumulated_latency_with_overhead = 0
        cur_selection = []

        opt_selection = []
        opt_cumulated_time = []
        opt_cur_cumulated_latency = 0
        per_query_online = []
        per_query_online_with_overhead = []

        for _ in range(1):
            for X_batch, y_top_1, y_times, query_list in data_loader:
                batch_size = y_times.size(0)
                # Optimal selection and latency (for comparison)
                opt_selection.extend(
                    [
                        (
                            query_id,
                            (
                                torch.where(row == 1)[0].tolist()[0]
                                if len(torch.where(row == 1)[0].tolist()) == 1
                                else len(torch.where(row == 1)[0].tolist())
                            ),
                        )
                        for query_id, row in zip(query_list, y_top_1)
                    ]
                )

                min_latencies = torch.min(y_times, dim=1).values.cpu().tolist()
                for latency in min_latencies:
                    opt_cur_cumulated_latency += latency
                    opt_cumulated_time.append(opt_cur_cumulated_latency)

                # Process each query in the batch one-by-one
                for i in range(batch_size):
                    X_single = {key: X_batch[key][i : i + 1] for key in X_batch}

                    # 1. Measure inference time, to mill second
                    start_time = time.time()
                    chosen_optimizer = online_router.select_optimizer(
                        X_single, query_list[i]
                    )
                    inference_time = (time.time() - start_time) * 1000

                    cur_selection.append((query_list[i], chosen_optimizer))

                    # 2. Collect observed latency
                    observed_latency = y_times[i, chosen_optimizer].item()
                    cur_cumulated_latency += observed_latency
                    cumulated_time.append(cur_cumulated_latency)
                    per_query_online.append(observed_latency)

                    # 3. Measure update time and add to overhead
                    start_time = time.time()
                    online_router.add_to_buffer_update_model(
                        X_single,
                        chosen_optimizer,
                        observed_latency,
                        query_list[i],
                    )
                    # to mill second
                    update_time = (time.time() - start_time) * 1000

                    # Add overhead to cumulated_time_with_overhead
                    cur_cumulated_latency_with_overhead += (
                        observed_latency + inference_time + update_time
                    )
                    cumulated_time_with_overhead.append(
                        cur_cumulated_latency_with_overhead
                    )
                    per_query_online_with_overhead.append(
                        observed_latency + inference_time + update_time
                    )

        all_result = {
            "online_cumulated_time": cumulated_time,
            "online_cumulated_time_with_overhead": cumulated_time_with_overhead,  # New field
            "online_selection": cur_selection,
            "opt_selection": opt_selection,
            "opt_cur_cumulated_latency": opt_cur_cumulated_latency,
            "offline_cumulated_time": offline_cumulated_time,
            "offline_selection": offline_query_selection,
            "per_query_offline": per_query_offline,
            "per_query_online": per_query_online,
            "per_query_online_with_overheads": per_query_online_with_overhead,
        }

        all_result_save = {
            "online_selection": cur_selection,
            "opt_selection": opt_selection,
            "offline_selection": offline_query_selection,
        }

        with open(
            f"./reinforce222222_convariate_shift_inference_{args.dataset}_{args.single_exp}_{args.is_train_model}.json",
            "w",
        ) as f:
            json.dump(all_result, f)
        # with open(f'./convariate_shift_inference_{args.dataset}_{args.single_exp}_debug_{args.is_train_model}.json',
        #           'w') as f:
        #     json.dump(all_result_save, f)

        print("Final result: ")
        cumulated_time = all_result["online_cumulated_time"][-1]
        cumulated_time_with_overhead = all_result[
            "online_cumulated_time_with_overhead"
        ][-1]
        opt_cur_cumulated_latency = all_result["opt_cur_cumulated_latency"]
        offline_cumulated_time = all_result["offline_cumulated_time"][-1]

        print(
            f"final offline test, {len(all_result['online_cumulated_time_with_overhead'])}"
        )
        (
            offlline_cur_cumulated_latency2,
            offline_cumulated_time2,
            offline_query_selection2,
            _,
        ) = online_router.inference(data_loader)

        print(
            f"shift_degree = {shift_degree}, dis without online after online is {offline_cumulated_time2[-1] - opt_cur_cumulated_latency}"
        )

        print(
            f"shift_degree = {shift_degree}, dis with online (no overhead) is {cumulated_time - opt_cur_cumulated_latency}"
        )
        print(
            f"shift_degree = {shift_degree}, dis with online (with overhead) is {cumulated_time_with_overhead - opt_cur_cumulated_latency}"
        )
        print(
            f"shift_degree = {shift_degree}, dis without online is {offline_cumulated_time - opt_cur_cumulated_latency}"
        )

        print(
            f"shift_degree = {shift_degree}, offline - online = {offline_cumulated_time - cumulated_time}, per = {100 * (offline_cumulated_time - cumulated_time) / offline_cumulated_time}"
        )
        print(
            f"shift_degree = {shift_degree}, offline_cumulated_time - online with overheads = {offline_cumulated_time - cumulated_time_with_overhead}, per = {100 * (offline_cumulated_time - cumulated_time_with_overhead) / offline_cumulated_time}"
        )
        print(online_router.his_track)


if __name__ == "__main__":
    set_global_seed(2550)
    parser = argparse.ArgumentParser()
    # Existing arguments (unchanged)
    parser.add_argument("--dataset", type=str, default="imdb", help="imdb, stack, tpch")
    parser.add_argument(
        "--epochs", type=int, default=4000, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./experiment_result/models/",
        help="Path to save/load the trained model",
    )

    parser.add_argument(
        "--load_embed",
        action="store_true",
        default=False,
        help="Set this flag to load embeddings.",
    )

    parser.add_argument(
        "--is_fix_emb",
        action="store_true",
        default=False,
        help="Set this flag to finetune embeddings or not",
    )

    parser.add_argument(
        "--threshold", type=float, default=0.02, help="Threshold for labels"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--embedding_dim", type=int, default=256, help="Embedding dimension"
    )
    # parser.add_argument("--single_exp", type=str, default="random_split_3", help="Workload name")
    # parser.add_argument("--single_exp", type=str, default="base_query_split_1", help="Workload name")
    # parser.add_argument("--single_exp", type=str, default="base_query_split_2", help="Workload name")
    parser.add_argument(
        "--single_exp", type=str, default="base_query_split_3", help="Workload name"
    )

    parser.add_argument(
        "--is_train_model",
        action="store_true",
        default=False,
        help="Set this flag to update model online",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "online"],
        help="Training mode: single (offline) or online",
    )
    parser.add_argument(
        "--num_self_attn_layers", type=int, default=4, help="Number of attention layers"
    )
    parser.add_argument(
        "--step_size", type=int, default=500, help="Step size for LR scheduler"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="Gamma for LR scheduler"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Balance classification/regression"
    )
    parser.add_argument("--loss_gamma", type=float, default=3, help="Focal loss gamma")
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.01,
        help=" For time-weighted sampling in the OnlineRouter",
    )
    parser.add_argument(
        "--regret_window", type=float, default=50, help="For shift detection"
    )
    parser.add_argument(
        "--dropout_samples", type=float, default=20, help="Number of MC Dropout samples"
    )

    # New arguments for online training
    parser.add_argument(
        "--online_lr", type=float, default=1e-5, help="Learning rate for online updates"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=200, help="Size of replay buffer"
    )
    parser.add_argument(
        "--update_frequency",
        type=int,
        default=1,
        help="Update frequency for online model",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon for epsilon-greedy exploration",
    )

    args = parser.parse_args()
    plogger.info(args)

    print(f"\n ---------------> Running {args.single_exp}")
    # Set dataset-specific constants
    if args.dataset == "imdb":
        cfg = IMDBConfig
    elif args.dataset == "stack":
        cfg = StackConfig
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_online_workload(args, cfg)
    plogger.info("Online training completed")
