# Standard library imports
import argparse
import json

# Local/project imports
from common import IMDBConfig, StackConfig
from parser.table_parser import load_db_info_json
from utils.io import set_global_seed

from .controller_offline import ModelBuilder
from .dataset import get_data_loader
from .encoder import Sql2VecEmbeddingV2
from .logger import plogger
from .workloads import load_workload_train_test_datasets


def train_single_workload(args, cfg):
    db_profile_res = load_db_info_json(cfg.DB_INFO_DICT)
    sql_vec = Sql2VecEmbeddingV2(config=cfg.CONFIG,
                                 db_profile_res=db_profile_res,
                                 checkpoint_file=cfg.EMBED_FILE)
    fq_instance = sql_vec

    datasets = load_workload_train_test_datasets(folder_name=cfg.TRAIN_TEST)

    sorted_exps = sorted(datasets.keys())
    if args.single_exp not in sorted_exps:
        raise f"{args.single_exp} not exist in give worklaods {sorted_exps}"
    workload_name = args.single_exp

    data_loaders, input_dim, output_dim = get_data_loader(
        cfg=cfg,
        datasets=datasets[workload_name],
        batch_size=args.batch_size,
        fq_instance=fq_instance,
        threshold=args.threshold,
        workload_name=workload_name,
    )

    builder = ModelBuilder(
        num_tables=len(set(db_profile_res.table_no_map.values())),
        num_columns=108,
        output_dim=output_dim,
        model_path_prefix=f"{args.model_path}/{workload_name}_model",
        embedding_path=args.model_path if args.load_embed else None,
        num_heads=4,
        embedding_dim=args.embedding_dim,
        is_fix_emb=args.is_fix_emb,
        num_layers=args.num_self_attn_layers,
        dataset=args.dataset,
        cfg=cfg)

    running_history, max_epoch_saved, saved_model_log_str = builder.run(
        data_loaders["train"],
        data_loaders,
        epochs=args.epochs,
        lr=args.lr,
        step_size=args.step_size,
        gamma=args.gamma,
        alpha=args.alpha,
        loss_gamma=args.loss_gamma,
        workload_name=workload_name)

    print("max_epoch_saved for train_single_workload is ")
    print(json.dumps(max_epoch_saved, indent=4))
    print("----->", saved_model_log_str)
    with open(f"./running_history_{workload_name}_{args.dataset}", "w") as f:
        json.dump(running_history, f)


if __name__ == "__main__":
    set_global_seed(2550)
    parser = argparse.ArgumentParser()

    parser.add_argument("--single_exp", type=str,
                        default="base_query_split_1",
                        help="Run training on a single workload")
    # datasets
    parser.add_argument('--dataset', type=str, default="imdb", help='imdb, stack, tpch')
    parser.add_argument("--folder_name", type=str,
                        default="./experiment_result/datasets/workload_data_train_test",
                        help="Folder path for training data")

    parser.add_argument("--train_data_path", type=str,
                        default="./experiment_result/datasets/data_seed_137",
                        help="Path to merged training data")
    parser.add_argument("--test_data_path", type=str,
                        default="./experiment_result/datasets/workload_data_test",
                        help="Path to test data")
    parser.add_argument("--model_path", type=str,
                        default="./experiment_result/models/",
                        help="Path to save the trained model")

    parser.add_argument("--epochs", type=int,
                        default=2000, help="Number of training epochs")
    parser.add_argument("--lr", type=float,
                        default=0.0005, help="Learning rate")

    parser.add_argument("--load_embed", action='store_true',
                        default=False,
                        help="Set this flag to load embeddings.")

    parser.add_argument("--is_fix_emb", action='store_true',
                        default=False,
                        help="Set this flag to finetune embeddings or not")

    parser.add_argument("--threshold", type=float,
                        default=0.02, help="Threshold for filtering data (the label arounds)")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="Batch size for training")

    parser.add_argument("--embedding_dim", type=int,
                        default=256, help="Embedding dimension size, same as train_pretrain_embedding2.py")
    parser.add_argument('--num_self_attn_layers', type=int, default=2, help='num_self_attn_layers')

    # New arguments for step_size and gamma
    parser.add_argument('--step_size', type=int, default=500, help='Step size for the learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for the learning rate scheduler')
    parser.add_argument('--alpha', type=float, default=0.5, help='Balance classificaiton and regression')
    parser.add_argument('--loss_gamma', type=float, default=2, help='A higher gamma in focal loss focuses more on hard-to-classify examples')

    args = parser.parse_args()
    plogger.info(args)

    if args.dataset == "imdb":
        cfg = IMDBConfig
    elif args.dataset == "stack":
        cfg = StackConfig
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_single_workload(args, cfg)
