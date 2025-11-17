import os
import pickle
import random
import traceback
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from math import log
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from db.pg_conn import PostgresConnector
from exp_buffer.buffer_mngr import BufferManager
from exp_buffer.sqllite import PlanRecord

# Import from local package
from expert_pool.join_order_expert.encoders.mcts_encoder import TreeBuilder
from expert_pool.join_order_expert.encoders.sql_to_vec import Sql2Vec
from expert_pool.join_order_expert.KNN import KNN
from expert_pool.join_order_expert.mcts import MCTSHinterSearch
from expert_pool.join_order_expert.models.mcts_net import TreeSQLNet, TreeSQLNetBuilder
from expert_pool.join_order_expert.tools.normalize import LatencyNormalizer
from torch.utils.data import DataLoader, Dataset


class ReplayMemory:
    Transition = namedtuple(
        "Transition",
        ("tree_feature", "sql_feature", "target_feature", "mask", "weight"),
    )

    """Replay memory for training. Based on hybrid_qo/NET.py ReplayMemory."""

    def __init__(self, capacity: int):
        """
        Args:
            capacity (int): Maximum number of transitions to store in memory.
        """
        self.capacity = capacity
        self.memory: List[Optional["ReplayMemory.Transition"]] = []
        self.position: int = 0

    def push(self, transition: "ReplayMemory.Transition") -> None:
        """
        Save a new transition.

        Args:
            transition (ReplayMemory.Transition): A transition object.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def weight_sample(self, batch_size: int) -> List[int]:
        """
        Weighted sampling based on transition weights.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            List[int]: List of sampled indices based on weight.
        """
        weight = []
        current_weight = 0
        for x in self.memory:
            current_weight += x.weight
            weight.append(current_weight)
        for idx in range(len(self.memory)):
            weight[idx] = weight[idx] / current_weight
        return random.choices(
            population=list(range(len(self.memory))), weights=weight, k=batch_size
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[List["ReplayMemory.Transition"], List[int]]:
        """
        Randomly sample a batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[List[Transition], List[int]]: A list of sampled transitions and their indices.
        """
        if len(self.memory) > batch_size:
            normal_batch = batch_size // 2
            idx_list1 = [
                random.randint(0, normal_batch - 1) for _ in range(normal_batch)
            ]
            idx_list2 = self.weight_sample(batch_size=batch_size - normal_batch)
            idx_list = idx_list1 + idx_list2
            res = [self.memory[idx] for idx in idx_list]
            return res, idx_list
        else:
            return self.memory, list(range(len(self.memory)))

    def update_weight(self, idx_list: List[int], weight_list: List[float]) -> None:
        """
        Update the weights of sampled transitions.

        Args:
            idx_list (List[int]): List of indices to update.
            weight_list (List[float]): Corresponding new weights.

        Returns:
            None
        """
        for idx, wei in zip(idx_list, weight_list):
            self.memory[idx] = self.memory[idx]._replace(weight=wei)

    def __len__(self) -> int:
        """
        Returns:
            int: Number of transitions currently stored.
        """
        return len(self.memory)

    def reset_memory(self) -> None:
        """
        Clears the replay memory.

        Returns:
            None
        """
        self.memory = []
        self.position = 0


class MCTSMemory:
    MCTSSample = namedtuple("MCTSSample", ("sql_feature", "order_feature", "label"))

    def __init__(self, capacity: int):
        """
        Replay memory for storing MCTS transitions.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.memory: List[MCTSMemory.MCTSSample] = []
        self.position = 0

    def push(self, sample: "MCTSMemory.MCTSSample"):
        """
        Save a transition into memory.

        Args:
            sample (MCTSSample): A transition sample containing
                - sql_feature
                - order_feature
                - label
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        Sample a random batch of transitions.

        Args:
            batch_size (int): Number of samples to retrieve.

        Returns:
            List[MCTSSample]: A batch of randomly selected transitions.
        """
        import random

        if len(self.memory) > batch_size:
            return random.sample(self.memory, batch_size)
        else:
            return self.memory

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return len(self.memory)

    def reset_memory(self):
        """Clear the memory."""
        self.memory = []
        self.position = 0


@dataclass
class MCTSConfig:
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # 1) MCTS search hyperparameters
    mcts_rollouts: int = 200
    tree_policy: Literal["uct", "puct"] = "uct"
    c_puct: float = 1.0
    max_depth: int = 64

    try_hint_num: int = 3

    # 2) Model architecture (TreeSQLNet / encoders)
    head_num: int = 10  # from previous Config
    input_size: int = 9  # 7 op-types + 2 ['total cost','plan rows']
    hidden_size: int = 64

    # 3) Training runtime
    batch_size: int = 256
    train_epochs: int = 10  # n_epochs in previous Config

    # 4) Control knobs / thresholds
    var_weight: float = 0.00
    max_hint_num: int = 20
    leading_length: int = 2
    threshold: float = log(3) / log(3 * 60 * 1000)  # log(3)/log(max_time_out)

    # 5) Normalization params
    offset: int = 20
    max_time_out: int = 3 * 60 * 1000  # ms
    max_value: int = 20

    # 6) Dataset / logging context
    train_database: str = ""
    test_database: str = ""

    # 7) Memory sizing
    mem_size: int = 2000

    # 8) Dataset-specific mappings / sizes
    max_alias_num: int = 40
    max_column: int = 100
    id2aliasname: Dict[int, str] = field(default_factory=dict)
    aliasname2id: Dict[str, int] = field(default_factory=dict)
    table_num: int = 50

    # 9) Extra MCTS tunables you had
    mcts_v: float = 1.1
    searchFactor: int = 4
    U_factor: float = 0.0

    # 10) Model paths
    cur_model_path: str = "./models/join_order_exp_models"
    temp_model_path: str = "./models/join_order_exp_models/temp"

    # (computed)
    @property
    def mcts_input_size(self) -> int:
        """(#alias × #alias) + #columns"""
        return self.max_alias_num * self.max_alias_num + self.max_column


class MCTSDataset(Dataset):
    def __init__(self, sql_plan, exe_time):
        self.sql_plan = sql_plan
        self.exe_time = exe_time

    def __len__(self):
        return len(self.sql_plan)

    def __getitem__(self, idx):
        sql_feature, tree_feature, alias, mask = self.sql_plan[idx]
        actual_latency = self.exe_time[idx]
        return sql_feature, tree_feature, alias, mask, actual_latency


class MCTSOptimizerExpert:

    def __init__(
        self,
        config: MCTSConfig,
        buffer_mngr: BufferManager,
        db_cli: PostgresConnector = None,
    ):
        self.config = config
        self.buffer_mngr = buffer_mngr

        # normalizer
        self.latency_normalizer = LatencyNormalizer(
            offset=self.config.offset,
            max_time_out=self.config.max_time_out,
            max_value=self.config.max_value,
        )

        # --- encoders
        self.plan_builder = TreeBuilder(
            id2aliasname=config.id2aliasname,
            aliasname2id=config.aliasname2id,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            latency_normalizer=self.latency_normalizer,
        )

        self.sql2vec = Sql2Vec(db_cli=db_cli, config=self.config)

        # --- TreeSQLNet value network
        self.value_network = TreeSQLNet(
            head_num=config.head_num,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            table_num=config.table_num,
            sql_size=config.max_alias_num * config.max_alias_num + config.max_column,
        ).to(config.device)
        self.tree_sql_builder = TreeSQLNetBuilder(
            value_network=self.value_network, var_weight=config.var_weight
        )

        # --- MCTS searcher
        self.mcts_searcher = MCTSHinterSearch(
            device=config.device,
            max_alias_num=config.max_alias_num,
            hidden_size=config.hidden_size,
            mcts_input_size=config.mcts_input_size,
            offset=config.offset,
            max_time_out=config.max_time_out,
            mcts_v=config.mcts_v,
            searchFactor=config.searchFactor,
            try_hint_num=config.try_hint_num,
            max_hint_num=config.max_hint_num,
        )

        # --- KNN
        self.knn = KNN(10)

        # bookkeeping
        self.run_name = datetime.now().strftime("%Y_%m_%d__%H%M%S")

        # --- Memory to assist the training.
        self.experience_memory = ReplayMemory(config.mem_size)
        self.mcts_memory = MCTSMemory(capacity=5000)

        # Model directory (use cur_model_path from config)
        self.model_dir = config.cur_model_path
        os.makedirs(self.model_dir, exist_ok=True)

    # ---------------- public API ----------------
    def train_and_save(self, train_data_limit: int = 10):
        training_data = self.buffer_mngr.get_plan_latency_pairs(limit=train_data_limit)
        if training_data is None or len(training_data) == 0:
            raise ValueError("No training data available")

        loader = self._get_data_loader(training_data)

        for epoch in range(self.config.train_epochs):
            print(f"Trainig epoch {epoch}")
            self._train_for_each_epoch(loader)

        # Save model after training
        self.save()
        print("Training and saving completed!")

    def _train_for_each_epoch(self, train_data_loader: DataLoader):
        # for each sql, train twice
        # here the batch_size is always to 1
        num_sql_executed = 0
        for (
            batch_sql_feature,
            batch_plan_feature,
            batch_alias,
            batch_mask,
            batch_y,
        ) in train_data_loader:
            num_sql_executed += 1
            print(f"   Trainig by sample with sql number {num_sql_executed}")

            batch_sql_feature = batch_sql_feature.to(self.config.device)
            batch_mask = batch_mask.to(self.config.device)

            # 0. predict based on current model, save to KNN
            plan_time_triple = self.tree_sql_builder.predict_with_uncertainty_batch(
                tree_features=[batch_plan_feature],
                sql_feature=batch_sql_feature,
                device=self.config.device,
            )
            self.knn.insertAValue(
                (
                    plan_time_triple[0],
                    self.latency_normalizer.encode(batch_y) - plan_time_triple[0][0],
                )
            )

            # 1. train the TreeSQLNet, and add features to the memory
            loss, plan_sql_vec_mean, plan_sql_vec_variance, target_fea_model = (
                self.tree_sql_builder.train_on_sample(
                    tree_feature=batch_plan_feature,
                    sql_feature=batch_sql_feature,
                    target_value_ms=batch_y,
                    mask=batch_mask,
                )
            )
            self.experience_memory.push(
                ReplayMemory.Transition(
                    tree_feature=batch_plan_feature,
                    sql_feature=batch_sql_feature,
                    target_feature=target_fea_model,
                    mask=batch_mask,
                    weight=abs(plan_sql_vec_mean - batch_y),
                )
            )

            # 2. train the MCTSHinterSearch
            train_on_sample_res = self.mcts_searcher.train_on_sample(
                tree_feature=batch_plan_feature,
                sql_feature=batch_sql_feature,
                target_value_ms=batch_y,
                alias_set=batch_alias,
            )
            if train_on_sample_res:
                search_loss, order_feature, target_fea_search = train_on_sample_res
                self.mcts_memory.push(
                    MCTSMemory.MCTSSample(
                        batch_sql_feature, order_feature, target_fea_search
                    )
                )

            # 3. train by batch on experience history
            if num_sql_executed < 1000 or num_sql_executed % 10 == 0:
                repeat = 4 if num_sql_executed < 1000 else 1

                for _ in range(repeat):
                    # Train network
                    net_samples, samples_idx = self.experience_memory.sample(
                        self.config.batch_size
                    )
                    loss_value, new_weight = self.tree_sql_builder.train_on_batch(
                        net_samples, self.config.device
                    )
                    self.experience_memory.update_weight(samples_idx, new_weight)

                    # Train MCTS search
                    mcts_samples = self.mcts_memory.sample(self.config.batch_size)
                    self.mcts_searcher.train_on_batch(mcts_samples)

                    # Exit early if loss is acceptable
                    if loss_value <= 3:
                        break

    def sql_enhancement(self, sql: str, db_cli: PostgresConnector):
        """
        Optimize SQL query using MCTS-based join order search.

        Args:
            sql: SQL query string
            db_cli: PostgreSQL connector

        Returns:
            Tuple of (optimized_sql, join_order_hint)
            - optimized_sql: SQL with join order hint prepended (if hint is used)
            - join_order_hint: Join order hint string (e.g., "/*+Leading(...)*/") or None
        """
        try:
            sql_vec, alias, join_list_with_predicate, join_list = self.sql2vec.to_vec(
                sql
            )

            sql_feature = torch.tensor(sql_vec, dtype=torch.float32).reshape(1, -1)

            # 1. baseline prediction & hint decision
            plan_json_pg, plan_time = db_cli.explain(query=sql)
            tree_feature = self.plan_builder.plan_to_feature_tree(plan_json_pg)

            plan_time_triple = self.tree_sql_builder.predict_with_uncertainty_batch(
                tree_features=[tree_feature],
                sql_feature=sql_feature,
                device=self.config.device,
            )

            # 2. search the join-order-hint with value network.
            chosen_leading_pair = self.search_best_join_order(
                db_cli=db_cli,
                plan_json_PG=plan_json_pg,
                alias=alias,
                sql_vec=sql_vec,
                sql=sql,
                join_list_with_predicate=join_list_with_predicate,
                join_lis=join_list,
            )

            # 3. decide hint or default plan
            knn_plan = abs(self.knn.kNeightboursSample(plan_time_triple[0]))
            should_try_hint = (
                chosen_leading_pair[0][0] < plan_time_triple[0][0]
                and abs(knn_plan) < self.config.threshold
                and self.latency_normalizer.decode(plan_time_triple[0][0]) > 100
            )

            if should_try_hint:
                # timeout window from predicted hinted time (×3), capped
                max_time_out = min(
                    int(self.latency_normalizer.decode(chosen_leading_pair[0][0]) * 3),
                    self.config.max_time_out,
                )
                join_order_hint = chosen_leading_pair[1]  # e.g., "/*+Leading(...)*/"
                optimized_sql = join_order_hint + " " + sql
                hinted_plan_json, _ = db_cli.explain(query=optimized_sql)
                hinted_tree_feature = self.plan_builder.plan_to_feature_tree(
                    hinted_plan_json
                )

                hinted_plan_time_triple = (
                    self.tree_sql_builder.predict_with_uncertainty_batch(
                        tree_features=[hinted_tree_feature],
                        sql_feature=sql_feature,
                        device=self.config.device,
                    )
                )

                predicted_time_ms = (
                    self.latency_normalizer.decode(hinted_plan_time_triple[0][0])
                    * 1000.0
                )  # seconds -> ms

            else:
                predicted_time_ms = self.latency_normalizer.decode(
                    plan_time_triple[0][0]
                )
                optimized_sql = sql
                join_order_hint = None

        except Exception as e:
            print(traceback.format_exc())
            optimized_sql = sql
            join_order_hint = None

        # Return optimized SQL and join order hint (consistent with hint_plan_sel_expert interface)
        return optimized_sql, join_order_hint

    def search_best_join_order(
        self,
        db_cli: PostgresConnector,
        plan_json_PG,
        alias,
        sql_vec,
        sql,
        join_list_with_predicate,
        join_lis,
    ):
        """Run MCTS to generate join hints, predict them, and pick the best one."""
        alias_id = [self.config.aliasname2id[a] for a in alias]

        id_joins_with_predicate = [
            (self.config.aliasname2id[p[0]], self.config.aliasname2id[p[1]])
            for p in join_list_with_predicate
        ]
        id_joins = [
            (self.config.aliasname2id[p[0]], self.config.aliasname2id[p[1]])
            for p in join_lis
        ]

        leading_length = self.config.leading_length
        if leading_length == -1 or leading_length > len(alias):
            leading_length = len(alias)

        join_list_with_predicate = self.mcts_searcher.find_hints(
            40,
            len(alias),
            sql_vec,
            id_joins,
            id_joins_with_predicate,
            alias_id,
            depth=leading_length,
        )

        leading_list = []
        tree_features = []
        leadings_utility_list = []
        for join in join_list_with_predicate:
            leading_hint = (
                "/*+Leading("
                + " ".join(
                    [self.config.id2aliasname[x] for x in join[0][:leading_length]]
                )
                + ")*/"
            )
            leading_list.append(leading_hint)
            leadings_utility_list.append(join[1])
            plan_json, _ = db_cli.explain(leading_hint + sql)
            tree_feature = self.plan_builder.plan_to_feature_tree(plan_json)
            tree_features.append(tree_feature)

        sql_feature = torch.tensor(
            sql_vec, device=self.config.device, dtype=torch.float32
        ).reshape(1, -1)
        plan_times = self.tree_sql_builder.predict_with_uncertainty_batch(
            tree_features=tree_features,
            sql_feature=sql_feature,
            device=self.config.device,
        )

        chosen_leading_pair = sorted(
            zip(
                plan_times[: self.config.max_hint_num],
                leading_list,
                leadings_utility_list,
            ),
            key=lambda x: x[0][0] + self.knn.kNeightboursSample(x[0]),
        )[0]
        return chosen_leading_pair

    def save(self, path_prefix: Optional[str] = None):
        """
        Save model to directory.

        Args:
            path_prefix: Optional directory path (defaults to config.cur_model_path)
        """
        target_dir = path_prefix or self.config.cur_model_path
        os.makedirs(target_dir, exist_ok=True)

        # Save model checkpoint
        checkpoint = {
            "tree_net_state_dict": self.value_network.state_dict(),  # TreeSQLNet
            "mcts_searcher_state_dict": self.mcts_searcher.prediction_net.to(
                "cpu"
            ).state_dict(),  # ValueNet
        }
        checkpoint_path = os.path.join(target_dir, "model.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save KNN
        knn_path = os.path.join(target_dir, "knn.pkl")
        with open(knn_path, "wb") as f:
            pickle.dump(self.knn, f)

        print(f"[save] {checkpoint_path}")
        print(f"[save] {knn_path}")
        return checkpoint_path, knn_path

    def load(self, model_dir: Optional[str] = None):
        """
        Load model from directory.

        Args:
            model_dir: Optional directory path (defaults to config.cur_model_path)
        """
        model_dir = model_dir or self.config.cur_model_path

        # Load KNN
        knn_path = os.path.join(model_dir, "knn.pkl")
        if not os.path.exists(knn_path):
            raise FileNotFoundError(f"KNN file not found: {knn_path}")
        with open(knn_path, "rb") as f:
            self.knn = pickle.load(f)

        # Load model checkpoint
        checkpoint_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.config.device)

        # TreeSQLNet & ValueNet
        self.value_network.load_state_dict(ckpt["tree_net_state_dict"])

        # Load ValueNet with compatibility for hybrid_qo's state dict keys
        # hybrid_qo uses "layer1" but our code uses "query_proj"
        # hybrid_qo has "rnn" but it's not used (commented out in forward), so we ignore it
        hybrid_qo_state_dict = ckpt["mcts_searcher_state_dict"]
        our_state_dict = {}

        for key, value in hybrid_qo_state_dict.items():
            # Map "layer1" to "query_proj"
            if key.startswith("layer1."):
                new_key = key.replace("layer1.", "query_proj.")
                our_state_dict[new_key] = value
            # Ignore "rnn" keys (not used in hybrid_qo and not in our model)
            elif key.startswith("rnn."):
                continue  # Skip rnn keys
            # Keep other keys as-is
            else:
                our_state_dict[key] = value

        self.mcts_searcher.prediction_net.load_state_dict(our_state_dict, strict=False)

        print(f"[load] {checkpoint_path}")
        print(f"[load] {knn_path}")
        return checkpoint_path, knn_path

    def _get_data_loader(self, training_data: List[PlanRecord]):
        """
        Load PlanRecords and convert to training format.

        Args:
            training_data: List of PlanRecord objects
        """

        # 1. preprocessing
        sql_plan = []
        exe_time = []
        for rec in training_data:
            try:
                sql_str = rec.query
                # actual_plan_json is already a dict, not JSON string
                plan_json = rec.actual_plan_json
                actual_latency = min(
                    rec.actual_latency, float(self.config.max_time_out)
                )

                actual_latency = self.latency_normalizer.encode(actual_latency)
                sql_vec, alias, _, _ = self.sql2vec.to_vec(sql_str)
                mask = (torch.rand(1, self.config.head_num) < 0.9).long()
                sql_feature = torch.tensor(sql_vec, dtype=torch.float32).reshape(1, -1)

                tree_feature = self.plan_builder.plan_to_feature_tree(plan_json)

                sql_plan.append([sql_feature, tree_feature, alias, mask])
                exe_time.append(actual_latency)
            except Exception as e:
                print(f"get has problenm: {e}")
                continue

        # Dataset / DataLoader
        dataset = MCTSDataset(sql_plan, exe_time)
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            pin_memory=(self.config.device.type == "cuda"),
            num_workers=0,
            collate_fn=self._identity_collate,
        )

        return loader

    @staticmethod
    def _identity_collate(batch):
        # currently, it is not support batch
        return batch[0]


if __name__ == "__main__":
    import argparse

    from exp_buffer.buffer_mngr import read_sql_files

    def _train(dbname: str, db_path: str, config: MCTSConfig):

        buffer_mngr = BufferManager(db_path)

        # Create and train expert
        with PostgresConnector(dbname) as conn:
            expert = MCTSOptimizerExpert(
                config=config, buffer_mngr=buffer_mngr, db_cli=conn
            )
            expert.train_and_save()
            print("Training completed!")
            print("Save expert training completed successfully!")

    def _predict(db_path: str, input_sql_dir: str, dbname: str, config: MCTSConfig):
        buffer_mngr = BufferManager(db_path)

        query_w_name = read_sql_files(input_sql_dir)
        with PostgresConnector(dbname) as conn:
            expert = MCTSOptimizerExpert(
                config=config, buffer_mngr=buffer_mngr, db_cli=conn
            )
            expert.load()
            for sql_name, sql in query_w_name:
                optimized_sql, join_order_hint = expert.sql_enhancement(
                    sql=sql, db_cli=conn
                )
                print(f"join_order_hint = {join_order_hint}")
                # Execute the sql with hint, save
                execution_id = buffer_mngr.run_log_query_exec(
                    conn=conn, sql=sql, hints=None, join_order_hint=join_order_hint
                )

                print(f"✓ Saved execution {execution_id} for {sql_name}: {sql[:60]}...")
                print(execution_id)

    parser = argparse.ArgumentParser(description="Train Expert from Unified Data")
    parser.add_argument("--dbname", type=str, default="imdb_ori", help="Database name")
    parser.add_argument(
        "--input_sql_dir",
        type=str,
        default="/Users/kevin/project_python/AI4QueryOptimizer/lqo_benchmark/workloads/bao/join_unique",
        help="Input SQL dir (one query per file)",
    )
    args = parser.parse_args()

    # ---------- JOB ----------
    config_job = MCTSConfig(
        max_alias_num=40,
        max_column=100,
        id2aliasname={
            0: "start",
            1: "chn",
            2: "ci",
            3: "cn",
            4: "ct",
            5: "mc",
            6: "rt",
            7: "t",
            8: "k",
            9: "lt",
            10: "mk",
            11: "ml",
            12: "it1",
            13: "it2",
            14: "mi",
            15: "mi_idx",
            16: "it",
            17: "kt",
            18: "miidx",
            19: "at",
            20: "an",
            21: "n",
            22: "cc",
            23: "cct1",
            24: "cct2",
            25: "it3",
            26: "pi",
            27: "t1",
            28: "t2",
            29: "cn1",
            30: "cn2",
            31: "kt1",
            32: "kt2",
            33: "mc1",
            34: "mc2",
            35: "mi_idx1",
            36: "mi_idx2",
            37: "an1",
            38: "n1",
            39: "a1",
        },
        aliasname2id={
            "kt1": 31,
            "chn": 1,
            "cn1": 29,
            "mi_idx2": 36,
            "cct1": 23,
            "n": 21,
            "a1": 39,
            "kt2": 32,
            "miidx": 18,
            "it": 16,
            "mi_idx1": 35,
            "kt": 17,
            "lt": 9,
            "ci": 2,
            "t": 7,
            "k": 8,
            "start": 0,
            "ml": 11,
            "ct": 4,
            "t2": 28,
            "rt": 6,
            "it2": 13,
            "an1": 37,
            "at": 19,
            "mc2": 34,
            "pi": 26,
            "mc": 5,
            "mi_idx": 15,
            "n1": 38,
            "cn2": 30,
            "mi": 14,
            "it1": 12,
            "cc": 22,
            "cct2": 24,
            "an": 20,
            "mk": 10,
            "cn": 3,
            "it3": 25,
            "t1": 27,
            "mc1": 33,
        },
    )

    # ---------- STACK ----------
    config_stack = MCTSConfig(
        max_alias_num=29,
        max_column=66,
        id2aliasname={
            0: "start",
            1: "a1",
            2: "acc",
            3: "account",
            4: "b",
            5: "b1",
            6: "b2",
            7: "c1",
            8: "c2",
            9: "pl",
            10: "q",
            11: "q1",
            12: "q2",
            13: "question",
            14: "s",
            15: "s1",
            16: "s2",
            17: "site",
            18: "so_user",
            19: "t",
            20: "t1",
            21: "t2",
            22: "tag",
            23: "tag_question",
            24: "tq",
            25: "tq1",
            26: "tq2",
            27: "u1",
            28: "u2",
        },
        aliasname2id={
            "start": 0,
            "a1": 1,
            "acc": 2,
            "account": 3,
            "b": 4,
            "b1": 5,
            "b2": 6,
            "c1": 7,
            "c2": 8,
            "pl": 9,
            "q": 10,
            "q1": 11,
            "q2": 12,
            "question": 13,
            "s": 14,
            "s1": 15,
            "s2": 16,
            "site": 17,
            "so_user": 18,
            "t": 19,
            "t1": 20,
            "t2": 21,
            "tag": 22,
            "tag_question": 23,
            "tq": 24,
            "tq1": 25,
            "tq2": 26,
            "u1": 27,
            "u2": 28,
        },
    )

    # _train(dbname=args.dbname, db_path=f"buffer_{args.dbname}.db", config=config_job)
    _predict(
        f"buffer_{args.dbname}.db", args.input_sql_dir, args.dbname, config=config_job
    )
