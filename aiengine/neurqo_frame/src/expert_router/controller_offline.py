# Standard library imports
import copy
import os
import random
from typing import List, Tuple

# Third-party imports
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local/project imports
from common import BaseConfig
from parser.table_parser import load_db_info_json
from utils.io import save_per_train_plot_data

from .encoder import Sql2VecEmbeddingV2
from .loss import HybridLossReg
from .logger import plogger
from .model import QueryOptMHSASuperNode


class ModelBuilder:
    def __init__(self, num_columns, output_dim, model_path_prefix, embedding_path, num_heads,
                 embedding_dim, is_fix_emb, num_layers, dataset, cfg: BaseConfig,
                 num_tables=None):

        db_profile_res = load_db_info_json(cfg.DB_INFO_DICT)
        num_tables = len(set(db_profile_res.table_no_map.values()))

        self.output_dim = output_dim
        self.dataset = dataset
        self.cfg = cfg
        self.embedding_path = embedding_path
        self.model = QueryOptMHSASuperNode(
            num_tables=num_tables,
            num_columns=num_columns,
            output_dim=output_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            embedding_path=self.embedding_path,
            is_fix_emb=is_fix_emb,
            num_layers=num_layers,
            dataset=dataset,
            cfg=cfg
        ).to(self.cfg.DEVICE)

        # self.saved_model_name = model_path_prefix + f"_mlp_multi_head_label_embed_atten4_time_best_{self.dataset}.pth"
        self.saved_model_name = model_path_prefix
        self.exploration_rate = 0.1

        # Initialize TensorBoard writer
        parent_directory_os = os.path.dirname(self.saved_model_name)
        log_dir = os.path.join(parent_directory_os, "tensorboard_logs")
        self.writer = SummaryWriter(log_dir=log_dir)

        self.sql_encoder = Sql2VecEmbeddingV2(db_profile_res=db_profile_res,
                                              checkpoint_file=cfg.EMBED_FILE)

        self.strategies = None

    def save_model(self, epoch, evl_method: str):
        """Save the model weights."""
        torch.save(self.model.state_dict(), f"{self.saved_model_name}_{evl_method}")
        plogger.info(f"Model saved to {self.saved_model_name}_{evl_method}, epoch={epoch}")

    def load_model(self, evl_method: str):
        """Load the model weights."""
        # self.model.load_state_dict(torch.load(f"{self.saved_model_name}_{evl_method}", map_location=self.cfg.DEVICE))
        self.model.load_state_dict(torch.load(self.saved_model_name, map_location=self.cfg.DEVICE))
        self.model.to(self.cfg.DEVICE)
        print(f"Model loaded from {self.saved_model_name}")
        plogger.info(f"Model loaded from {self.saved_model_name}")

    def train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch_multi, y_times, query_id in loader:
            X_batch, y_batch_multi, y_times = X_batch, y_batch_multi.to(self.cfg.DEVICE), y_times.to(self.cfg.DEVICE)
            optimizer.zero_grad()
            class_logits, reg_times, _ = self.model(X_batch)
            loss = criterion(class_logits, reg_times, y_batch_multi, y_times)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _evaluate_strategy_batch(self, probs, reg_times, y_times, y_batch_multi, q_batch_ids, strategy: str):
        w = 0.7
        k = 3
        """Helper to evaluate a single strategy."""
        correct_any = 0
        total_time = 0
        best_action_set = {}

        EXECUTION_TIME_OUT_tensor = torch.tensor(self.cfg.EXECUTION_TIME_OUT, device=probs.device)

        for i, (prob_row, reg_row, y_time_row, q_id) in enumerate(zip(probs, reg_times, y_times, q_batch_ids)):
            # Ground-truth correct indices
            correct_index = [idx for idx, val in enumerate(y_batch_multi[i]) if val == 1]

            # Pick predicted index based on strategy
            if strategy == "class_only":
                pred_idx = torch.argmax(prob_row)  # Highest probability
            elif strategy == "reg_only":
                pred_idx = torch.argmin(reg_row)  # Lowest predicted time
            elif strategy == "hypered_1":  # Hard margin (prob > 0.5, then min time)
                valid_mask = prob_row > 0.5
                if valid_mask.sum() == 0:
                    pred_idx = torch.argmin(reg_row)
                else:
                    valid_times = reg_row.masked_fill(~valid_mask, float('inf'))
                    pred_idx = torch.argmin(valid_times)
            elif strategy == "hypered_2":  # Our usage
                norm_times = reg_row / EXECUTION_TIME_OUT_tensor
                score = prob_row - norm_times
                pred_idx = torch.argmax(score)
            elif strategy == "weighted_score":
                norm_times = reg_row / EXECUTION_TIME_OUT_tensor
                pred_idx = torch.argmax(w * prob_row - (1 - w) * norm_times)
            elif strategy == "topk_class":
                topk_indices = torch.topk(prob_row, k).indices
                pred_idx = topk_indices[torch.argmin(reg_row[topk_indices])]
            elif strategy == "thresh_reg":
                threshold = reg_row.median()
                valid_mask = reg_row < threshold
                pred_idx = torch.argmax(prob_row.masked_fill(~valid_mask, float('-inf')))
            else:
                raise ValueError(f"Strategy '{strategy}' is not supported")

            if int(pred_idx) in correct_index:
                correct_any += 1
            total_time += y_time_row[pred_idx].item()
            best_action_set[q_id] = pred_idx.item()

        return correct_any, total_time, best_action_set

    def evaluate_metric_batch(self, probs, reg_times, y_times, q_batch_ids, y_batch_multi):
        results = {
            "correct_any": {},
            "total_time": {},
            "best_action_set": {}
        }

        for strategy in self.strategies:
            correct_any, total_time, best_action_set = self._evaluate_strategy_batch(
                probs, reg_times, y_times, y_batch_multi, q_batch_ids, strategy
            )
            results["correct_any"][strategy] = correct_any
            results["total_time"][strategy] = total_time
            results["best_action_set"][strategy] = best_action_set

        return results["correct_any"], results["best_action_set"], results["total_time"]

    def evaluate_topk(self, loader):

        self.model.eval()

        total_time_all_test_query = {strategy: 0 for strategy in self.strategies}
        best_action_set = {strategy: {} for strategy in self.strategies}
        correct_any_map = {strategy: 0 for strategy in self.strategies}
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch_multi, y_times, q_batch_ids in loader:
                y_batch_multi = y_batch_multi.to(self.cfg.DEVICE)
                X_batch, y_times = X_batch, y_times.to(self.cfg.DEVICE)

                class_logits, reg_times, _ = self.model(X_batch)
                probs = torch.sigmoid(class_logits)

                # Evaluate all strategies for this batch
                correct_any_map_cur, best_action_set_cur, total_time_cur = self.evaluate_metric_batch(
                    probs, reg_times, y_times, q_batch_ids, y_batch_multi
                )

                # Accumulate results
                for strategy in self.strategies:
                    correct_any_map[strategy] += correct_any_map_cur[strategy]
                    total_time_all_test_query[strategy] += total_time_cur[strategy]
                    best_action_set[strategy].update(best_action_set_cur[strategy])

                total_samples += len(q_batch_ids)

        # Compute accuracies
        accuracy_map = {strategy: val / total_samples for strategy, val in correct_any_map.items()}
        return accuracy_map, best_action_set, total_time_all_test_query

    def run(self, train_loader, test_loaders: dict, epochs, lr, step_size, gamma, alpha, loss_gamma, workload_name):
        class_weights = train_loader.dataset.get_class_weights().to(
            self.cfg.DEVICE)  # Assume this method exists in your dataset

        self.strategies = ["class_only", "reg_only", "hypered_1", "hypered_2",
                           "weighted_score", "topk_class", "thresh_reg"]

        criterion = HybridLossReg(cfg=self.cfg, alpha=alpha, gamma=loss_gamma, class_weights=class_weights)

        if self.embedding_path:
            optimizer = optim.Adam([
                {"params": self.model.embedding_model.parameters(), "lr": 5e-5},
                {"params": self.model.shared.parameters(), "lr": lr},
                {"params": self.model.class_head.parameters(), "lr": lr},
                {"params": self.model.reg_head.parameters(), "lr": lr}
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler now takes step_size and gamma from the arguments
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # In run():
        global_best_time = {strategy: float('inf') for strategy in self.strategies}
        final_log_to_print = {strategy: "" for strategy in self.strategies}
        running_history = []
        for epoch in tqdm(range(epochs), desc=f"Training {workload_name}"):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)

            # Step the learning rate scheduler
            lr_scheduler.step()

            # Log training loss to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)

            accuracy_map, best_action_set, total_time_all_test_query = self.evaluate_topk(test_loaders["test"])

            # Log each epoch
            plogger.info(f"Epoch {epoch + 1}/{epochs}, Info: ")
            plogger.info(f"Avg Test Acc: {accuracy_map}")
            plogger.info(f"best_action_set = {best_action_set}")
            plogger.info(f"Total Sum Time = {total_time_all_test_query} \n")

            running_history.append(
                {
                    "Workload": train_loader.dataset.get_current_worklaod(),
                    "Epoch": epoch,
                    "AccuracyMap": accuracy_map,
                    "best_action_set": best_action_set,
                    "total_time_all_test_query": total_time_all_test_query,
                }
            )

            # print best epoch
            for strategy in self.strategies:
                if total_time_all_test_query[strategy] <= global_best_time[strategy]:
                    global_best_time[strategy] = total_time_all_test_query[strategy]
                    self.save_model(epoch, strategy)
                    final_log_to_print[strategy] = (
                        f"Workload: {train_loader.dataset.get_current_worklaod()}, "
                        f"Epoch: {epoch:03d}, "
                        f"Test Acc: {accuracy_map[strategy]:.2f}, "
                        f"Sum Time: {total_time_all_test_query[strategy] / 1000:.6f}, "
                        f"Predicted: {best_action_set[strategy]}"
                    )

        self.writer.close()
        return running_history, final_log_to_print, ""

    def inference(self, loader, save_embedding=True, save_name=""):
        """Perform inference using the 'hyper_2' strategy to select optimizers."""
        self.model.eval()  # Set model to evaluation mode
        predictions_dict = {}  # Store query label to optimizer predictions
        predictions_time = 0  # Store query label to optimizer predictions

        total_reward = 0.0  # Sum of negative execution times (rewards)
        total_regret = 0.0  # Sum of regret compared to optimal
        query_count = 0  # Total number of queries processed

        embeddings, latencies, query_ids = [], [], []
        EXECUTION_TIME_OUT_tensor = torch.tensor(self.cfg.EXECUTION_TIME_OUT, device=self.cfg.DEVICE)

        with torch.no_grad():  # Disable gradient computation for inference
            for X_batch, _, y_execution_times, q_batch_ids in loader:
                # Move data to the appropriate device
                y_execution_times = y_execution_times.to(self.cfg.DEVICE)

                batch_size = y_execution_times.size(0)
                query_count += batch_size

                # Get model predictions: class logits and regression times
                class_logits, reg_times, emb_output = self.model(X_batch)

                if save_embedding:
                    emb_output = emb_output.cpu().numpy()
                    embeddings.append(emb_output)
                    # use the PG'a latency as the label for embedding
                    y_target = y_execution_times[:, 3].unsqueeze(-1).cpu().numpy()
                    latencies.append(y_target)
                    query_ids.extend(q_batch_ids)

                probs = torch.sigmoid(class_logits)  # Convert logits to probabilities

                actions = torch.zeros(batch_size, dtype=torch.long, device=self.cfg.DEVICE)

                for i, (prob_row, reg_row, y_time_row, q_id) in enumerate(
                        zip(probs, reg_times, y_execution_times, q_batch_ids)):
                    # Select optimizers using 'hyper_2' strategy
                    # valid_mask = prob_row > 0.5
                    # if valid_mask.sum() == 0:
                    #     pred_idx = torch.argmin(reg_row)
                    # else:
                    #     valid_times = reg_row.masked_fill(~valid_mask, float('inf'))
                    #     pred_idx = torch.argmin(valid_times)

                    norm_times = reg_row / EXECUTION_TIME_OUT_tensor
                    score = prob_row - norm_times
                    pred_idx = torch.argmax(score)

                    predictions_dict[q_id] = pred_idx.item()
                    predictions_time += y_time_row[pred_idx].item()
                    actions[i] = pred_idx.item()

                # Compute rewards (negative execution times) for selected actions
                rewards = -y_execution_times.gather(1, actions.unsqueeze(1)).squeeze()
                total_reward += rewards.sum().item()

                # Compute regret: difference from optimal execution time
                optimal_rewards = -y_execution_times.min(dim=1)[0]
                batch_regret = (optimal_rewards - rewards).sum().item()
                total_regret += batch_regret

                # Log batch-level information
                plogger.info(f"Batch processed, Actions: {actions.tolist()}, "
                             f"Rewards: {rewards.tolist()}")

        # Compute average metrics
        avg_reward = total_reward / query_count if query_count > 0 else 0
        avg_regret = total_regret / query_count if query_count > 0 else 0

        # Log final inference results
        plogger.info(f"Total Reward: {total_reward:.4f}, Avg Reward: {avg_reward:.4f}, "
                     f"Total Regret: {total_regret:.4f}, Avg Regret: {avg_regret:.4f}, "
                     f"Queries: {query_count}")

        save_per_train_plot_data(embeddings, latencies, [], query_ids,
                                 output_file=f"{self.cfg.RESULT_DATA_BASE}/inference_res_collection_{self.dataset}{save_name}.data")

        return predictions_dict, predictions_time

    def inference_single_sql(self, sql: str, query_id: str, db_cli) -> List[Tuple[int, float]]:
        """
        Perform inference on a single SQL query without using DataLoader.
        This follows the same preprocessing as QueryFeatureDataset to ensure consistency.
        
        Args:
            sql: SQL query string to optimize
            query_id: Unique identifier for the query
            db_cli: Database connection (PostgresConnector)

        Returns:
            List[Tuple[int, float]]: Sorted list of (optimizer_index, score) tuples, 
                                     ordered by score descending (highest first)
        """
        self.model.eval()  # Set model to evaluation mode

        # Encode SQL query to get features (returns raw features with IDs starting from 0)
        sql_feature = self.sql_encoder.encode_query(
            query_id=query_id, sql=sql, train_database=self.cfg.DB_NAME, pg_runner=db_cli)

        # Apply global shift (+1) as done in QueryFeatureDataset
        # This shifts table_ids and col_ids to avoid 0 padding conflicts
        sql_feature_shifted = copy.deepcopy(sql_feature)

        # Shift join_conditions: table1_id, col1_id, table2_id, col2_id
        if len(sql_feature_shifted['join_conditions']) > 0:
            sql_feature_shifted['join_conditions'][:, 0] += 1  # table1_id
            sql_feature_shifted['join_conditions'][:, 1] += 1  # col1_id
            sql_feature_shifted['join_conditions'][:, 2] += 1  # table2_id
            sql_feature_shifted['join_conditions'][:, 3] += 1  # col2_id

        # Shift filter_conditions: table_id, col_id, ...
        if len(sql_feature_shifted['filter_conditions']) > 0:
            sql_feature_shifted['filter_conditions'][:, 0] += 1  # table_id
            sql_feature_shifted['filter_conditions'][:, 1] += 1  # col_id

        # Convert to tensors and add batch dimension (batch_size=1)
        # Ensure correct shape for empty arrays (should be (0, 4) or (0, 3), not (0,))
        join_conditions_array = sql_feature_shifted['join_conditions']
        filter_conditions_array = sql_feature_shifted['filter_conditions']

        if join_conditions_array.size == 0:
            join_conditions_array = join_conditions_array.reshape(0, 4)
        if filter_conditions_array.size == 0:
            filter_conditions_array = filter_conditions_array.reshape(0, 3)

        join_conditions = torch.from_numpy(join_conditions_array).float()
        filter_conditions = torch.from_numpy(filter_conditions_array).float()
        table_sizes = torch.from_numpy(sql_feature_shifted['table_sizes']).float()

        # No padding needed for single query, but we still need batch dimension
        # Shape: [1, num_joins, 4] for join_conditions (even if num_joins=0)
        # Shape: [1, num_filters, 3] for filter_conditions (even if num_filters=0)
        # Shape: [1, num_tables] for table_sizes
        join_conditions = join_conditions.unsqueeze(0)  # [1, num_joins, 4]
        filter_conditions = filter_conditions.unsqueeze(0)  # [1, num_filters, 3]
        table_sizes = table_sizes.unsqueeze(0)  # [1, num_tables]

        # Format as batch (single sample)
        X_batch = {
            'join_conditions': join_conditions.to(self.cfg.DEVICE),
            'filter_conditions': filter_conditions.to(self.cfg.DEVICE),
            'table_sizes': table_sizes.to(self.cfg.DEVICE)
        }

        # Perform inference
        with torch.no_grad():
            class_logits, reg_times, _ = self.model(X_batch)

        # Apply 'hyper_2' strategy to select optimizer
        probs = torch.sigmoid(class_logits[0])  # Remove batch dimension: [output_dim]
        reg_row = reg_times[0]  # Remove batch dimension: [output_dim]

        # Use 'hyper_2' strategy: score = prob - (normalized_time)
        EXECUTION_TIME_OUT_tensor = torch.tensor(self.cfg.EXECUTION_TIME_OUT, device=self.cfg.DEVICE)
        norm_times = reg_row / EXECUTION_TIME_OUT_tensor
        score = probs - norm_times

        # Sort by score descending (highest score first)
        sorted_indices = torch.argsort(score, descending=True)
        sorted_scores = score[sorted_indices]

        # Return list of (optimizer_index, score) tuples
        result = [(idx.item(), score_val.item()) for idx, score_val in zip(sorted_indices, sorted_scores)]

        return result

    def reconfigure_experts(self, option_experts, cur_active_experts):

        # check remoable ones
        removable_experts = list(set(option_experts) & set(cur_active_experts))

        # check addable ones
        addable_experts = [ele for ele in option_experts if ele not in cur_active_experts]

        if len(addable_experts) > 0 and len(removable_experts) > 0:
            if random.random() < 0.5:  # add
                expert_to_add = random.choice(addable_experts)
                cur_active_experts.append(expert_to_add)
                print(f"------> add {expert_to_add}, current {cur_active_experts}")
            else:  # remove
                expert_to_remove = random.choice(removable_experts)
                cur_active_experts.remove(expert_to_remove)
                print(f"------> remove {expert_to_remove}, current {cur_active_experts}")
            return cur_active_experts

        elif len(addable_experts) > 0:  # add
            expert_to_add = random.choice(addable_experts)
            cur_active_experts.append(expert_to_add)
            print(f"------> add {expert_to_add}, current {cur_active_experts}")
            return cur_active_experts
        elif len(removable_experts) > 0:  # remove
            expert_to_remove = random.choice(removable_experts)
            cur_active_experts.remove(expert_to_remove)
            print(f"------> remove {expert_to_remove}, current {cur_active_experts}")
            return cur_active_experts
        else:
            return cur_active_experts

    def inference_expert_reconfig(self, loader, option_experts, num_runs=20):
        """
        Args:
            loader: Original DataLoader with test data (batch_size should be 1)
            option_experts: List of experts to chose
            num_runs: Number of times to repeat the inference with shuffling
        """
        import random

        self.model.eval()  # Set model to evaluation mode

        all_results = []  # Store results from each run

        # Assuming ALL_METHODS is defined globally or as a class attribute
        postgresql_idx = self.cfg.ALL_METHODS.index('PostgreSQL')  # Index for PostgreSQL
        EXECUTION_TIME_OUT_tensor = torch.tensor(self.cfg.EXECUTION_TIME_OUT, device=self.cfg.DEVICE)
        global_inf_id = 0
        for run in range(num_runs):
            print("next run --------------------> \n")

            predictions_dict = {}  # Store query label to optimizer predictions
            predictions_time = 0  # Total time for system predictions
            postgresql_time = 0  # Total time for PostgreSQL
            query_count = 0  # Total number of queries processed
            cumulative_system_times = []  # List for cumulative system times
            cumulative_postgresql_times = []  # List for cumulative PostgreSQL times
            current_system_total = 0  # Running total for system
            current_postgresql_total = 0  # Running total for PostgreSQL

            differ = 0

            active_experts = self.cfg.ALL_METHODS.copy()
            with torch.no_grad():  # Disable gradient computation for inference
                # Manually iterate over shuffled samples
                for X_batch, _, y_execution_times, q_batch_ids in loader:

                    # Reconfigure experts for 4 samples
                    if global_inf_id % 4 == 0:
                        active_experts = self.reconfigure_experts(option_experts, active_experts)
                    global_inf_id += 1

                    query_count += 1  # Since batch_size=1, each iteration is 1 query

                    # Get model predictions: class logits and regression times
                    class_logits, reg_times, emb_output = self.model(X_batch)

                    probs = torch.sigmoid(class_logits)  # Convert logits to probabilities

                    # Since batch_size=1, we only have one row to process
                    prob_row = probs[0]
                    reg_row = reg_times[0]
                    y_time_row = y_execution_times[0]
                    q_id = q_batch_ids[0]

                    print(f"all methos is {self.cfg.ALL_METHODS}, current methods are {active_experts}")
                    active_indices = [self.cfg.ALL_METHODS.index(exp) for exp in active_experts]

                    # Select optimizers using 'hyper_1' strategy
                    norm_times = reg_row / EXECUTION_TIME_OUT_tensor
                    score = prob_row - norm_times
                    original_pred_idx = torch.argmax(score).item()
                    predictions_dict[q_id] = original_pred_idx

                    # Update times
                    system_time = y_time_row[original_pred_idx].item()
                    postgresql_time_sample = y_time_row[postgresql_idx].item()

                    current_system_total += system_time
                    current_postgresql_total += postgresql_time_sample

                    # Append to cumulative lists
                    cumulative_system_times.append(system_time)
                    cumulative_postgresql_times.append(postgresql_time_sample)

                    predictions_time += system_time
                    postgresql_time += postgresql_time_sample

                    differ += postgresql_time_sample - system_time

                print("=====? differ is", differ - (predictions_time - postgresql_time))

            # Store results for this run
            run_result = {
                'run': run + 1,
                'predictions': predictions_dict,
                'total_time': predictions_time,
                'postgresql_total_time': postgresql_time,
                'cumulative_system_times': cumulative_system_times,
                'cumulative_postgresql_times': cumulative_postgresql_times,
                'query_count': query_count
            }
            all_results.append(run_result)

        return all_results, predictions_dict

    def inference_expert_random(self, loader, num_runs=20):
        self.model.eval()  # Set model to evaluation mode

        all_results = []  # Store results from each run
        EXECUTION_TIME_OUT_tensor = torch.tensor(self.cfg.EXECUTION_TIME_OUT, device=self.cfg.DEVICE)
        for run in range(num_runs):
            print("next run --------------------> \n")

            predictions_dict = {}  # Store query label to optimizer predictions (hypered_1)
            predictions_dict_random = {}  # Store random predictions separately
            predictions_time = 0  # Total time for hypered_1 predictions
            random_time = 0  # Total time for random predictions
            query_count = 0  # Total number of queries processed
            cumulative_system_times = []  # List for cumulative hypered_1 times
            cumulative_random_times = []  # List for cumulative random times

            active_experts = self.cfg.ALL_METHODS.copy()
            with torch.no_grad():  # Disable gradient computation for inference
                for X_batch, _, y_execution_times, q_batch_ids in loader:
                    query_count += 1  # Since batch_size=1, each iteration is 1 query

                    # Get model predictions: class logits and regression times
                    class_logits, reg_times, emb_output = self.model(X_batch)

                    probs = torch.sigmoid(class_logits)  # Convert logits to probabilities

                    # Since batch_size=1, we only have one row to process
                    prob_row = probs[0]
                    reg_row = reg_times[0]
                    y_time_row = y_execution_times[0]
                    q_id = q_batch_ids[0]

                    active_indices = [self.cfg.ALL_METHODS.index(exp) for exp in active_experts]

                    # Filter probabilities and regression times for active experts only
                    active_probs = prob_row[active_indices]
                    active_reg_times = reg_row[active_indices]

                    # Original hypered_2 strategy
                    norm_times = reg_row / EXECUTION_TIME_OUT_tensor
                    score = prob_row - norm_times
                    pred_idx = torch.argmax(score)
                    original_pred_idx = active_indices[pred_idx.item()]
                    predictions_dict[q_id] = original_pred_idx

                    # Random expert selection
                    pred_idx_random = torch.randint(0, len(active_indices), (1,)).item()
                    random_pred_idx = active_indices[pred_idx_random]
                    predictions_dict_random[q_id] = random_pred_idx

                    # Update times for both hypered_1 and random
                    system_time = y_time_row[original_pred_idx].item()  # hypered_1 time
                    random_system_time = y_time_row[random_pred_idx].item()  # random time

                    cumulative_system_times.append(system_time)
                    cumulative_random_times.append(random_system_time)

                    predictions_time += system_time
                    random_time += random_system_time

                # Store results for this run
                run_result = {
                    'run': run + 1,
                    'predictions_hypered_1': predictions_dict,  # hypered_1 predictions
                    'predictions_random': predictions_dict_random,  # Random predictions
                    'total_time_hypered_1': predictions_time,  # Total time for hypered_1
                    'total_time_random': random_time,  # Total time for random
                    'cumulative_system_times': cumulative_system_times,  # Times for hypered_1
                    'cumulative_random_times': cumulative_random_times,  # Times for random
                    'query_count': query_count
                }
                all_results.append(run_result)

        return all_results, predictions_dict, predictions_dict_random

    def run_with_varios_alpha(self, train_loader, test_loaders: dict, epochs, lr, step_size, gamma, alpha, loss_gamma,
                              workload_name):
        # todo, this is only to test the labmda_effects
        self.saved_model_name = f"{self.saved_model_name}_{alpha}"
        # update model path name
        class_weights = train_loader.dataset.get_class_weights().to(
            self.cfg.DEVICE)  # Assume this method exists in your dataset
        self.strategies = ["hypered_2"]
        criterion = HybridLossReg(alpha=alpha, gamma=loss_gamma, class_weights=class_weights)
        if self.embedding_path:
            optimizer = optim.Adam([
                {"params": self.model.embedding_model.parameters(), "lr": 5e-5},
                {"params": self.model.shared.parameters(), "lr": lr},
                {"params": self.model.class_head.parameters(), "lr": lr},
                {"params": self.model.reg_head.parameters(), "lr": lr}
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler now takes step_size and gamma from the arguments
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # In run():
        global_best_time = {strategy: float('inf') for strategy in self.strategies}
        final_log_to_print = {strategy: "" for strategy in self.strategies}

        running_history = []

        for epoch in tqdm(range(epochs), desc=f"Training {workload_name}"):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)

            # Step the learning rate scheduler
            lr_scheduler.step()

            # Log training loss to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)

            accuracy_map, best_action_set, total_time_all_test_query = self.evaluate_topk(test_loaders["test"])

            # Log each epoch
            plogger.info(f"Epoch {epoch + 1}/{epochs}, Info: ")
            plogger.info(f"Avg Test Acc: {accuracy_map}")
            plogger.info(f"best_action_set = {best_action_set}")
            plogger.info(f"Total Sum Time = {total_time_all_test_query} \n")
            running_history.append(
                {
                    "Workload": train_loader.dataset.get_current_worklaod(),
                    "Epoch": epoch,
                    "AccuracyMap": accuracy_map,
                    "best_action_set": best_action_set,
                    "total_time_all_test_query": total_time_all_test_query,
                }
            )

            # print best epoch
            for strategy in self.strategies:
                if total_time_all_test_query[strategy] <= global_best_time[strategy]:
                    global_best_time[strategy] = total_time_all_test_query[strategy]
                    self.save_model(epoch, strategy)
                    final_log_to_print[strategy] = (
                        f"Workload: {train_loader.dataset.get_current_worklaod()}, "
                        f"Epoch: {epoch:03d}, "
                        f"Test Acc: {accuracy_map[strategy]:.2f}, "
                        f"Sum Time: {total_time_all_test_query[strategy] / 1000:.6f}, "
                        f"Predicted: {best_action_set[strategy]}"
                    )

        self.writer.close()
        return running_history, final_log_to_print, ""
