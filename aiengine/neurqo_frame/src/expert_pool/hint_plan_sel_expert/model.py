import json
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import field
import numpy as np
import joblib
from expert_pool.hint_plan_sel_expert.featurize import TreeFeaturizer
from expert_pool.hint_plan_sel_expert.tree_cnn import TreeCNN
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
from db.pg_conn import PostgresConnector
from typing import List, Optional, Tuple
from exp_buffer.sqllite import PlanRecord
from dataclasses import dataclass
import os
import shutil
from exp_buffer.buffer_mngr import BufferManager


# Bao-compatible function for pipeline inverse transform
def _inv_log1p(x):
    """Inverse of log1p: exp(x) - 1 (Bao-compatible)."""
    return np.exp(x) - 1


# Bao-compatible file path helpers
def _nn_path(base):
    return os.path.join(base, "nn_weights")


def _x_transform_path(base):
    return os.path.join(base, "x_transform")


def _y_transform_path(base):
    return os.path.join(base, "y_transform")


def _channels_path(base):
    return os.path.join(base, "channels")


def _n_path(base):
    return os.path.join(base, "n")


def _arm_idx_to_hints(arm_idx: int) -> List[str]:
    """
    Convert arm index to list of hint SQL statements.

    Args:
        arm_idx: Index from 0 to 4

    Returns:
        List of hint SQL statements
    """
    _ALL_OPTIONS = [
        "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
        "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"
    ]

    # Start by turning all off
    hints = [f"SET {option} TO off" for option in _ALL_OPTIONS]

    # Then turn specific ones on based on arm_idx
    if arm_idx == 0:
        # All on
        hints.extend([f"SET {option} TO on" for option in _ALL_OPTIONS])
    elif arm_idx == 1:
        hints.extend([
            "SET enable_hashjoin TO on",
            "SET enable_indexonlyscan TO on",
            "SET enable_indexscan TO on",
            "SET enable_mergejoin TO on",
            "SET enable_seqscan TO on"
        ])
    elif arm_idx == 2:
        hints.extend([
            "SET enable_hashjoin TO on",
            "SET enable_indexonlyscan TO on",
            "SET enable_nestloop TO on",
            "SET enable_seqscan TO on"
        ])
    elif arm_idx == 3:
        hints.extend([
            "SET enable_hashjoin TO on",
            "SET enable_indexonlyscan TO on",
            "SET enable_seqscan TO on"
        ])
    elif arm_idx == 4:
        hints.extend([
            "SET enable_hashjoin TO on",
            "SET enable_indexonlyscan TO on",
            "SET enable_indexscan TO on",
            "SET enable_nestloop TO on",
            "SET enable_seqscan TO on"
        ])

    return hints


def collate_batch(batch):
    xs, ys = zip(*batch)  # ys: tuple of tensors shaped [1] each
    ys = torch.stack(ys, dim=0)  # [B, 1]
    return list(xs), ys


class TreeCNNRegDataset(Dataset):
    def __init__(self, trees, y_scaled):
        self.trees = trees
        self.y = torch.tensor(y_scaled, dtype=torch.float32)

    def __len__(self): return len(self.trees)

    def __getitem__(self, idx):
        return self.trees[idx], self.y[idx]


@dataclass
class TreeCNNConfig:
    train_epochs: int = 100
    batch_size: int = 16
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    cur_model_path = "./models/tree_expert_models/current"
    temp_model_path = "./models/tree_expert_models/temp"


class TreeCNNRegExpert:
    """
    TreeCNN expert combining encoding, network, preprocessing, and storage.
    """

    def __init__(self, config: TreeCNNConfig, buffer_mngr: BufferManager):
        super().__init__()

        self.buffer_mngr = buffer_mngr
        self.config = config
        # Auto-select device at runtime
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pipeline = Pipeline([
            ("log", FunctionTransformer(np.log1p, inverse_func=_inv_log1p, validate=True)),
            ("scale", MinMaxScaler())
        ])

        self.encoder = TreeFeaturizer()
        self.net: Optional[TreeCNN] = None
        self.in_channels: int = 0
        self.n_trained: int = 0

    def train_and_save(self, max_retries: int = 5, train_data_limit: int = 10):
        training_data = self.buffer_mngr.get_plan_latency_pairs(limit=train_data_limit)
        if training_data is None or len(training_data) == 0:
            raise ValueError("No training data available")

        has_current_model = os.path.exists(_nn_path(self.config.cur_model_path))

        attempt = 1
        while True:
            # Train a new model once
            loader, _ = self._get_data_loader(training_data)
            self.net = TreeCNN(self.in_channels).to(self.config.device)
            optimizer = optim.Adam(self.net.parameters())
            loss_fn = nn.MSELoss()
            self._run_train_epochs(loader=loader, optimizer=optimizer, loss_fn=loss_fn)
            # Save to temp (via unified save API)
            self.save(self.config.temp_model_path)
            # If no current exists, promote and stop
            if not has_current_model:
                self._promote_temp_to_current()
                print("[train] No current model; promoted temp to current.")
                break
            # Compare new(temp) vs old(current)
            if self.should_replace_model(self.config.cur_model_path, self.config.temp_model_path):
                self._promote_temp_to_current()
                print("[train] temp model accepted and promoted to current.")
                break
            if attempt >= max_retries:
                print("[train] Could not train model with better regression profile.")
                break
            print("New model rejected when compared with old model. Trying to retrain with emphasis.")
            print("Retry #", attempt)
            attempt += 1

    def compute_regressions(self, model_prefix: Optional[str]) -> Tuple[int, float]:
        model = enc = pipe = None
        if model_prefix:
            model, enc, pipe = self._load_eval_bundle(model_prefix)
        total_regressed = 0
        total_regression = 0.0
        for plan_group in self.buffer_mngr.get_plan_latency_groups():
            plan_group = list(plan_group)
            plans = [x.actual_plan_json for x in plan_group]
            best_latency = min(x.actual_latency for x in plan_group)
            if model_prefix:
                trees = enc.transform(plans)
                model.eval()
                with torch.no_grad():
                    pred_scaled = model(trees).cpu().numpy().reshape(-1, 1)
                    pred_times = pipe.inverse_transform(pred_scaled).reshape(-1)
                selection = int(np.argmin(pred_times))
            else:
                selection = 0
            selected_plan_latency = float(plan_group[selection].actual_latency)
            if selected_plan_latency > best_latency * 1.01:
                total_regressed += 1
            total_regression += (selected_plan_latency - best_latency)
        return total_regressed, total_regression

    def should_replace_model(self, old_prefix: Optional[str], new_prefix: str) -> bool:
        new_num_reg, new_reg_amnt = self.compute_regressions(new_prefix)
        cur_num_reg, cur_reg_amnt = self.compute_regressions(old_prefix)
        print("Old model # regressions:", cur_num_reg, "regression amount:", cur_reg_amnt)
        print("New model # regressions:", new_num_reg, "regression amount:", new_reg_amnt)
        if new_num_reg == 0:
            print("New model had no regressions.")
            return True
        elif cur_num_reg >= new_num_reg and cur_reg_amnt >= new_reg_amnt:
            print("New model with better regression profile than the old model.")
            return True
        else:
            print("New model did not have a better regression profile.")
            return False

    def _promote_temp_to_current(self):
        """Replace current (cur_model_path) directory contents with temp (temp_model_path) directory contents."""
        tmp_dir = self.config.temp_model_path
        cur_dir = self.config.cur_model_path

        # Remove current directory if exists
        if os.path.exists(cur_dir):
            shutil.rmtree(cur_dir, ignore_errors=True)

        # Copy temp directory to current
        if os.path.exists(tmp_dir):
            shutil.copytree(tmp_dir, cur_dir)

    def _run_train_epochs(self,
                          loader: DataLoader,
                          optimizer: optim.Optimizer,
                          loss_fn: nn.Module) -> float:
        """Run epochs and return final avg loss; supports simple early stopping."""
        losses = []
        for epoch in range(self.config.train_epochs):
            self.net.train()
            total = 0.0
            for batch_x, batch_y in loader:
                batch_y = batch_y.to(self.config.device, non_blocking=True)

                preds = self.net(batch_x)
                loss = loss_fn(preds, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += loss.item()

            avg = total / max(1, len(loader))
            losses.append(avg)
            print(f"Epoch {epoch}, loss={avg:.4f}")

            # early stopping
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    print("Stopped training from convergence condition at epoch", epoch)
                    break
        else:
            print("Stopped training after max epochs")
        return losses[-1] if losses else np.inf

    def _load_eval_bundle(self, model_dir: str):
        """
        Load model+encoder+pipeline from a directory for evaluation (non-mutating).
        Supports Bao-compatible format: nn_weights, x_transform, y_transform, channels, n
        """
        # Load in Bao-compatible format
        with open(_channels_path(model_dir), "rb") as f:
            in_channels = joblib.load(f)

        model = TreeCNN(in_channels).to(self.config.device)
        model.load_state_dict(torch.load(_nn_path(model_dir), map_location=self.config.device))
        model.eval()

        with open(_y_transform_path(model_dir), "rb") as f:
            pipeline = joblib.load(f)
        with open(_x_transform_path(model_dir), "rb") as f:
            encoder = joblib.load(f)

        return model, encoder, pipeline

    @staticmethod
    def _gen_candidate_qeps(sql: str, db_cli: PostgresConnector):
        plan_jsons = []
        hints_list = []
        for arm_idx in range(5):
            hints = _arm_idx_to_hints(arm_idx)
            db_cli.drop_buffer_cache()
            db_cli.apply_hints(hints)
            plan_json, _ = db_cli.explain(query=sql)

            hints_list.append(hints)
            plan_jsons.append(plan_json)
        return plan_jsons, hints_list

    def sql_enhancement(self, sql: str, db_cli: PostgresConnector):
        plans, hints = self._gen_candidate_qeps(sql=sql, db_cli=db_cli)
        assert len(plans) > 0

        # Parse and encode plans
        parsed_plans = [json.loads(plan) if isinstance(plan, str) else plan for plan in plans]
        # self.encoder.fit(parsed_plans)
        trees = self.encoder.transform(parsed_plans)

        # Predict costs for all plans
        self.net.eval()
        with torch.no_grad():
            scaled = self.net(trees).cpu().numpy()
            predicted_costs = self.pipeline.inverse_transform(scaled.reshape(-1, 1)).flatten()

        # Find the best plan and hint
        best_plan_idx = np.argmin(predicted_costs)
        print(f"Predicted best_plan_idx = {best_plan_idx}")
        best_hint_list = hints[best_plan_idx]

        # Convert hint list to SQL string format
        # hints is a list of SET statements like ["SET enable_hashjoin TO on", ...]
        hint_sql = ";\n".join(best_hint_list) + ";\n" if best_hint_list else ""

        # Return optimized SQL with hints prepended
        return hint_sql + sql, best_hint_list

    def save(self, path_prefix: Optional[str] = None):
        """
        Save current in-memory model to a directory in Bao-compatible format.
        Saves: nn_weights, x_transform, y_transform, channels, n
        """
        target_dir = path_prefix or self.config.cur_model_path
        os.makedirs(target_dir, exist_ok=True)

        # Save in Bao-compatible format
        torch.save(self.net.state_dict(), _nn_path(target_dir))
        with open(_y_transform_path(target_dir), "wb") as f:
            joblib.dump(self.pipeline, f)
        with open(_x_transform_path(target_dir), "wb") as f:
            joblib.dump(self.encoder, f)
        with open(_channels_path(target_dir), "wb") as f:
            joblib.dump(self.in_channels, f)
        with open(_n_path(target_dir), "wb") as f:
            joblib.dump(self.n_trained, f)

        print(f"Saved to {target_dir} (Bao-compatible format)")

    def load(self, model_dir: Optional[str] = None):
        """
        Load model from directory in Bao-compatible format.
        Loads: nn_weights, x_transform, y_transform, channels, n
        Handles compatibility with Bao models that may reference different modules.
        """
        model_dir = model_dir or self.config.cur_model_path

        # Load in Bao-compatible format
        with open(_n_path(model_dir), "rb") as f:
            self.n_trained = joblib.load(f)
        with open(_channels_path(model_dir), "rb") as f:
            self.in_channels = joblib.load(f)

        self.net = TreeCNN(self.in_channels).to(self.config.device)
        self.net.load_state_dict(torch.load(_nn_path(model_dir), map_location=self.config.device))
        self.net.eval()

        with open(_x_transform_path(model_dir), "rb") as f:
            self.encoder = joblib.load(f)

        with open(_y_transform_path(model_dir), "rb") as f:
            self.pipeline = joblib.load(f)

        print(f"Loaded from {model_dir} (Bao-compatible format)")

    def _get_data_loader(self, training_data: List[PlanRecord]):
        """
        Load PlanRecords, fit encoder & target pipeline,
        and build a shuffled DataLoader with a custom collate_fn.
        """
        total_samples = len(training_data)
        plans, execution_times = [], []

        for rec in training_data:
            plans.append(rec.actual_plan_json)
            execution_times.append(rec.actual_latency)
        valid_samples = len(plans)

        # Encode plans
        self.encoder.fit(plans)
        trees = self.encoder.transform(plans)

        # Fit target pipeline (log1p + scaling) and transform
        execution_times = np.array(execution_times).reshape(-1, 1)
        y_scaled = self.pipeline.fit_transform(execution_times).astype(np.float32)

        # Set model dims / counters
        self.in_channels = trees[0][0].shape[-1]  # feature dim
        self.n_trained = valid_samples

        # Dataset / DataLoader
        dataset = TreeCNNRegDataset(trees, y_scaled)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_batch,
            pin_memory=(self.config.device.type == "cuda"),
            num_workers=0,
        )

        # Return loader and training metadata for reporting
        return loader, {"total_samples": total_samples, "valid_samples": valid_samples}


if __name__ == "__main__":
    import argparse
    from exp_buffer.buffer_mngr import read_sql_files


    def _train(db_path: str):
        config = TreeCNNConfig(
            train_epochs=100,
            batch_size=16)

        buffer_mngr = BufferManager(db_path)
        expert = TreeCNNRegExpert(config=config, buffer_mngr=buffer_mngr)
        expert.train_and_save(train_data_limit=20)
        print("Training completed!")
        print("Save expert training completed successfully!")


    def _predict(db_path: str, input_sql_dir: str, dbname: str):
        config = TreeCNNConfig(
            train_epochs=100,
            batch_size=16)

        buffer_mngr = BufferManager(db_path)
        expert = TreeCNNRegExpert(config=config, buffer_mngr=buffer_mngr)
        expert.load()

        query_w_name = read_sql_files(input_sql_dir)
        with PostgresConnector(dbname) as conn:
            for sql_name, sql in query_w_name:
                query_id = sql_name
                new_sql, hints = expert.sql_enhancement(sql=sql, db_cli=conn)
                print(new_sql)

                # new_sql only used in integration with PG not offline
                execution_id = buffer_mngr.run_log_query_exec(
                    conn=conn, sql=sql, hints=hints)

                print(f"✓ Saved execution {execution_id} for {query_id}: {sql[:60]}...")
                print(execution_id)


    parser = argparse.ArgumentParser(description="Train Bao Expert from Unified Data")
    parser.add_argument('--dbname', type=str, default='imdb_ori', help='Database name')
    parser.add_argument(
        '--input_sql_dir',
        type=str,
        default="/Users/kevin/project_python/AI4QueryOptimizer/lqo_benchmark/workloads/bao/join_unique",
        help='Input SQL dir (one query per file)')

    args = parser.parse_args()

    # _train(f"buffer_{args.dbname}.db")
    _predict(f"buffer_{args.dbname}.db", args.input_sql_dir, args.dbname)
