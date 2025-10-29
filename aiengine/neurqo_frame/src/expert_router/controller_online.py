# Standard library imports
import random
from collections import deque, namedtuple
from typing import Dict, List

# Third-party imports
import numpy as np
import torch
import torch.optim as optim

# Local/project imports
from common import BaseConfig

from .loss import HybridLossReg
from .logger import plogger
from .model import QueryOptMHSASuperNode


class OnlineRouter:
    BufferEntry = namedtuple("BufferEntry", ["X_batch", "chosen_optimizer", "observed_latency", "timestamp", "weight"])

    def __init__(self,
                 cfg: BaseConfig,
                 pretrained_model: QueryOptMHSASuperNode,
                 buffer_size=1000,
                 batch_size=16,
                 update_frequency=10,
                 lr=1e-5,
                 decay_rate=0.01,
                 regret_window=50,
                 dropout_samples=10,
                 alpha=0.5,
                 loss_gamma=3,
                 class_weights=None):
        self.cfg = cfg
        self.model = pretrained_model
        self.model.eval()
        self.buffer = deque(maxlen=buffer_size)  # (features, optimizer, latency, timestamp, weight)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.decay_rate = decay_rate  # For time-weighted sampling
        self.regret_window = deque(maxlen=regret_window)  # For shift detection
        self.dropout_samples = dropout_samples  # Number of MC Dropout samples
        self.optimizer = optim.Adam(
            list(self.model.shared.parameters()) +
            list(self.model.class_head.parameters()) +
            list(self.model.reg_head.parameters()),
            lr=lr
        )

        self.criterion = HybridLossReg(cfg=self.cfg, alpha=alpha, gamma=loss_gamma, class_weights=class_weights)
        self.query_count = 0
        self.output_dim = pretrained_model.output_dim
        self.avg_regret = 0.0
        plogger.info(f"OnlineRouter initialized with DCTS, buffer_size={buffer_size}, lr={lr}")

    def select_optimizer(self, X_batch: dict, query_id=None):
        """Thompson Sampling with Offline Prior and Robust Variance."""
        with torch.no_grad():
            device = X_batch['join_conditions'].device
            self.model.eval()  # Offline prediction first

            # Offline prediction as prior
            class_logits_off, reg_times_off, _ = self.model(X_batch)
            offline_probs = torch.sigmoid(class_logits_off).squeeze(0)
            offline_valid_mask = offline_probs > 0.5
            offline_pred_idx = torch.argmin(reg_times_off.squeeze(0).masked_fill(~offline_valid_mask, float(
                'inf')) if offline_valid_mask.sum() > 0 else reg_times_off.squeeze(0))
            offline_mean = reg_times_off.squeeze(0)  # Use as prior mean

            self.model.train()  # Enable dropout for TS
            dropout_samples = max(3, self.dropout_samples - (self.query_count // 50))  # Min 3 samples, reduce over time
            sampled_latencies = torch.zeros(dropout_samples, self.output_dim, device=device)

            for i in range(dropout_samples):
                _, reg_times, _ = self.model(X_batch)
                sampled_latencies[i] = reg_times.squeeze(0)

            # Robust variance calculation
            mean_latencies = sampled_latencies.mean(dim=0)
            variance = sampled_latencies.var(dim=0, unbiased=True)  # Use unbiased variance
            variance = torch.nan_to_num(variance, nan=1e-6, posinf=1e6)  # Handle nan/inf
            max_variance = variance.max()
            threshold = 0.05 * mean_latencies.max()

            # TS with offline prior
            if max_variance > threshold and self.query_count < 50:  # Explore early, then trust learning
                # Sample from Gaussian with offline mean and learned variance
                sampled_values = torch.normal(mean=offline_mean, std=variance.sqrt())
                chosen_optimizer = torch.argmin(sampled_values).item()
                plogger.info(f"High uncertainty: {max_variance:.4f}, Explored, Choice: {chosen_optimizer}")
            else:
                chosen_optimizer = torch.argmin(mean_latencies).item()  # Exploit learned mean
                plogger.info(f"Low uncertainty: {max_variance:.4f}, Exploited, Choice: {chosen_optimizer}")

            self.model.eval()
            return chosen_optimizer

    def add_to_buffer_update_model(self, X_batch: dict, chosen_optimizer: int, observed_latency: float, query_id=None):
        timestamp = self.query_count
        weight = 1.0
        X_batch_np = {key: value.squeeze(0).cpu().numpy() for key, value in X_batch.items()}
        entry = self.BufferEntry(X_batch_np, chosen_optimizer, observed_latency, timestamp, weight)
        self.buffer.append(entry)
        self.query_count += 1

        self.model.eval()
        with torch.no_grad():
            _, reg_times, _ = self.model(X_batch)  # [1, output_dim]
            best_pred_latency = reg_times.min().item()
            regret = observed_latency - best_pred_latency
            self.regret_window.append(regret)
            self.avg_regret = np.mean(self.regret_window)

        if self.avg_regret > 0.2 * best_pred_latency and len(self.regret_window) == self.regret_window.maxlen:
            plogger.info(f"Shift detected: avg regret={self.avg_regret:.4f}")
            self.flush_old_buffer()

        if self.query_count % self.update_frequency == 0 and len(self.buffer) >= self.batch_size:
            self.update_model()

    def flush_old_buffer(self):
        """Remove old entries if shift detected."""
        midpoint = self.buffer_size // 2
        while len(self.buffer) > midpoint:
            self.buffer.popleft()

    def dynamic_padding_collate(self, X_batch: List[Dict]):
        # X_batch: List of feature dicts

        # Find maximum lengths for joins and filters in the batch
        max_joins = max(len(x['join_conditions']) for x in X_batch)
        max_filters = max(len(x['filter_conditions']) for x in X_batch)

        # Pad join_conditions and filter_conditions
        padded_join_conditions = []
        padded_filter_conditions = []

        for x in X_batch:
            joins = torch.from_numpy(x['join_conditions'])  # Shape: [num_joins, feature_dim]
            filters = torch.from_numpy(x['filter_conditions'])  # Shape: [num_filters, feature_dim]

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
        join_conditions_batch = torch.stack(padded_join_conditions)  # Shape: [batch_size, max_joins, 4]
        filter_conditions_batch = torch.stack(padded_filter_conditions)  # Shape: [batch_size, max_filters, 3]

        # Reconstruct X_batch with padded tensors
        X_padded = {
            'join_conditions': join_conditions_batch,
            'filter_conditions': filter_conditions_batch,
            'table_sizes': torch.stack([torch.from_numpy(x['table_sizes']) for x in X_batch])
        }
        return X_padded

    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return

        # Sample uniformly (temporary, will refine to query-centric later)
        batch = random.sample(list(self.buffer), self.batch_size)
        X_batch_list = [item.X_batch for item in batch]
        chosen_optimizers = torch.tensor([item.chosen_optimizer for item in batch], device=self.cfg.DEVICE)
        observed_latencies = torch.tensor([item.observed_latency for item in batch], device=self.cfg.DEVICE)

        X_batch = self.dynamic_padding_collate(X_batch_list)
        self.model.train()
        self.optimizer.zero_grad()
        class_logits, reg_times, _ = self.model(X_batch)

        # Ground truth: use observed latency for chosen optimizer
        y_times = reg_times.clone().detach()
        y_multi = torch.zeros_like(y_times)
        for i, (opt, lat) in enumerate(zip(chosen_optimizers, observed_latencies)):
            y_times[i, opt] = lat
            y_multi[i, opt] = 1.0  # Mark chosen as "good" for now

        # Dynamic alpha based on regret
        alpha = 0.5 + 0.3 * min(self.avg_regret / (self.avg_regret + 1), 1.0)  # Cap at 0.8
        self.criterion.alpha = alpha
        loss = self.criterion(class_logits, reg_times, y_multi, y_times)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.model.eval()
        plogger.info(f"Updated with loss={loss.item():.4f}, alpha={alpha:.4f}, regret={self.avg_regret:.4f}")

    # baseline method
    def inference(self, loader):
        self.model.eval()  # Set model to evaluation mode
        predictions_time = 0  # Store query label to optimizer predictions
        predictions_time_list = []
        query_selection = []
        per_query_offline = []

        with torch.no_grad():  # Disable gradient computation for inference
            for X_batch, _, y_execution_times, q_batch_ids in loader:
                # Move data to the appropriate device
                y_execution_times = y_execution_times.to(BaseConfig.DEVICE)

                # Get model predictions: class logits and regression times
                class_logits, reg_times, emb_output = self.model(X_batch)
                probs = torch.sigmoid(class_logits)  # Convert logits to probabilities
                for i, (prob_row, reg_row, y_time_row, q_id) in enumerate(
                        zip(probs, reg_times, y_execution_times, q_batch_ids)):
                    # Select optimizers using 'hyper_1' strategy
                    valid_mask = prob_row > 0.5
                    if valid_mask.sum() == 0:
                        pred_idx = torch.argmin(reg_row)
                    else:
                        valid_times = reg_row.masked_fill(~valid_mask, float('inf'))
                        pred_idx = torch.argmin(valid_times)

                    predictions_time += y_time_row[pred_idx].item()
                    predictions_time_list.append(predictions_time)
                    query_selection.append((q_id, pred_idx.cpu().item()))

                    per_query_offline.append(y_execution_times[i, pred_idx].item())

        return predictions_time, predictions_time_list, query_selection, per_query_offline
