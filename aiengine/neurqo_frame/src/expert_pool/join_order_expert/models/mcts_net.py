import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from expert_pool.join_order_expert.models.torchfold import Fold

# Import from local package
from expert_pool.join_order_expert.models.tree_lstm import TreeLSTM
from torch import Tensor
from torch.nn import init


class Head(nn.Module):

    def __init__(self, hidden_size):
        super(Head, self).__init__()
        self.hidden_size = hidden_size
        self.head_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.head_layer(x)
        return out


class TreeSQLNet(nn.Module):
    """Trained network"""

    def __init__(
        self,
        head_num: int,
        input_size: int,
        hidden_size: int,
        table_num: int,
        sql_size: int,
    ):
        super(TreeSQLNet, self).__init__()

        self.table_num = table_num
        self.hidden_size = hidden_size
        self.head_num = head_num

        self.input_size = input_size
        self.sql_size = sql_size

        self.tree_lstm = TreeLSTM(input_size=input_size, hidden_size=hidden_size)
        self.sql_layer = nn.Linear(sql_size, hidden_size)

        self.head_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, head_num + 1),
        )

        self.table_embeddings = nn.Embedding(table_num, hidden_size)
        self.heads = nn.ModuleList(
            [Head(self.hidden_size) for _ in range(self.head_num + 1)]
        )
        self.relu = nn.ReLU()

    def leaf(self, alias_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Embed table id (leaf node); returns (embedding, zeros)"""
        table_embedding = self.table_embeddings(alias_id)
        return (
            table_embedding,
            torch.zeros(table_embedding.shape, dtype=torch.float32),
        )

    def input_feature(self, feature: list) -> Tensor:
        """Convert input feature list to tensor of shape (N, input_size)"""
        return torch.tensor(feature, dtype=torch.float32).reshape(-1, self.input_size)

    def sql_feature(self, feature: list) -> Tensor:
        """Convert SQL-level feature to 1×sql_size tensor"""
        return torch.tensor(feature, dtype=torch.float32).reshape(1, -1)

    def target_vec(self, target: float) -> Tensor:
        """Repeat scalar target across all heads → (1, head_num)"""
        return torch.tensor([target] * self.head_num, dtype=torch.float32).reshape(
            1, -1
        )

    def tree_node(
        self,
        h_left: Tensor,
        c_left: Tensor,
        h_right: Tensor,
        c_right: Tensor,
        feature: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply tree LSTM node merge"""
        h, c = self.tree_lstm(h_left, c_left, h_right, c_right, feature)
        return (h, c)

    def logits(
        self, encoding: Tensor, sql_feature: Tensor, prt: bool = False
    ) -> Tensor:
        """Concatenate encoding and sql_feature and output logits"""
        sql_hidden = self.relu(self.sql_layer(sql_feature))
        out_encoding = torch.cat([encoding, sql_hidden], dim=1)
        out = self.head_layer(out_encoding)
        return out  # shape: (batch_size, head_num + 1)

    def zero_hc(self, input_dim: int = 1) -> Tuple[Tensor, Tensor]:
        """Return zero-initialized hidden and cell state"""
        return (
            torch.zeros(input_dim, self.hidden_size),
            torch.zeros(input_dim, self.hidden_size),
        )


class TreeSQLNetBuilder:
    def __init__(self, value_network: TreeSQLNet, var_weight: float):
        self.value_network = value_network

        for _, p in self.value_network.named_parameters():
            if p.ndim == 2:
                init.xavier_normal_(p)
            else:
                init.uniform_(p)

        self.optimizer = optim.Adam(
            value_network.parameters(), lr=3e-4, betas=(0.9, 0.999)
        )
        self.loss_function = MSEVAR(var_weight)
        self.var_weight = var_weight

    def _train_step(self, multi_value: torch.tensor, target_feature: torch.tensor, var):
        # 1. forward + backpropagate
        loss_value = self.loss_function(
            multi_value=multi_value, target=target_feature, var=var
        )
        self.optimizer.zero_grad()
        loss_value.backward()

        # 2. Clip gradients elementwise to [-2, 2] to prevent exploding grads
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-2, 2)
        self.optimizer.step()
        return loss_value

    def train_on_sample(
        self,
        tree_feature: torch.tensor,
        sql_feature: torch.tensor,
        target_value_ms: float,
        mask: torch.tensor,
    ) -> Tuple[float, float, float, torch.tensor]:

        target_feature = torch.tensor(
            [target_value_ms] * self.value_network.head_num, dtype=torch.float32
        ).reshape(1, -1)

        multi_value = self.plan_to_value(
            tree_feature=tree_feature, sql_feature=sql_feature
        )
        loss_value = self._train_step(
            multi_value=multi_value[:, : self.value_network.head_num] * mask,
            target_feature=target_feature * mask,
            var=multi_value[:, self.value_network.head_num],
        )

        plan_sql_vec_mean, plan_sql_vec_variance = self.mean_and_std(
            multi_value=multi_value[:, : self.value_network.head_num]
        )

        return (
            loss_value.item(),
            plan_sql_vec_mean[0],
            plan_sql_vec_variance[0],
            target_feature,
        )

    def train_on_batch(self, samples: List, device: torch.device):
        if not samples:
            return None
        fold = Fold(cuda=(device.type == "cuda"))
        target_features = []
        masks = []
        multi_list = []
        target_values = []
        for one_sample in samples:
            # Convert each plan to neural network computation
            multi_value = self.plan_to_value_fold(
                tree_feature=one_sample.tree_feature,
                sql_feature=one_sample.sql_feature,
                fold=fold,
            )

            # Collect all the pieces
            masks.append(one_sample.mask)
            target_features.append(one_sample.target_feature)
            target_values.append(one_sample.target_feature.mean().item())
            multi_list.append(multi_value)

        # Execute the Graph
        multi_value = fold.apply(self.value_network, [multi_list])[0]
        # Combine all samples into batch tensors
        mask = torch.cat(masks, dim=0)
        target_feature = torch.cat(target_features, dim=0)

        loss_value = self._train_step(
            multi_value=multi_value[:, : self.value_network.head_num] * mask,
            target_feature=target_feature * mask,
            var=multi_value[:, self.value_network.head_num],
        )

        plan_sql_vec_mean, plan_sql_vec_variance = self.mean_and_std(
            multi_value=multi_value[:, : self.value_network.head_num]
        )

        new_weight = [
            abs(x - target_values[idx]) * target_values[idx]
            for idx, x in enumerate(plan_sql_vec_mean)
        ]

        return loss_value, new_weight

    def predict_with_uncertainty_batch(
        self,
        tree_features: List[torch.tensor],
        sql_feature: torch.tensor,
        device: torch.device,
    ):
        fold = Fold(cuda=(device.type == "cuda"))

        nodes = []
        for tree_feature in tree_features:
            node = self.plan_to_value_fold(tree_feature, sql_feature, fold)
            nodes.append(node)

        multi_value = fold.apply(self.value_network, [nodes])[0]

        mean, variance = self.mean_and_std(
            multi_value=multi_value[:, : self.value_network.head_num]
        )
        uncertainty = torch.exp(
            multi_value[:, self.value_network.head_num] * self.var_weight
        ).data.reshape(-1)

        return list(zip(mean, variance, uncertainty.tolist()))

    def plan_to_value(self, tree_feature, sql_feature):
        def recursive(tree_feature):
            if isinstance(tree_feature[1], tuple):
                feature = tree_feature[0]
                h_left, c_left = recursive(tree_feature=tree_feature[1])
                h_right, c_right = recursive(tree_feature=tree_feature[2])
                return self.value_network.tree_node(
                    h_left, c_left, h_right, c_right, feature
                )
            else:
                feature = tree_feature[0]
                h_left, c_left = self.value_network.leaf(tree_feature[1])
                h_right, c_right = self.value_network.zero_hc()
                return self.value_network.tree_node(
                    h_left, c_left, h_right, c_right, feature
                )

        plan_feature = recursive(tree_feature=tree_feature)
        multi_value = self.value_network.logits(plan_feature[0], sql_feature)
        return multi_value

    def plan_to_value_fold(
        self, tree_feature: torch.Tensor, sql_feature: torch.Tensor, fold: Fold
    ):
        def recursive(tree_feature):
            if isinstance(tree_feature[1], tuple):
                # This is a join node - process left and right children
                feature = tree_feature[0]
                h_left, c_left = recursive(tree_feature=tree_feature[1]).split(2)
                h_right, c_right = recursive(tree_feature=tree_feature[2]).split(2)
                return fold.add("tree_node", h_left, c_left, h_right, c_right, feature)
            else:
                # This is a leaf node (table scan)
                feature = tree_feature[0]
                h_left, c_left = fold.add("leaf", tree_feature[1]).split(2)
                h_right, c_right = fold.add("zero_hc", 1).split(2)
                return fold.add("tree_node", h_left, c_left, h_right, c_right, feature)

        plan_feature, c = recursive(tree_feature=tree_feature).split(2)
        multi_value = fold.add("logits", plan_feature, sql_feature)
        return multi_value

    def plan_to_value_linear_fold(self, tree_feature, sql_feature, fold):
        def recursive(tree_feature, depth=1):
            if isinstance(tree_feature[1], tuple):
                feature = tree_feature[0]
                h_left, c_left = recursive(
                    tree_feature=tree_feature[1], depth=depth + 1
                ).split(2)
                h_right, c_right = recursive(
                    tree_feature=tree_feature[2], depth=depth + 1
                ).split(2)
                return fold.add("tree_node", h_left, c_left, h_right, c_right, feature)
            else:
                feature = tree_feature[0]
                h_left, c_left = fold.add("leaf", tree_feature[1]).split(2)
                h_right, c_right = fold.add("zero_hc", 1).split(2)
                return fold.add("tree_node", h_left, c_left, h_right, c_right, feature)

        plan_feature = recursive(tree_feature=tree_feature)
        multi_value = fold.add("logits", plan_feature, sql_feature)
        return multi_value

    def plan_to_value_mlp_fold(self, tree_feature, sql_feature, fold):
        def recursive(tree_feature, depth=1):
            if isinstance(tree_feature[1], tuple):
                feature = tree_feature[0]
                h_left, c_left = recursive(
                    tree_feature=tree_feature[1], depth=depth + 1
                ).split(2)
                h_right, c_right = recursive(
                    tree_feature=tree_feature[2], depth=depth + 1
                ).split(2)
                return fold.add("tree_node", h_left, c_left, h_right, c_right, feature)
            else:
                feature = tree_feature[0]
                h_left, c_left = fold.add("leaf", tree_feature[1]).split(2)
                h_right, c_right = fold.add("zero_hc", 1).split(2)
                return fold.add("tree_node", h_left, c_left, h_right, c_right, feature)

        plan_feature = recursive(tree_feature=tree_feature)
        multi_value = fold.add("logits", plan_feature, sql_feature)
        return multi_value

    def mean_and_std(
        self, multi_value: torch.Tensor
    ) -> Tuple[List[float], List[float]]:
        mean_value = multi_value.mean(dim=1)  # [B]
        std_value = multi_value.std(dim=1, unbiased=False)  # [B]
        return mean_value.detach().cpu().tolist(), std_value.detach().cpu().tolist()


class MSEVAR(nn.Module):
    """MSE with variance (uncertainty-aware regression).

    This loss is derived from the negative log-likelihood of a Gaussian:
        L = (1 / (2σ²)) * (y - μ)² + (1/2) * log(σ²)
    where μ is the predicted mean, σ² is the predicted variance.

    In this implementation, variance is predicted as `var`, scaled by `var_weight`.
    """

    def __init__(self, var_weight: float):
        super(MSEVAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, multi_value, target, var):
        # predicted variance term, scaled by var_weight
        var_wei = (self.var_weight * var).reshape(-1, 1)

        # 1) data-fit term (squared error weighted by inverse variance)
        #    => encourages mean prediction μ close to target y
        loss1 = torch.exp(-var_wei) * (multi_value - target) ** 2

        # 2) variance penalty term (acts like + log σ²)
        #    => prevents model from driving variance to infinity
        loss2 = var_wei

        # 3) optional regularization term (not used here, just set to 0)
        #    => in some implementations, could be L2 reg or KL term
        loss3 = 0

        # total loss: expected NLL of Gaussian with mean=multi_value, var=exp(var_wei)
        loss = loss1 + loss2 + loss3
        return loss.mean()


class ValueNet(nn.Module):
    """
    ValueNet

    A small fusion network that scores a (query_encoding, join_order_sequence) pair.

    Components (logic unchanged):
      1) A 1-layer MLP that projects the query encoding QE -> R^{hidden_size}.
      2) An embedding table that maps each join-order token id to R^{hidden_size}.
      3) A 1D CNN over the embedded join-order sequence with kernel size 5 and
         temporal max-pooling over the full sequence length to produce a single vector.
      4) Concatenation of the query vector and the pooled CNN vector, followed by
         a 3-layer MLP head that outputs a single scalar.

    Expected shapes:
      - QE: Float tensor of shape (B, in_dim)
      - JO: Long tensor of shape (B, config.max_hint_num) with token ids in [0, n_words)

    Output:
      - value: Float tensor of shape (B, 1)
    """

    def __init__(
        self, max_hint_num: int, in_dim: int, n_words: int = 40, hidden_size: int = 64
    ) -> None:
        super(ValueNet, self).__init__()
        self.dim = in_dim
        self.max_hint_num = max_hint_num
        self.hs = hidden_size

        # 1) Project query encoding to hidden_size
        self.query_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(inplace=True),
        )

        # 2) Token embeddings for join-order ids (size = n_words vocab, dim = hidden_size)
        #    (Comment preserved: "#2 * max_column_in_table * size")
        self.table_embeddings = nn.Embedding(n_words, hidden_size)

        # 3) Temporal CNN over embedded join-order sequence.
        #    Input to Conv1d is (B, C=self.hs, T=config.max_hint_num).
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hs, out_channels=self.hs, kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.hs, out_channels=self.hs, kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.hs, out_channels=self.hs, kernel_size=5, padding=2
            ),
            nn.MaxPool1d(
                kernel_size=self.max_hint_num
            ),  # pools over full time to (B, C, 1)
        )

        # 4) Final MLP head on concatenated [query_vec ; jo_vec] -> scalar
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, QE: torch.Tensor, JO: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            QE: (B, in_dim) float tensor — per-query features/encoding.
            JO: (B, config.max_hint_num) long tensor — join-order token ids.

        Returns:
            (B, 1) float tensor — predicted value.
        """
        # Project query encoding, then ensure (B, hs)
        # (Original code reshaped; keep behavior identical.)
        query_vec = self.query_proj(QE).reshape(-1, self.hs)  # (B, hs)

        # Embed join-order ids: (B, max_hint_num, hs)
        jo_emb = self.table_embeddings(JO).reshape(-1, self.max_hint_num, self.hs)

        # CNN expects (B, C, T) -> (B, hs, max_hint_num)
        jo_cnn_in = jo_emb.permute(0, 2, 1)  # (B, hs, max_hint_num)

        # After MaxPool1d over full T, shape is (B, hs, 1)
        jo_vec = self.cnn(jo_cnn_in)  # (B, hs, 1)

        # Flatten pooled temporal dim: (B, hs)
        jo_vec = jo_vec.reshape(-1, self.hs)

        # Concatenate query and join-order vectors: (B, 2*hs)
        fused = torch.cat((query_vec, jo_vec), dim=1)

        # Regress to a scalar: (B, 1)
        out = self.output_layer(fused)
        return out
