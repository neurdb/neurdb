# Third-party imports
import torch
import torch.nn as nn

# Local/project imports
from common.base_config import BaseConfig


class EmbedSuperNode(nn.Module):
    def __init__(self, num_tables, num_columns, embedding_dim, num_heads, num_layers, dataset):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dataset = dataset

        # Embeddings for tables and columns
        self.table_embeddings = nn.Embedding(num_tables + 1, embedding_dim, padding_idx=0)
        self.column_embeddings = nn.Embedding(num_columns + 1, embedding_dim, padding_idx=0)

        # Projection layers for join and filter vectors
        self.join_projection = nn.Linear(4 * embedding_dim, embedding_dim)
        self.filter_projection = nn.Linear(2 * embedding_dim + 1, embedding_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Super token
        self.super_token = nn.Embedding(1, embedding_dim)

        # Fully connected layer for regression task
        self.fc = nn.Linear(embedding_dim + num_tables, 1)

        self._init_xavier()

    def forward(self, x):
        # Extract input tensors
        join_conditions = x["join_conditions"].to(BaseConfig.DEVICE, torch.float32)
        filter_conditions = x["filter_conditions"].to(BaseConfig.DEVICE, torch.float32)
        table_sizes = x["table_sizes"].to(BaseConfig.DEVICE, torch.float32)

        # Create masks for padded positions
        join_mask = (join_conditions[:, :, 0] == 0)
        filter_mask = (filter_conditions[:, :, 0] == 0)
        sequence_mask = torch.cat([join_mask, filter_mask], dim=1)

        # Process Join Conditions
        table1_id = join_conditions[:, :, 0].long()
        col1_id = join_conditions[:, :, 1].long()
        table2_id = join_conditions[:, :, 2].long()
        col2_id = join_conditions[:, :, 3].long()

        table1_emb = self.table_embeddings(table1_id)
        col1_emb = self.column_embeddings(col1_id)
        table2_emb = self.table_embeddings(table2_id)
        col2_emb = self.column_embeddings(col2_id)

        join_vector_raw = torch.cat([table1_emb, col1_emb, table2_emb, col2_emb], dim=-1)
        join_vector = self.join_projection(join_vector_raw)

        # Process Filter Conditions
        table_id = filter_conditions[:, :, 0].long()
        col_id = filter_conditions[:, :, 1].long()
        selectivity = filter_conditions[:, :, 2].unsqueeze(-1)

        table_emb = self.table_embeddings(table_id)
        col_emb = self.column_embeddings(col_id)

        filter_vector = torch.cat([table_emb, col_emb, selectivity], dim=-1)
        filter_vector = self.filter_projection(filter_vector)

        # Combine into Sequence
        sequence = torch.cat([join_vector, filter_vector],
                             dim=1)  # [batch_size, max_joins + max_filters, embedding_dim]

        # Add Super Token
        batch_size = sequence.size(0)
        super_token = self.super_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 1, embedding_dim]
        sequence_with_super = torch.cat([super_token, sequence],
                                        dim=1)  # [batch_size, max_joins + max_filters + 1, embedding_dim]

        # Update Mask for Super Token
        super_mask = torch.zeros(batch_size, 1, device=BaseConfig.DEVICE, dtype=torch.bool)  # Super token is not masked
        sequence_mask = torch.cat([super_mask, sequence_mask], dim=1)  # [batch_size, max_joins + max_filters + 1]

        # Apply Transformer Encoder
        attn_output = self.transformer_encoder(
            sequence_with_super,
            src_key_padding_mask=sequence_mask
        )  # [batch_size, max_joins + max_filters + 1, embedding_dim]

        # Extract Super Token Output
        super_output = attn_output[:, 0, :]  # [batch_size, embedding_dim]

        # Log-normalize non-zero table sizes
        # balances scale compatibility with semantic meaning,
        # mask = (table_sizes != 0)  # Identify non-zero positions
        # table_sizes_normalized = table_sizes.clone()
        # table_sizes_normalized[mask] = torch.log1p(table_sizes[mask])
        # x_full = torch.cat([super_output, table_sizes_normalized], dim=-1)  # [batch_size, embedding_dim + num_tables]

        x_full = torch.cat([super_output, table_sizes], dim=-1)  # [batch_size, embedding_dim + num_tables]

        return x_full, super_output

    def _init_xavier(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def save(self, best_model_state, model_path_prefix):
        torch.save(best_model_state, model_path_prefix + f"/model_pre_train_superNode_{self.dataset}.pth")

    def load(self, model_path_prefix):
        print(f"loading embedding from " + model_path_prefix + f"/model_pre_train_superNode_{self.dataset}.pth")
        state_dict = torch.load(model_path_prefix + f"/model_pre_train_superNode_{self.dataset}.pth",
                                map_location=torch.device(BaseConfig.DEVICE))
        return state_dict


class QueryOptMHSASuperNode(nn.Module):
    def __init__(self, num_tables, num_columns, output_dim, embedding_dim, num_heads,
                 embedding_path, is_fix_emb: bool, dataset, num_layers,
                 cfg: BaseConfig):
        super(QueryOptMHSASuperNode, self).__init__()

        self.cfg = cfg
        self.output_dim = output_dim
        self.embedding_model = EmbedSuperNode(
            num_tables, num_columns, embedding_dim, num_heads, num_layers, dataset)
        if embedding_path:
            state_dict = self.embedding_model.load(embedding_path)
            self.embedding_model.load_state_dict(state_dict)

            # Fix embeddings
            if is_fix_emb:
                print("[QueryOptimizerMultiHeadAtten2] load embed, and fix it ")
                for param in self.embedding_model.parameters():
                    param.requires_grad = False
            else:
                print("[QueryOptimizerMultiHeadAtten2] load embed, but finetune it ")
        else:
            print("[QueryOptimizerMultiHeadAtten2] train from scrach")

        # Shared network after attention
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim + num_tables, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Output heads
        self.class_head = nn.Linear(32, output_dim)
        self.reg_head = nn.Linear(32, output_dim)

        # Initialize weights
        self._init_weights(embedding_path)

    def _init_weights(self, embedding_path):
        """Initialize weights for linear layers using Xavier initialization."""
        for module in self.modules():

            # Skip embedding_model and its submodules, if we load it from pre-trained result
            if module is self.embedding_model or module in self.embedding_model.modules():
                continue

            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x):
        x_full, super_output = self.embedding_model(x)

        # Pass through shared network
        shared_output = self.shared(x_full)  # [batch_size, 32]

        # Output heads
        class_logits = self.class_head(shared_output)  # [batch_size, output_dim]

        # todo: this always predoce the zero, if negative. so use softplus torch.softplus(x) = log(1 + exp(x))
        # reg_times = torch.relu(self.reg_head(shared_output))

        # reg_times = torch.softplus(self.reg_head(shared_output))# [batch_size, output_dim], ensure non-negative

        # Regression head with scaled sigmoid
        max_value = self.cfg.EXECUTION_TIME_OUT
        reg_times = torch.sigmoid(self.reg_head(shared_output)) * max_value  # [0, max_value]

        return class_logits, reg_times, x_full
        # return class_logits, reg_times, super_output
