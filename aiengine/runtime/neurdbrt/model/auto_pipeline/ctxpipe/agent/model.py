from enum import Enum, auto

import torch
from torch import nn

from auto_pipeline import comp
from auto_pipeline.config import DQNConfig

ENABLE_MODULO_HOOK = False
ENABLE_CONTEXT_GATE = True

N_DIM_EMBED = 1024
N_DIM_FIRST = 128
D = N_DIM_FIRST + N_DIM_EMBED
INFO_EXTRACTION_POS = 10
CTX_INTEGRATION_POS = 1

N_PRINT = 4


class ForwardMode(Enum):
    CLOSED = auto()
    GATED = auto()
    OPEN = auto()


def make_ctx_plugin_model(n_input: int, n_output: int) -> nn.Module:
    result = nn.Sequential(
        nn.Linear(n_input, N_DIM_FIRST),
        nn.LeakyReLU(),
        # nn.Linear(N_DIM_FIRST, N_DIM_FIRST),
        # nn.LeakyReLU(),
        # nn.Linear(128, 64),
        # nn.LeakyReLU(),
        # nn.Linear(64, 32),
        # nn.LeakyReLU(),
        nn.Linear(N_DIM_FIRST, n_output),
        nn.Tanh(),
    )

    return result

    # encoder_layer = nn.TransformerEncoderLayer(d_model=N_DIM_EMBED, nhead=8, dim_feedforward=512, batch_first=True)
    # return nn.TransformerEncoder(encoder_layer, num_layers=4)


def make_mm_layer(shape: tuple, device: torch.device) -> torch.Tensor:
    MEAN = 3 / shape[0]
    result = torch.normal(mean=MEAN, std=MEAN / 3, size=shape).to(device)

    result.requires_grad_(True)
    return result


def forward_context_gate(model, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    # print(x.shape, ctx.shape, model.context_gate.shape)

    info = torch.concat([x, ctx], dim=1)

    info = torch.matmul(info, model.context_gate)
    print("C MTML:", info.shape, info[:N_PRINT, :N_PRINT])

    info = info + model.context_gate_bias
    info = torch.nan_to_num(info)
    print("C +BAS:", info.shape, info[:N_PRINT, :N_PRINT])

    info = torch.sigmoid(info)
    print("C GATE:", info.shape, info[:N_PRINT, :N_PRINT])
    print("----------------------------------------")

    return info


class DQN(nn.Module):

    def __init__(self, num_inputs, actions_dim, config: DQNConfig):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, N_DIM_FIRST),
            nn.LeakyReLU(),
            nn.Linear(N_DIM_FIRST, N_DIM_FIRST),
            nn.LeakyReLU(),
            nn.Linear(N_DIM_FIRST, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, actions_dim),
            nn.Tanh(),
        )

        self._config = config
        self.enable_ctxpipe = config.enable_context_plugin

        if self.enable_ctxpipe:
            self.ctx_linear = make_ctx_plugin_model(
                n_input=N_DIM_EMBED, n_output=actions_dim
            )
            # self.context_gate_model = make_ctx_gate_model(n_input=actions_dim + N_DIM_FIRST)
            self.context_gate = make_mm_layer(
                (actions_dim + N_DIM_FIRST, actions_dim), device=config.device
            )
            self.context_gate_bias = make_mm_layer((actions_dim,), device=config.device)

    def forward(
        self, x: torch.Tensor, ctx: torch.Tensor, mode: ForwardMode = ForwardMode.GATED
    ):
        print("DQN Forward")

        device = self._config.device

        data_feature = x[:, : self._config.data_dim].to(device)
        print("data", data_feature.shape, data_feature)

        x = torch.concat([data_feature, x[:, self._config.data_dim :]], dim=-1)
        input_feature = x

        if not self.enable_ctxpipe:
            input_feature = self.nn(input_feature)
            print(f"No context: {input_feature[0]}")
            return input_feature

        ### CTXPIPE
        for i in range(len(self.nn)):
            input_feature = self.nn[i](input_feature)

            if i == len(self.nn) - INFO_EXTRACTION_POS:
                if ENABLE_CONTEXT_GATE:
                    ctx_integration = self.ctx_linear(ctx)

                    if mode == ForwardMode.CLOSED:
                        ctx_gate = torch.zeros(ctx_integration.shape).to(device)
                        ctx_gate.requires_grad_(False)

                    elif mode == ForwardMode.GATED:
                        ctx_gate = forward_context_gate(
                            self, input_feature, ctx_integration
                        )

                    elif mode == ForwardMode.OPEN:
                        ctx_gate = torch.ones(ctx_integration.shape).to(device)
                        ctx_gate.requires_grad_(False)

                    else:
                        raise ValueError(f"No such mode: {mode}")

                    ctx_integration = torch.mul(ctx_integration, ctx_gate)

            if i == len(self.nn) - CTX_INTEGRATION_POS:
                if ENABLE_CONTEXT_GATE:
                    print(f"Before gate: {input_feature[0]}")
                    print(f"Context: {ctx_integration[0]}")
                    print(f"Gate: {ctx_gate[0]}")
                    ctx_integration = torch.mul(ctx_integration - 1.0, ctx_gate) + 1.0
                    print(f"Context after gate: {ctx_integration[0]}")
                    input_feature = input_feature * ctx_integration
                    print(f"With context: {input_feature[0]}")

                if ENABLE_MODULO_HOOK:
                    input_feature.register_hook(self.modulo_func_hook)

        return input_feature


class RnnDQN(nn.Module):

    def __init__(self, actions_dim, config: DQNConfig):
        super(RnnDQN, self).__init__()

        self._config = config
        self.enable_ctxpipe = config.enable_context_plugin

        # input dim
        self.data_feature_dim = self._config.data_dim
        self.seq_feature_dim = len(comp.logic_pipeline_1)

        # seq_embedding_param
        prim_nums = (
            len(
                set(comp.imputernums)
                | set(comp.encoders)
                | set(comp.fpreprocessings)
                | set(comp.fengines)
                | set(comp.fselections)
            )
            + 1
            + 1
        )
        seq_embedding_dim = config.seq_embedding_dim
        # seq_lstm param
        seq_hidden_size = config.seq_hidden_size
        seq_num_layers = config.seq_num_layers
        # predictor param
        predictor_embedding_dim = config.predictor_embedding_dim
        # lpip param
        self.lpipeline_nums = comp.num_lpipelines
        lpipeline_embedding_dim = config.lpipeline_embedding_dim

        # sequence networks
        self.seq_embedding = nn.Embedding(prim_nums, seq_embedding_dim)

        self.seq_lstm = nn.LSTM(
            input_size=seq_embedding_dim,
            hidden_size=seq_hidden_size,
            num_layers=seq_num_layers,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        # predictor
        self.predictor_embedding = nn.Embedding(
            comp.num_predictors, predictor_embedding_dim
        )

        # logic pipeline
        self.lpipeline_embedding = nn.Embedding(
            comp.num_lpipelines, lpipeline_embedding_dim
        )

        self.nn = nn.Sequential(
            nn.Linear(
                self.data_feature_dim
                + seq_hidden_size * self.seq_feature_dim
                + predictor_embedding_dim
                + lpipeline_embedding_dim,
                N_DIM_FIRST,
            ),
            nn.LeakyReLU(),
            nn.Linear(N_DIM_FIRST, N_DIM_FIRST),
            nn.LeakyReLU(),
            nn.Linear(N_DIM_FIRST, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, actions_dim),
            nn.Tanh(),
        )

        if self.enable_ctxpipe:
            self.ctx_linear = make_ctx_plugin_model(
                n_input=N_DIM_EMBED, n_output=actions_dim
            )
            self.context_gate = make_mm_layer(
                (actions_dim + N_DIM_FIRST, actions_dim), device=config.device
            )
            self.context_gate_bias = make_mm_layer((actions_dim,), device=config.device)

    def forward(
        self, x, ctx: torch.Tensor, mode: ForwardMode = ForwardMode.GATED
    ):  # x batch_size * state
        print("RnnDQN Forward")

        device = self._config.device

        data_feature = x[:, : self.data_feature_dim].to(
            device
        )  # (batch_size , data_dim))

        seq_feature = (
            x[:, self.data_feature_dim : self.data_feature_dim + self.seq_feature_dim]
            .type(torch.LongTensor)
            .to(device)
        )  # (batch_size , 6)

        predictor_feature = (
            x[
                :,
                self.data_feature_dim
                + self.seq_feature_dim : self.data_feature_dim
                + self.seq_feature_dim
                + 1,
            ]
            .type(torch.LongTensor)
            .to(device)
        )  # (batch_size )

        lpipeline_feature = (
            x[
                :,
                self.data_feature_dim
                + self.seq_feature_dim
                + 1 : self.data_feature_dim
                + self.seq_feature_dim
                + 2,
            ]
            .type(torch.LongTensor)
            .to(device)
        )  # (batch_size )

        seq_embed_feature = self.seq_embedding(
            seq_feature
        )  # (batch_size , 6 , seq_embedding_dim)

        seq_hidden_feature, (h_1, c_1) = self.seq_lstm(
            seq_embed_feature
        )  # (6 , batch_size , seq_hidden_size)

        seq_hidden_feature = torch.flatten(
            seq_hidden_feature, start_dim=1
        )  # (batch_size , 6 * seq_hidden_size)

        predictor_embed_feature = self.predictor_embedding(
            predictor_feature
        )  # (batch_size, 1, predictor_embedding_dim)

        lpipeline_embed_deature = self.lpipeline_embedding(
            lpipeline_feature
        )  # (batch_size, 1, predictor_embedding_dim)

        predictor_embed_feature = torch.flatten(predictor_embed_feature, start_dim=1)
        lpipeline_embed_deature = torch.flatten(lpipeline_embed_deature, start_dim=1)

        input_feature = torch.cat(
            (
                data_feature,
                seq_hidden_feature,
                predictor_embed_feature,
                lpipeline_embed_deature,
            ),
            1,
        )

        if not self.enable_ctxpipe:
            input_feature = self.nn(input_feature)
            print(f"No context: {input_feature[0]}")
            return input_feature

        ### CTXPIPE
        for i in range(len(self.nn)):
            input_feature = self.nn[i](input_feature)

            ### S3
            if i == len(self.nn) - INFO_EXTRACTION_POS:
                if ENABLE_CONTEXT_GATE:
                    ctx_integration = self.ctx_linear(ctx)

                    if mode == ForwardMode.CLOSED:
                        ctx_gate = torch.zeros(ctx_integration.shape).to(device)
                        ctx_gate.requires_grad_(False)

                    elif mode == ForwardMode.GATED:
                        ctx_gate = forward_context_gate(
                            self, input_feature, ctx_integration
                        )

                    elif mode == ForwardMode.OPEN:
                        ctx_gate = torch.ones(ctx_integration.shape).to(device)
                        ctx_gate.requires_grad_(False)

                    else:
                        raise ValueError(f"No such mode: {mode}")

                    ctx_integration = torch.mul(ctx_integration, ctx_gate)

            if i == len(self.nn) - CTX_INTEGRATION_POS:
                if ENABLE_CONTEXT_GATE:
                    print(f"Before gate: {input_feature[0]}")
                    print(f"Context: {ctx_integration[0]}")
                    print(f"Gate: {ctx_gate[0]}")
                    ctx_integration = torch.mul(ctx_integration - 1.0, ctx_gate) + 1.0
                    print(f"Context after gate: {ctx_integration[0]}")
                    input_feature = input_feature * ctx_integration
                    print(f"With context: {input_feature[0]}")

                if ENABLE_MODULO_HOOK:
                    input_feature.register_hook(self.modulo_func_hook)

        return input_feature
