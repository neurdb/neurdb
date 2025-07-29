import os
import pickle
from abc import ABCMeta
from typing import Any, List, Optional, Union

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.optim import Adam, AdamW

from auto_pipeline import env, comp, deterministic, util
from config import AgentConfig, default_config, default_dqn_config

from ..ctx import TableEmbedder, TextEmbedder, embedder
from ..env.pipeline import Pipeline
from .model import DQN, ForwardMode, RnnDQN
from .replay import ReplayBuffer

CONFIG = default_config
DQNCONFIG = default_dqn_config


class AgentModel(metaclass=ABCMeta):
    def __init__(self, name: str, nn: nn.Module, action_dim: int, lr: float) -> None:
        self._nn = nn.to(env.DEVICE)
        self._name = name
        self._action_dim = action_dim
        self._optimizer = Adam(nn.parameters(), lr=lr)

    @property
    def name(self):
        return self._name

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._nn(*args, **kwds)

    def load(self, model_path: str):
        if os.path.exists(model_path):
            self._nn.load_state_dict(torch.load(model_path, map_location=env.DEVICE))
        else:
            logger.warning(f"No such model weight file: {model_path}")

    def save(self, model_path: str):
        torch.save(self._nn.state_dict(), model_path)


class ImputerNumModel(AgentModel):
    def __init__(self) -> None:
        super().__init__(
            name="ImputerNum",
            nn=RnnDQN(comp.num_imputernums, DQNCONFIG),
            action_dim=comp.num_imputernums,
            lr=CONFIG.learning_rate,
        )


class EncoderModel(AgentModel):
    def __init__(self) -> None:
        super().__init__(
            name="Encoder",
            nn=RnnDQN(comp.num_encoders, DQNCONFIG),
            action_dim=comp.num_encoders,
            lr=CONFIG.learning_rate,
        )


class FeaturePreprocessingModel(AgentModel):
    def __init__(self) -> None:
        super().__init__(
            name="FeaturePreprocessing",
            nn=RnnDQN(comp.num_fpreprocessings, DQNCONFIG),
            action_dim=comp.num_fpreprocessings,
            lr=CONFIG.learning_rate,
        )


class FeatureEngineModel(AgentModel):
    def __init__(self) -> None:
        super().__init__(
            name="FeatureEngine",
            nn=RnnDQN(comp.num_fengines, DQNCONFIG),
            action_dim=comp.num_fengines,
            lr=CONFIG.learning_rate,
        )


class FeatureSelectionModel(AgentModel):
    def __init__(self) -> None:
        super().__init__(
            name="FeatureSelection",
            nn=RnnDQN(comp.num_fselections, DQNCONFIG),
            action_dim=comp.num_fselections,
            lr=CONFIG.learning_rate,
        )


class LogicalPipelineModel(AgentModel):
    def __init__(self, state_dim: int) -> None:
        super().__init__(
            name="LogicPipeline",
            nn=DQN(state_dim, comp.num_lpipelines, DQNCONFIG),
            action_dim=comp.num_lpipelines,
            lr=CONFIG.learning_rate,
        )


class Agent:
    def __init__(self, config: AgentConfig):
        self._config = config
        self.buffer = ReplayBuffer(config.max_buff)
        self.no_random = False

        self.imputernum_model = ImputerNumModel()
        self.encoder_model = EncoderModel()
        self.fpreprocessing_model = FeaturePreprocessingModel()
        self.fengine_model = FeatureEngineModel()
        self.fselection_model = FeatureSelectionModel()
        self.lpipeline_model = LogicalPipelineModel(config.lpip_state_dim)

        self.imputernum_model_optim = self.imputernum_model.optimizer
        self.encoder_model_optim = self.encoder_model.optimizer
        self.fpreprocessing_model_optim = self.fpreprocessing_model.optimizer
        self.fengine_model_optim = self.fengine_model.optimizer
        self.fselection_model_optim = self.fselection_model.optimizer
        self.lpipeline_model_optim = self.lpipeline_model.optimizer

        self.last_raw_dataset_ctx = None

    def set_last_raw_dataset_ctx(self, value: torch.Tensor):
        self.last_raw_dataset_ctx = value

    def _get_model_by_name(self, name: str) -> AgentModel:
        if name == "ImputerNum":
            return self.imputernum_model
        elif name == "Encoder":
            return self.encoder_model
        elif name == "FeaturePreprocessing":
            return self.fpreprocessing_model
        elif name == "FeatureEngine":
            return self.fengine_model
        elif name == "FeatureSelection":
            return self.fselection_model
        elif name == "LogicPipeline":
            return self.lpipeline_model

        raise ValueError(f"No such model: {name}")

    def act(
        self,
        pipeline: Pipeline,
        state: Union[np.ndarray, torch.Tensor],
        index: str,
        tryed_list=[],
        epsilon=None,
    ):
        if epsilon is None:
            epsilon = self._config.epsilon_min

        model = self._get_model_by_name(index)

        randnum = deterministic.epsilon_greedy_rng.random()
        if self.no_random or (randnum > epsilon):
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(env.DEVICE)

            if isinstance(embedder, TextEmbedder):
                if len(pipeline.train_x.index) > 100:
                    train_x_csv = pipeline.train_x.sample(n=100).to_csv(index=False)
                else:
                    train_x_csv = pipeline.train_x.to_csv(index=False)
            elif isinstance(embedder, TableEmbedder):
                train_x_csv = pipeline.train_x.sample(n=embedder.MAX_N_ROWS)
                if len(train_x_csv.columns) > embedder.MAX_N_COLUMNS:
                    train_x_csv = train_x_csv.iloc[:, -embedder.MAX_N_COLUMNS :]

            ctx_embeddings = embedder.embed(train_x_csv)  # type:ignore
            ctx_embeddings = ctx_embeddings.unsqueeze(dim=0).to(env.DEVICE)

            q_value: torch.Tensor = model(state, ctx_embeddings).cpu()

            action_index_list = [i for i in range(model.action_dim)]
            action_index_list = np.array(
                [i for i in list(action_index_list) if i not in tryed_list]
            )
            q_value_temp = np.array(
                [
                    i.detach()
                    for index, i in enumerate(list(q_value[0]))
                    if index not in tryed_list
                ]
            )

            # with open("action.temp.log", "a") as f:
            #     f.write(f"{q_value_temp}\n")

            action: int = action_index_list[q_value_temp.argmax()]
        else:
            q_value_temp = np.array(
                [
                    index
                    for index, i in enumerate(np.zeros(model.action_dim))
                    if index not in tryed_list
                ]
            )
            action_index_list = [i for i in range(model.action_dim)]
            action_index_list = list(
                np.array([i for i in list(action_index_list) if i not in tryed_list])
            )
            action: int = deterministic.random_action_rng.choice(action_index_list, 1)[
                0
            ]

        return action, randnum > epsilon

    def learn_lp(self) -> None:
        s0, a, r, ctx = self.buffer.lp_sample(self._config.logic_batch_size)
        a = torch.tensor(np.array(a), dtype=torch.long)
        r = torch.tensor(np.array(r), dtype=torch.float)
        s0 = torch.tensor(s0, dtype=torch.float)

        ctx = torch.stack([c for c in ctx], dim=0)

        if not self._config.enable_context_plugin:
            self._do_learn_lp(s0, a, r, ctx, None)
        else:
            if not self._config.enable_ocg_experience_replay:
                m = ForwardMode.GATED
                logger.debug(f"No OCG. Forward mode: {m.name}")
                self._do_learn_lp(s0, a, r, ctx, m)
            else:
                for m in ForwardMode:
                    logger.debug(f"Set forward mode to: {m.name}")
                    self._do_learn_lp(s0, a, r, ctx, m)

    def _do_learn_lp(self, s0, a, r, ctx, m):
        loss_log = []
        if os.path.exists(self._config.lp_loss_log_file_name):
            loss_log = np.load(
                self._config.lp_loss_log_file_name, allow_pickle=True
            ).tolist()

        q_values = self.lpipeline_model(s0.to(env.DEVICE), ctx.to(env.DEVICE), m).cpu()

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = (q_value - r.detach()).pow(2).mean()
        if torch.isnan(loss):
            print("WARN  loss is NaN. Skipping back-prop")
            return

        self.lpipeline_model_optim.zero_grad()
        loss.backward()
        self.lpipeline_model_optim.step()

        if m is None or m == ForwardMode.GATED:
            loss_log.append(loss.item())
            np.save(self._config.lp_loss_log_file_name, loss_log)

    def learn_components(self, test=False) -> None:
        if not test:
            s0, a, r, s1, done, index, logic_pipeline_id, ctx = self.buffer.sample(
                self._config.batch_size
            )
        else:
            s0, a, r, s1, done, index, logic_pipeline_id, ctx = self.buffer.sample(5)

        s0_imputernum = []
        a_imputernum = []
        r_imputernum = []
        s1_imputernum = []
        done_imputernum = []
        ctx_imputernum = []

        s0_encoder = []
        a_encoder = []
        r_encoder = []
        s1_encoder = []
        done_encoder = []
        sfi_encoder = []
        ctx_encoder = []

        s0_fpreprocessing = []
        a_fpreprocessing = []
        r_fpreprocessing = []
        s1_fpreprocessing = []
        done_fpreprocessing = []
        sfi_fpreprocessing = []
        ctx_fpreprocessing = []

        s0_fengine = []
        a_fengine = []
        r_fengine = []
        s1_fengine = []
        done_fengine = []
        sfi_fengine = []
        ctx_fengine = []

        s0_fselection = []
        a_fselection = []
        r_fselection = []
        s1_fselection = []
        done_fselection = []
        sfi_fselection = []
        ctx_fselection = []

        for i in range(len(index)):
            if index[i] == "ImputerNum":
                if a[i] != comp.num_imputernums:
                    a_imputernum.append(a[i])
                    r_imputernum.append(r[i])
                    s1_imputernum.append(s1[i])
                    done_imputernum.append(done[i])
                    s0_imputernum.append(s0[i])
                    ctx_imputernum.append(ctx[i])

            elif index[i] == "Encoder":
                if a[i] != comp.num_encoders:
                    a_encoder.append(a[i])
                    r_encoder.append(r[i])
                    s1_encoder.append(s1[i])
                    done_encoder.append(done[i])
                    s0_encoder.append(s0[i])
                    sfi_encoder.append(logic_pipeline_id[i])
                    ctx_encoder.append(ctx[i])

            elif index[i] == "FeaturePreprocessing":
                s0_fpreprocessing.append(s0[i])
                a_fpreprocessing.append(a[i])
                r_fpreprocessing.append(r[i])
                s1_fpreprocessing.append(s1[i])
                done_fpreprocessing.append(done[i])
                sfi_fpreprocessing.append(logic_pipeline_id[i])
                ctx_fpreprocessing.append(ctx[i])

            elif index[i] == "FeatureEngine":
                s0_fengine.append(s0[i])
                a_fengine.append(a[i])
                r_fengine.append(r[i])
                s1_fengine.append(s1[i])
                done_fengine.append(done[i])
                sfi_fengine.append(logic_pipeline_id[i])
                ctx_fengine.append(ctx[i])

            elif index[i] == "FeatureSelection":
                s0_fselection.append(s0[i])
                a_fselection.append(a[i])
                r_fselection.append(r[i])
                s1_fselection.append(s1[i])
                done_fselection.append(done[i])
                sfi_fselection.append(logic_pipeline_id[i])
                ctx_fselection.append(ctx[i])

        a_imputernum = util.device_tensor(a_imputernum, torch.long)
        a_encoder = util.device_tensor(a_encoder, torch.long)
        a_fpreprocessing = util.device_tensor(a_fpreprocessing, torch.long)
        a_fengine = util.device_tensor(a_fengine, torch.long)
        a_fselection = util.device_tensor(a_fselection, torch.long)

        r_imputernum = util.device_tensor(r_imputernum)
        r_encoder = util.device_tensor(r_encoder)
        r_fpreprocessing = util.device_tensor(r_fpreprocessing)
        r_fengine = util.device_tensor(r_fengine)
        r_fselection = util.device_tensor(r_fselection)

        done_imputernum = util.device_tensor(done_imputernum)
        done_encoder = util.device_tensor(done_encoder)
        done_fpreprocessing = util.device_tensor(done_fpreprocessing)
        done_fengine = util.device_tensor(done_fengine)
        done_fselection = util.device_tensor(done_fselection)

        s0_imputernum = util.device_tensor(s0_imputernum)
        s0_encoder = util.device_tensor(s0_encoder)
        s0_fpreprocessing = util.device_tensor(s0_fpreprocessing)
        s0_fengine = util.device_tensor(s0_fengine)
        s0_fselection = util.device_tensor(s0_fselection)

        s1_imputernum = util.device_tensor(s0_imputernum)
        s1_encoder = util.device_tensor(s1_encoder)
        s1_fpreprocessing = util.device_tensor(s1_fpreprocessing)
        s1_fengine = util.device_tensor(s1_fengine)
        s1_fselection = util.device_tensor(s1_fselection)

        ctx_imputernum = util.device_tensor(ctx_imputernum)
        ctx_encoder = util.device_tensor(ctx_encoder)
        ctx_fpreprocessing = util.device_tensor(ctx_fpreprocessing)
        ctx_fengine = util.device_tensor(ctx_fengine)
        ctx_fselection = util.device_tensor(ctx_fselection)

        s0s = [s0_imputernum, s0_encoder, s0_fpreprocessing, s0_fengine, s0_fselection]
        s1s = [s1_imputernum, s1_encoder, s1_fpreprocessing, s1_fengine, s1_fselection]
        aas = [a_imputernum, a_encoder, a_fpreprocessing, a_fengine, a_fselection]
        rrs = [r_imputernum, r_encoder, r_fpreprocessing, r_fengine, r_fselection]
        dones = [
            done_imputernum,
            done_encoder,
            done_fpreprocessing,
            done_fengine,
            done_fselection,
        ]
        sfis = [
            torch.Tensor((1,)),
            sfi_encoder,
            sfi_fpreprocessing,
            sfi_fengine,
            sfi_fselection,
        ]
        ctxs = [
            ctx_imputernum,
            ctx_encoder,
            ctx_fpreprocessing,
            ctx_fengine,
            ctx_fselection,
        ]

        if not self._config.enable_context_plugin:
            self._do_learn_components(s0s, s1s, aas, rrs, dones, sfis, ctxs, None)
        else:
            if not self._config.enable_ocg_experience_replay:
                m = ForwardMode.GATED
                logger.debug(f"No OCG. Forward mode: {m.name}")
                self._do_learn_components(s0s, s1s, aas, rrs, dones, sfis, ctxs, m)
            else:
                for m in ForwardMode:
                    logger.debug(f"Set forward mode to: {m.name}")
                    self._do_learn_components(s0s, s1s, aas, rrs, dones, sfis, ctxs, m)

    def _do_learn_components(
        self,
        s0s: List[torch.Tensor],
        s1s: List[torch.Tensor],
        aas: List[torch.Tensor],
        rrs: List[torch.Tensor],
        dones: List[torch.Tensor],
        sfis: List[torch.Tensor],
        ctxs: List[torch.Tensor],
        m: Optional[ForwardMode] = None,
    ):
        result = []

        loss_log = {"0": [], "2": [], "3": [], "4": [], "5": []}
        if os.path.exists(self._config.loss_log_file_name):
            with open(self._config.loss_log_file_name, "rb") as f:
                loss_log: dict = pickle.load(f)

        if len(s0s[0]) > 0:
            # print(s0_imputernum.shape, ctx_imputernum.shape)
            q_values_0 = self.imputernum_model(s0s[0], ctxs[0], m)
            # print(q_values_0.shape)
            # print(s1_imputernum.shape, ctx_imputernum.shape)
            next_q_values_0 = self.encoder_model(s1s[0], ctxs[0], m)
            # print(next_q_values_0.shape)
            next_q_value_0 = next_q_values_0.max(1)[0]
            # print(next_q_value_0.shape)
            q_value_0 = q_values_0.gather(1, aas[0].unsqueeze(1)).squeeze(1)
            # print(q_value_0.shape)
            expected_q_value_0 = rrs[0] + next_q_value_0 * (1 - dones[0])

            loss_0 = (q_value_0 - expected_q_value_0.detach()).pow(2).mean()

            if torch.isnan(loss_0):
                logger.warning("loss_0 is NaN. Skipping back-prop")
            else:
                self.imputernum_model_optim.zero_grad()
                loss_0.backward()
                self.imputernum_model_optim.step()

                logger.debug(
                    f"loss_log[0]: {loss_log['0']}, loss_0.item(): {loss_0.item()}"
                )

                if m is None or m == ForwardMode.GATED:
                    result.append(loss_0.item())
                    loss_log["0"].append(loss_0.item())
        else:
            result.append(-1)

        if len(s0s[1]) > 0:
            # print(s0_encoder.shape, ctx_encoder.shape)
            q_values_2 = self.encoder_model(s0s[1], ctxs[1], m)
            # print(q_values_2.shape)
            # print(s1_encoder.shape, ctx_encoder.shape)
            for logiclineid in sfis[1]:
                if logiclineid in [0, 1]:
                    next_q_values_2 = self.fpreprocessing_model(s1s[1], ctxs[1], m)
                elif logiclineid in [2, 3]:
                    next_q_values_2 = self.fengine_model(s1s[1], ctxs[1], m)
                if logiclineid in [4, 5]:
                    next_q_values_2 = self.fselection_model(s1s[1], ctxs[1], m)
            # print(next_q_values_2.shape)
            next_q_value_2 = next_q_values_2.max(1)[0]
            # print(next_q_value_2.shape)
            q_value_2 = q_values_2.gather(1, aas[1].unsqueeze(1)).squeeze(1)
            # print(q_value_2.shape)
            expected_q_value_2 = rrs[1] + next_q_value_2 * (1 - dones[1])
            # print(expected_q_value_2.shape)

            loss_2 = (q_value_2 - expected_q_value_2.detach()).pow(2).mean()

            if torch.isnan(loss_2):
                logger.warning("loss_2 is NaN. Skipping back-prop")
            else:
                self.encoder_model_optim.zero_grad()
                loss_2.backward()
                self.encoder_model_optim.step()

                logger.debug(
                    f"loss_log[2]: {loss_log['2']}, loss_2.item(): {loss_2.item()}"
                )

                if m is None or m == ForwardMode.GATED:
                    result.append(loss_2.item())
                    loss_log["2"].append(loss_2.item())
        else:
            result.append(-1)

        if len(s0s[2]) > 0:
            q_values_3 = self.fpreprocessing_model(s0s[2], ctxs[2], m)

            for logiclineid in sfis[2]:
                if logiclineid in [0, 5]:
                    next_q_values_3 = self.fengine_model(s1s[2], ctxs[2], m)
                elif logiclineid in [1, 3]:
                    next_q_values_3 = self.fselection_model(s1s[2], ctxs[2], m)
                elif logiclineid in [2, 4]:
                    next_q_values_3 = self.fpreprocessing_model(s1s[2], ctxs[2], m)

            next_q_value_3 = next_q_values_3.max(1)[0]
            q_value_3 = q_values_3.gather(1, aas[2].unsqueeze(1)).squeeze(1)
            expected_q_value_3 = rrs[2] + next_q_value_3 * (1 - dones[2])

            loss_3 = (q_value_3 - expected_q_value_3.detach()).pow(2).mean()

            if torch.isnan(loss_3):
                logger.warning("loss_3 is NaN. Skipping back-prop")
            else:
                self.fpreprocessing_model_optim.zero_grad()
                loss_3.backward()
                self.fpreprocessing_model_optim.step()

                logger.debug(
                    f"loss_log[3]: {loss_log['3']}, loss_3.item(): {loss_3.item()}"
                )

                if m is None or m == ForwardMode.GATED:
                    result.append(loss_3.item())
                    loss_log["3"].append(loss_3.item())
        else:
            result.append(-1)

        if len(s0s[3]) > 0:
            q_values_4 = self.fengine_model(s0s[3], ctxs[3], m)
            for logiclineid in sfis[3]:
                if logiclineid in [0, 2]:
                    next_q_values_4 = self.fselection_model(s1s[3], ctxs[3], m)
                elif logiclineid in [1, 5]:
                    next_q_values_4 = self.fengine_model(s1s[3], ctxs[3], m)
                elif logiclineid in [3, 4]:
                    next_q_values_4 = self.fpreprocessing_model(s1s[3], ctxs[3], m)

            next_q_value_4 = next_q_values_4.max(1)[0]
            q_value_4 = q_values_4.gather(1, aas[3].unsqueeze(1)).squeeze(1)
            expected_q_value_4 = rrs[3] + next_q_value_4 * (1 - dones[3])

            loss_4 = (q_value_4 - expected_q_value_4.detach()).pow(2).mean()

            if torch.isnan(loss_4):
                logger.warning("loss_4 is NaN. Skipping back-prop")
            else:
                self.fengine_model_optim.zero_grad()
                loss_4.backward()
                self.fengine_model_optim.step()

                logger.debug(
                    f"loss_log[4]: {loss_log['4']}, loss_4.item(): {loss_4.item()}"
                )

                if m is None or m == ForwardMode.GATED:
                    result.append(loss_4.item())
                    loss_log["4"].append(loss_4.item())
        else:
            result.append(-1)

        if len(s0s[4]) > 0:
            q_values_5 = self.fselection_model(s0s[4], ctxs[4], m)
            for logiclineid in sfis[4]:
                if logiclineid in [0, 3]:
                    next_q_values_5 = self.fselection_model(s1s[4], ctxs[4], m)
                elif logiclineid in [1, 4]:
                    next_q_values_5 = self.fengine_model(s1s[4], ctxs[4], m)
                elif logiclineid in [2, 5]:
                    next_q_values_5 = self.fpreprocessing_model(s1s[4], ctxs[4], m)

            next_q_value_5 = next_q_values_5.max(1)[0]
            q_value_5 = q_values_5.gather(1, aas[4].unsqueeze(1)).squeeze(1)
            expected_q_value_5 = rrs[4] + next_q_value_5 * (1 - dones[4])

            loss_5 = (q_value_5 - expected_q_value_5.detach()).pow(2).mean()

            if torch.isnan(loss_5):
                logger.warning("loss_5 is NaN. Skipping back-prop")
            else:
                self.fselection_model_optim.zero_grad()
                loss_5.backward()
                self.fselection_model_optim.step()

                logger.debug(
                    f"loss_log[5]: {loss_log['5']}, loss_5.item(): {loss_5.item()}"
                )

                if m is None or m == ForwardMode.GATED:
                    result.append(loss_5.item())
                    loss_log["5"].append(loss_5.item())
        else:
            result.append(-1)

        with open(self._config.loss_log_file_name, "wb") as f:
            pickle.dump(loss_log, f)

        return result

    def load_weights(self, output: str, tag=""):
        if output is None:
            return

        for m, p in {
            self.imputernum_model: "imputernum_model.pkl",
            self.encoder_model: "encoder_model.pkl",
            self.fpreprocessing_model: "fpreprocessing_model.pkl",
            self.fengine_model: "fengine_model.pkl",
            self.fselection_model: "fselection_model.pkl",
            self.lpipeline_model: "logical_pipeline.pkl",
        }.items():
            m.load(self._model_path(output, p, tag))

    def save_model(self, output: str, tag=""):
        for m, p in {
            self.imputernum_model: "imputernum_model.pkl",
            self.encoder_model: "encoder_model.pkl",
            self.fpreprocessing_model: "fpreprocessing_model.pkl",
            self.fengine_model: "fengine_model.pkl",
            self.fselection_model: "fselection_model.pkl",
            self.lpipeline_model: "logical_pipeline.pkl",
        }.items():
            m.save(self._model_path(output, p, tag))

    def _model_path(self, out_dir: str, pkl_name: str, tag: str):
        if tag == "":
            return os.path.join(out_dir, f"{pkl_name}")
        else:
            return os.path.join(out_dir, f"{tag}_{pkl_name}")
