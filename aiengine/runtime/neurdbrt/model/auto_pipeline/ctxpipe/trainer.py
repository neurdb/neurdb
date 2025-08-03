import gc
import math
import os
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from .. import comp
from ..config import AgentConfig
from .agent.dqn import Agent
from .ctx import TableEmbedder, TextEmbedder, embedder
from .env.enviroment import Environment
from .env.primitives.imputercat import ImputerCatPrim
from .env.primitives.primitive import Primitive

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Trainer:

    def __init__(self, agent: Agent, env: Environment, test_pred, config: AgentConfig):
        self.agent = agent
        self.agent.no_random = False

        self.env = env
        self._config = config
        self.test_pred = test_pred

        self.imputernum_state = None
        self.imputernum_action = None

        self.epsilon_start = self._config.epsilon_start
        self.epsilon_final = self._config.epsilon_min
        self.epsilon_decay = self._config.eps_decay

        self.outputdir = self._config.model_dir

    def step(
        self,
        fr: int,
        state: np.ndarray,
        one_pip_ismodel: list,
        seq: list,
    ):
        epsilon = self._epsilon_by_frame(fr)
        logger.debug(f"epsilon={epsilon}")
        pipeline_index = self.env.pipeline.get_index()
        has_num_nan, has_cat_nan = self.env.has_nan()
        tried_list = []

        action = -1
        step: Primitive = Primitive()
        reward = -1

        try:
            ctx_embedding = embedder.embed(self._ctx_data(embedder=embedder))  # type: ignore
        except:
            logger.error("Context embedding error")
            state, one_pip_ismodel, seq = self._reset(epsilon=epsilon)
            return state, one_pip_ismodel, seq

        """select an action by epsilon-greedy"""
        logic_pipeline_id = self.env.pipeline.logic_pipeline_id
        curr_component = comp.lpipelines[logic_pipeline_id][pipeline_index]

        logger.debug(f"Current component: {curr_component}")

        step_result: Optional[List] = None
        while True:
            if curr_component == "ImputerNum":
                action, step = self._act_imputernum(
                    ctx_embedding, has_num_nan, state, tried_list, epsilon
                )
            elif curr_component == "ImputerCat":
                action, step = self._act_imputercat(has_cat_nan)
            elif curr_component == "Encoder":
                action, step = self._act_encoder(state, tried_list, epsilon)
            elif curr_component == "FeaturePreprocessing":
                action, step = self._act_feature_preprocessing(
                    state, tried_list, epsilon
                )
            elif curr_component == "FeatureEngine":
                action, step = self._act_feature_engine(state, tried_list, epsilon)
            elif curr_component == "FeatureSelection":
                action, step = self._act_feature_selection(state, tried_list, epsilon)
            else:
                raise ValueError(f"No such component type: {curr_component}")

            if action in tried_list:
                repeat_time += 1
                continue

            tried_list.append(action)

            """execute the action"""
            step_result, err = self.env.step(step)

            if err is None:
                logger.debug(f"[{curr_component}] ACT {step.name}")
                break

            if err is not None and len(tried_list) > 0:
                step_result = None
                reward = err
                logger.warning(f"Exceeded max retry time. Reward: {reward}")
                break

            logger.warning(
                f"Retried [{len(tried_list)}] for {curr_component}: {tried_list}"
            )

        if step_result is None:
            is_pipeline_done = True
        else:
            state, reward, next_state, is_pipeline_done = step_result
            seq.append(action)

            if curr_component == "ImputerNum":
                self.imputernum_state = state
                self.imputernum_action = action
            elif curr_component == "ImputerCat":
                self.agent.buffer.add(
                    self.imputernum_state,
                    self.imputernum_action,
                    reward,
                    next_state,
                    False,
                    "ImputerNum",
                    logic_pipeline_id,
                    ctx_embedding,
                )
            else:
                self.agent.buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    is_pipeline_done,
                    curr_component,
                    logic_pipeline_id,
                    ctx_embedding,
                )

            state = next_state

            """save checkpoint"""
            if fr % self._config.checkpoint_interval == 0:
                self.agent.save_model(self.outputdir, tag=f"ctx_{fr}")

            """update model"""
            logger.debug(
                f"buffer_size={self.agent.buffer.size()}, lp_size={self.agent.buffer.lp_size()}"
            )
            if (
                self.agent.buffer.size() >= self._config.batch_size
                and self.agent.buffer.lp_size() >= self._config.logic_batch_size
                and fr % self._config.backpropagate_interval == 0
            ):
                self.agent.learn_components()
                self.agent.learn_lp()
                self._result_log["learn_time"] += 1

        if is_pipeline_done:
            logger.info(
                f"Dataset: {self.curr_dataset}. "
                f"Pipeline: {self.env.pipeline.sequence}. "
                f"Predictor: {self.env.pipeline.predictor.name}. "
                f"Reward: {reward}"
            )

            if step_result is not None:
                """add sample"""
                self.agent.buffer.lp_add(
                    self.env.lpip_state,
                    logic_pipeline_id,
                    reward,
                    self.agent.last_raw_dataset_ctx,
                )

                # self._log_pipeline(one_pip_ismodel, reward, seq)

            state, one_pip_ismodel, seq = self._reset(epsilon)

        return state, one_pip_ismodel, seq

    def _reset(self, epsilon: float):
        self.env.reset()

        self.env.pipeline.logic_pipeline_id, _ = self.agent.act(
            self.env.pipeline, self.env.lpip_state, "LogicPipeline", epsilon=epsilon
        )

        state = self.env.get_state()
        one_pip_ismodel = []
        seq = []

        return state, one_pip_ismodel, seq

    @property
    def curr_dataset(self):
        return self._config.classification_task_dic[self.env.pipeline.taskid]["dataset"]

    def train(self, pre_fr=0):
        one_pip_ismodel = []
        seq = []

        self._init_log()

        self.agent.load_weights(self.outputdir, tag=f"ctx_{pre_fr}")

        self.env.reset()
        self.env.pipeline.logic_pipeline_id, _ = self.agent.act(
            self.env.pipeline,
            self.env.lpip_state,
            "LogicPipeline",
            epsilon=self._epsilon_by_frame(0),
        )

        state = self.env.get_state()

        for fr in range(pre_fr + 1, self._config.frames + 1):
            state, one_pip_ismodel, seq = self.step(fr, state, one_pip_ismodel, seq)

            logger.info(
                f"[TRAIN] Step {fr} - dataset: {self.curr_dataset} "
                f"seq: {self.env.pipeline.sequence} "
                f"model: {self.env.pipeline.predictor.name}"
            )

            gc.collect()

    def _epsilon_by_frame(self, frame_id):
        return self.epsilon_final + (
            self.epsilon_start - self.epsilon_final
        ) * math.exp(-1.0 * frame_id / self.epsilon_decay)

    def _ctx_data(self, embedder: Union[TextEmbedder, TableEmbedder]):
        if isinstance(embedder, TextEmbedder):
            if len(self.env.pipeline.train_x.index) > 100:
                result = self.env.pipeline.train_x.sample(n=100).to_csv(index=False)
            else:
                result = self.env.pipeline.train_x.to_csv(index=False)
        elif isinstance(embedder, TableEmbedder):
            result = self.env.pipeline.train_x.sample(n=embedder.MAX_N_ROWS)
            if len(result.columns) > embedder.MAX_N_COLUMNS:
                result = result.iloc[:, -embedder.MAX_N_COLUMNS :]

        return result

    def _do_agent_act(
        self, state, component_type, tried_list, epsilon
    ) -> Tuple[int, bool]:
        action, is_model = self.agent.act(
            self.env.pipeline, state, component_type, tried_list, epsilon
        )

        return action, is_model

    def _log_action(self, component_type: str, step: Primitive):
        pass
        # with open("action.temp.log", "a") as f:
        #     f.write(f"{component_type} {step.name}\n")

    def _act_imputernum(
        self, ctx_embedding, has_num_nan, state, tried_list, epsilon
    ) -> Tuple[int, Primitive]:
        self.agent.set_last_raw_dataset_ctx(ctx_embedding)

        if not has_num_nan:
            return len(comp.imputernums), Primitive()

        action, is_model = self._do_agent_act(state, "ImputerNum", tried_list, epsilon)
        step: Primitive = deepcopy(comp.imputernums[action])

        if is_model:
            self._log_action("imputernum", step)

        return action, step

    def _act_imputercat(self, has_cat_nan) -> Tuple[int, Primitive]:
        if not has_cat_nan:
            return -1, Primitive()

        return -1, ImputerCatPrim()

    def _act_encoder(self, state, tried_list, epsilon) -> Tuple[int, Primitive]:
        if not self.env.has_cat_cols():
            return len(comp.encoders), Primitive()

        action, is_model = self._do_agent_act(state, "Encoder", tried_list, epsilon)
        step = deepcopy(comp.encoders[action])

        if is_model:
            self._log_action("encoder", step)

        return action, step

    def _act_feature_preprocessing(
        self, state, tried_list, epsilon
    ) -> Tuple[int, Primitive]:
        action, is_model = self._do_agent_act(
            state, "FeaturePreprocessing", tried_list, epsilon
        )
        step = deepcopy(comp.fpreprocessings[action])

        if is_model:
            self._log_action("featurepreprocessing", step)

        return action, step

    def _act_feature_engine(self, state, tried_list, epsilon) -> Tuple[int, Primitive]:
        action, is_model = self._do_agent_act(
            state, "FeatureEngine", tried_list, epsilon
        )
        step = deepcopy(comp.fengines[action])

        if is_model:
            self._log_action("featureengine", step)

        return action, step

    def _act_feature_selection(
        self, state, tried_list, epsilon
    ) -> Tuple[int, Primitive]:
        action, is_model = self._do_agent_act(
            state, "FeatureSelection", tried_list, epsilon
        )
        step = deepcopy(comp.fselections[action])

        if is_model:
            self._log_action("featureselection", step)

        return action, step

    def _init_log(self):
        self._result_log = {}
        if os.path.exists(self._config.result_log_file_name):
            self._result_log: Any = np.load(
                self._config.result_log_file_name, allow_pickle=True
            ).item()

        for k in ["reward_dic", "max_action", "max_reward", "seq_log"]:
            if k not in self._result_log:
                self._result_log[k] = {}
        if "learn_time" not in self._result_log:
            self._result_log["learn_time"] = 0

    def _log_pipeline(self, one_pip_ismodel, reward, seq):
        for i in range(len(one_pip_ismodel)):
            if one_pip_ismodel[i] == True:
                if self.env.pipeline.taskid not in self._result_log["max_reward"]:
                    self._result_log["max_reward"][self.env.pipeline.taskid] = {}

                if i not in self._result_log["max_reward"][self.env.pipeline.taskid]:
                    self._result_log["max_reward"][self.env.pipeline.taskid][i] = []

                if i not in self._result_log["max_action"]:
                    self._result_log["max_action"][i] = []

                self._result_log["max_reward"][self.env.pipeline.taskid][i].append(
                    reward
                )
                self._result_log["max_action"][i].append(seq[4])

        if self.env.pipeline.taskid not in self._result_log["seq_log"]:
            self._result_log["seq_log"][self.env.pipeline.taskid] = []

        self._result_log["seq_log"][self.env.pipeline.taskid].append(
            (
                self._result_log["learn_time"],
                [i.name for i in self.env.pipeline.sequence],
                reward,
                one_pip_ismodel,
            )
        )

        if self.env.pipeline.taskid not in self._result_log["reward_dic"]:
            self._result_log["reward_dic"][self.env.pipeline.taskid] = []

        self._result_log["reward_dic"][self.env.pipeline.taskid].append(reward)
        np.save(self._config.result_log_file_name, self._result_log)
        # self.agent.save_model(self.outputdir, 'best')
