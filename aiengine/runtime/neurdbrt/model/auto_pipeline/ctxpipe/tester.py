import math
import os
import time
from copy import deepcopy
from typing import List, Optional

import numpy as np
from loguru import logger

from .. import comp
from ..config import AgentConfig
from .agent.dqn import Agent
from .env.enviroment import Environment
from .env.primitives.imputercat import ImputerCatPrim
from .env.primitives.primitive import Primitive

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Tester:

    def __init__(self, agent: Agent, env: Environment, test_pred, config: AgentConfig):
        self.agent = agent
        self.agent.no_random = True

        self.env = env
        self.test_pred = test_pred

        self._config = config
        self.epsilon_start = self._config.epsilon_start
        self.epsilon_final = self._config.epsilon_min
        self.epsilon_decay = self._config.eps_decay

        self.outputdir = self._config.model_dir

        self._result: Optional[List[Primitive]] = None

    def _epsilon_by_frame(self, frame_id):
        return self.epsilon_final + (
            self.epsilon_start - self.epsilon_final
        ) * math.exp(-1.0 * frame_id / self.epsilon_decay)

    def get_five_items_from_pipeline(
        self,
        fr,
        state,
        reward_dic,
        seq,
        taskid,
        need_save=True,
        dataset_name="UNKNOWN",
        tag: str = "0",
    ):
        tryed_list = []
        epsilon = self._epsilon_by_frame(fr)
        pipeline_index = self.env.pipeline.get_index()
        has_num_nan, has_cat_nan = self.env.has_nan()

        action = -1
        step = Primitive()

        """predict action by epsilon-greedy"""
        logic_pipeline_id = self.env.pipeline.logic_pipeline_id
        curr_component = comp.lpipelines[logic_pipeline_id][pipeline_index]

        if curr_component == "ImputerNum":
            if has_num_nan:
                action, isModel = self.agent.act(
                    self.env.pipeline,
                    state,
                    curr_component,
                    tryed_list,
                    epsilon,
                    # taskid=taskid,
                )
                temp = comp.imputernums[action]
                step = deepcopy(temp)
            else:
                action = len(comp.imputernums)
                step = Primitive()

        elif curr_component == "ImputerCat":
            action = -1
            if has_cat_nan:
                step = ImputerCatPrim()
            else:
                step = Primitive()

        elif curr_component == "Encoder":
            if self.env.has_cat_cols():
                action, isModel = self.agent.act(
                    self.env.pipeline,
                    state,
                    curr_component,
                    tryed_list,
                    epsilon,
                    # taskid=taskid,
                )
                temp = comp.encoders[action]
                step = deepcopy(temp)
            else:
                action = len(comp.encoders)
                step = Primitive()

        elif curr_component in [
            "FeaturePreprocessing",
            "FeatureEngine",
            "FeatureSelection",
        ]:
            action, isModel = self.agent.act(
                self.env.pipeline,
                state,
                curr_component,
                tryed_list,
                epsilon,
                # taskid=taskid,
            )
            if curr_component == "FeaturePreprocessing":
                temp = comp.fpreprocessings[action]
                step = deepcopy(temp)
            elif curr_component == "FeatureEngine":
                temp = comp.fengines[action]
                step = deepcopy(temp)
            elif curr_component == "FeatureSelection":
                temp = comp.fselections[action]
                step = deepcopy(temp)

        """execute action"""
        step_result, err = self.env.step(step, has_timeout=False)
        tryed_list.append(action)
        repeat_time = 0

        """if execute fail, try again"""
        while step_result is None:
            if curr_component == "ImputerNum":
                if has_num_nan:
                    try:
                        action, isModel = self.agent.act(
                            self.env.pipeline,
                            state,
                            curr_component,
                            tryed_list,
                            epsilon,
                            # taskid=taskid,
                        )
                    except:
                        logger.error(f"error state: {state}")
                        raise RuntimeError("act failed")

                        # return

                    temp = comp.imputernums[action]
                    step = deepcopy(temp)
                else:
                    action = len(comp.imputernums)
                    step = Primitive()

            elif curr_component == "ImputerCat":  # imputercat
                action = -1
                if has_cat_nan:
                    step = ImputerCatPrim()
                else:
                    step = Primitive()

            elif curr_component == "Encoder":  # encoder
                if self.env.has_cat_cols():
                    action, isModel = self.agent.act(
                        self.env.pipeline,
                        state,
                        curr_component,
                        tryed_list,
                        epsilon,
                        # taskid=taskid,
                    )
                    temp = comp.encoders[action]
                    step = deepcopy(temp)
                else:
                    action = len(comp.encoders)
                    step = Primitive()

            elif curr_component in [
                "FeaturePreprocessing",
                "FeatureEngine",
                "FeatureSelection",
            ]:
                action, isModel = self.agent.act(
                    self.env.pipeline,
                    state,
                    curr_component,
                    tryed_list,
                    epsilon,
                    # taskid=taskid,
                )
                if curr_component == "FeaturePreprocessing":
                    temp = comp.fpreprocessings[action]
                    step = deepcopy(temp)
                elif curr_component == "FeatureEngine":
                    temp = comp.fengines[action]
                    step = deepcopy(temp)
                elif curr_component == "FeatureSelection":
                    temp = comp.fselections[action]
                    step = deepcopy(temp)

            if action in tryed_list:
                repeat_time += 1
                continue

            tryed_list.append(action)
            step_result, err = self.env.step(step, has_timeout=False)

        """get (st, r, st+1, done) for this execute"""
        state, reward, next_state, done = step_result
        seq.append(step.name)
        state = next_state

        """if done, evaluate and save result"""
        if done:
            self._result = self.env.pipeline.sequence

            with open(self._config.pipelines_file_name, "a") as f:
                f.write(
                    f"{tag}\t{dataset_name}\t{self.env.pipeline.sequence}\t{reward}\n"
                )

            self.end_time = self.env.end_time
            self.env.reset(
                taskid=taskid,
                default=False,
                metric=comp.metrics[0],
                predictor=comp.predictors[self.test_pred],
            )
            self.env.pipeline.logic_pipeline_id, _ = self.agent.act(
                self.env.pipeline,
                self.env.lpip_state,
                "LogicPipeline",
                epsilon=self._epsilon_by_frame(0),
            )
            if self.env.pipeline.taskid not in reward_dic:
                reward_dic[self.env.pipeline.taskid] = {
                    "reward": {},
                    "seq": {},
                    "time": {},
                }

            reward_dic[self.env.pipeline.taskid]["reward"][self.pre_fr] = reward
            reward_dic[self.env.pipeline.taskid]["seq"][self.pre_fr] = seq
            reward_dic[self.env.pipeline.taskid]["time"][self.pre_fr] = (
                self.end_time - self.start_time
            )

            if need_save:
                np.save(self._config.test_reward_dic_file_name, reward_dic)

        return state, reward_dic, seq, reward, done

    @property
    def result(self) -> List[Primitive]:
        if self._result is None:
            raise ValueError(
                "Pipeline has not been found yet. Please run inference first."
            )

        return self._result

    def inference(
        self, data_path, tag: str = "56000", dataset_name="UNKNOWN"
    ):  # -> tuple[Any | list[Any], Any]:
        self.agent.load_weights(self.outputdir, tag=tag)
        self.pre_fr = 0

        score = 0
        reward_dic = {}
        seq = []
        select_cl = 0

        if self._config.data_in_memory:
            taskid = -1  # dummy
            select_cl = comp.selected_prim.id
        else:
            datasetname = data_path.split("/")[-2]

            i = None
            for taskid in self._config.classification_task_dic:
                if (
                    datasetname
                    == self._config.classification_task_dic[taskid]["dataset"]
                ):
                    i = taskid

            if i is None:
                raise ValueError("Invalid i")

            for cl in comp.predictors:
                if cl.name == self._config.classification_task_dic[i]["model"]:
                    select_cl = cl

        self.start_time = time.time()
        self.env.reset(
            taskid=i,
            default=False,
            metric=comp.metrics[0],
            predictor=select_cl,
        )
        self.env.pipeline.logic_pipeline_id, _ = self.agent.act(
            self.env.pipeline,
            self.env.lpip_state,
            "LogicPipeline",
            epsilon=self._epsilon_by_frame(0),
        )

        state = self.env.get_state()

        reward = None
        for fr in range(self.pre_fr + 1, self.pre_fr + 7):
            state, reward_dic, seq, reward, done = self.get_five_items_from_pipeline(
                fr,
                state,
                reward_dic,
                seq,
                taskid=i,
                need_save=False,
                dataset_name=dataset_name,
                tag=tag,
            )

        if reward is None:
            raise ValueError("Invalid reward")

        score = reward

        return seq, score
