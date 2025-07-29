from config import default_agent_config, default_env_config
from ctxpipe.agent.dqn import Agent
from ctxpipe.dataset import Dataset
from ctxpipe.env.enviroment import Environment
from ctxpipe.tester import Tester
from ctxpipe.trainer import Trainer


class AgentManager:
    def __init__(self):
        """
        data_path: the path of the dataset

        model is saved in information files.
        """
        self.agent = Agent(default_agent_config)
        self.env = Environment(default_env_config, train=False)
        self.tester = Tester(self.agent, self.env, 0, default_agent_config)

    def train(self, resume_from=0):
        self.trainer = Trainer(self.agent, self.env, 0, default_agent_config)
        self.trainer.train(pre_fr=resume_from)

    def inference(self, dataset: Dataset, tag):
        return self.tester.inference(dataset.path, tag, dataset.name)

    @property
    def result(self):
        return self.tester.result
