from loguru import logger
from ctxpipe.agentman import AgentManager
from ctxpipe.dataset import Dataset

# from ctxpipe.stats import Stats


class PipelineGenerator:
    def __init__(
        self,
        dataset: Dataset,
        model_tag="56000",
    ):
        self._dataset = dataset
        self._model_tag = model_tag
        self._agentman = AgentManager()

    def generate(self):
        """
        generate_pipeline genertes AIPipe for the current dataset.
        """
        print("\033[0;33;40mstart run AIPipe:\033[0m")
        self.ai_sequence, self.ml_score = self._agentman.inference(
            self._dataset, self._model_tag
        )
        print("\033[0;32;40msucceed\033[0m")
        print("\n")

    @property
    def result(self):
        return self._agentman.result

    def output(self):
        """
        output outputs the results to stdout and stats DB.
        """
        logger.info(
            "inference done. dataset={dataset}, acc={acc}",
            dataset=self._dataset.name,
            acc=self.ml_score,
        )

        # trained_step = int(self._model_tag.split("_")[1])

        # stats = Stats.create(
        #     iteration=trained_step,
        #     notebook=self._dataset.name,
        #     dataset=self._dataset.name,
        #     hipipe_acc=None,
        #     aipipe_acc=self.ml_score,
        # )
        # return stats
