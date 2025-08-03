import env
from ctxpipe.dataset import Dataset

env.init()

import gc
import json
import os
import sys
import time
import traceback
from glob import glob

from loguru import logger

import config
import util
from auto_pipeline.config import default_config as conf
from .ctxpipe.info import Info
from .ctxpipe.pipegen import PipelineGenerator
# from ctxpipe.stats import Stats, init_stats_db


class CollectionBuilder:
    def __init__(self, info: Info) -> None:
        self._info = info

    def build(self):
        for dataset_dir in glob(os.path.join(self._info.dataset_prefix, "*", "")):
            self._parse_task_info(dataset_dir)

    def _parse_task_info(self, dataset_dir: str):
        ds = dataset_dir[:-1].split("/")[-1]
        ds_csv = os.path.join(dataset_dir, "data.csv")
        logger.info(f"processing dataset {ds} at {ds_csv}")

        info = util.read_json(os.path.join(dataset_dir, "info.json"))
        label_name = info["label"]
        logger.debug(f"label_name: {label_name}")

        with open(os.path.join(ds_csv)) as f:
            meta_line = f.readline()[:-1]

        logger.debug(f"meta_line: {meta_line}")

        try:
            label_index = meta_line.split(",").index(label_name)
            logger.debug(f"label_index: {label_index}")
        except ValueError:
            logger.warning(f"label column not found in dataset {ds}. skip")
            return

        self._update_files(
            dataset=Dataset(ds, ds_csv, label_index),
            predictor_name=env.eval_predictor_name,
        )

    def _update_files(self, dataset: Dataset, predictor_name: str):
        """
        update adds information of current task and dataset in information
        files if not exist.

        NOTE: ctasks = classification tasks
        """
        dataset_info = util.read_json(self._info.dataset_info_path)
        ctasks = util.read_json(self._info.task_info_path)
        test_index = util.read_json(self._info.task_index_path)

        ### check if current task is in saved information file
        task_id = str(len(ctasks))
        test_index.append(task_id)
        task_name = (
            dataset.path.split("/")[-2]
            + "_"
            + predictor_name
            + "_"
            + str(dataset.label_column_id)
        )

        exist_task = False
        for task in ctasks:
            if task_name == ctasks[task]["task_name"]:
                exist_task = True
                break

        if not exist_task:
            ctasks[task_id] = dataset.make_task_info(predictor_name, task_name)

        if dataset.name not in dataset_info:
            dataset_info[dataset.name] = dataset.info

        ### update information files
        util.write_json(self._info.dataset_info_path, dataset_info)
        util.write_json(self._info.task_info_path, ctasks)
        util.write_json(self._info.task_index_path, test_index)


class Setup:
    def __init__(self, info: Info) -> None:
        self._info = info
        # init_stats_db(self._info.stats_db_file_path)

    def _init_info_path(self):
        config.set_info(self._info)
        config.init()

    def evaluate_pipeline(self, start: int, end: int = -1, dry_run=False):
        self._init_info_path()

        self._reset_failed_file()
        self._reset_done_file()

        datasets = self._get_datasets()
        ctasks = self._get_ctasks()

        if end == -1:
            end = start + 1

        test_model_iterations = list(range(start, end + 1, conf.checkpoint_interval))
        for it in test_model_iterations:
            logger.info(f"evaluating on model iteration {it}...")
            self._evaluate_iteration(it, datasets, ctasks, dry_run)
            logger.info(f"evaluating on model iteration {it} done")

    def _reset_failed_file(self):
        with open(self._info.done_file_path, "w") as f:
            f.write(
                "iteration\tnotebook_path\tdataset_path\tlabel_index\tmodel\treason\ttraceback\n"
            )

    def _reset_done_file(self):
        with open(self._info.done_file_path, "w") as f:
            f.write("iteration\tnotebook_path\tdataset_path\tlabel_index\tmodel\n")

    def _get_datasets(self) -> dict:
        result: dict = util.read_json(self._info.dataset_info_path)

        logger.info("datasetinfo loaded")
        logger.debug(f"datasetinfo.(first): {next(iter(result.items()))}")

        return result

    def _get_ctasks(self):
        with open(self._info.task_info_path, "r") as f:
            classification_task_dic: dict = json.load(f)

            result = {}
            for v in classification_task_dic.values():
                result[v["dataset"]] = v
            logger.info("classification_dataset_dic loaded")
            logger.debug(
                f"classification_dataset_dic.(first): {next(iter(result.items()))}"
            )

        return result

    def _evaluate_iteration(
        self,
        iteration: int,
        datasets: dict,
        ctasks: dict,
        dry_run: bool,
    ):
        model_tag = f"ctx_{iteration}"
        logger.info(f"model_tag={model_tag}")

        for _, info in datasets.items():
            dataset_name = info["dataset"]
            # if Stats.select().where(Stats.notebook == notebook).exists():
            #     logger.warning(f"notebook {notebook} executed")
            #     continue

            # if (
            #     not dry_run
            #     and Stats.select()
            #     .where((Stats.dataset == dataset_name) & (Stats.iteration == iteration))
            #     .exists()
            # ):
            #     logger.warning(f"dataset {dataset_name} executed")
            #     continue

            try:
                csv_file = ctasks[dataset_name]["csv_file"]
            except KeyError:
                logger.warning(f"dataset {dataset_name} not found")
                continue

            dataset_path = f"{self._info.dataset_prefix}/{dataset_name}/{csv_file}"
            label_index = info["index"]

            logger.info(
                "{},{},{},{}",
                dataset_name,
                dataset_path,
                label_index,
                env.eval_predictor_name,
            )

            self._do_evaluate(
                dataset=Dataset(dataset_name, dataset_path, label_index),
                model=env.eval_predictor_name,
                dry_run=dry_run,
                model_tag=model_tag,
            )

            gc.collect()

    def _do_evaluate(
        self,
        dataset: Dataset,
        model: str,  # Only supports those in "support_model".
        dry_run: bool = False,  # If true, do not write result to files
        model_tag: str = "56000",
    ):
        f = None
        done_f = None

        iteration = int(model_tag.split("_")[1])

        if not dry_run:
            f = open(self._info.failed_file_path, "a")
            done_f = open(self._info.done_file_path, "a")

        try:
            pg = PipelineGenerator(dataset, model_tag)

            start_time = time.time()
            ###
            pg.generate()
            ###
            end_time = time.time()

            stats = pg.output()
            stats.execution_time = end_time - start_time
            stats.save()

            logger.info(
                "dataset {} at {} finished",
                dataset.name,
                dataset.path,
            )
            logger.info(stats)

            if done_f:
                done_f.write(
                    f"{iteration}\t{dataset.name}\t"
                    f"{dataset.path}\t{dataset.label_column_id}"
                    f"\t{model}\n"
                )

        except Exception as e:
            print(traceback.format_exc())
            traceback_str = traceback.format_exc().replace("\n", "\\n")
            if f:
                f.write(
                    f"{iteration}\t{dataset.name}\t"
                    f"{dataset.path}\t{dataset.label_column_id}"
                    f"\t{model}\t{str(e)}\t{traceback_str}\n"
                )

        finally:
            if f:
                f.close()
            if done_f:
                done_f.close()


def evaluate(info: Info, start: int, end: int = -1, dry_run: bool = False):
    setup = Setup(info)

    """Run this first to create information files"""
    CollectionBuilder(info).build()

    """Then *comment above line* and evaluate pipeline search"""
    setup.evaluate_pipeline(start=start, end=end, dry_run=dry_run)


def evaluate_scalability():
    evaluate(
        Info(
            aipipe_core_prefix=f"{conf.exp_dir}/scal/aipipe",
            result_prefix=f"{conf.exp_dir}/scal/result",
            dataset_prefix=f"data/scalability_test",
        ),
        start=32000,
    )


def evaluate_on_diffprep_dataset():
    evaluate(
        Info(
            aipipe_core_prefix=f"{conf.exp_dir}/aipipe",
            result_prefix=f"{conf.exp_dir}/result",
            dataset_prefix=f"data/diffprep_dataset",
        ),
        start=int(sys.argv[1]),
        end=int(sys.argv[2]),
    )


if __name__ == "__main__":
    evaluate_on_diffprep_dataset()
    # evaluate_scalability()
