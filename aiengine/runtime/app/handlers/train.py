from logger.logger import logger
import argparse
from connection import NeurDBModelHandler
from app.handlers.setup import Setup
from dataloader.steam_libsvm_dataset import StreamingDataSet
from typing import List


async def train(
    model_name: str,
    training_libsvm: StreamingDataSet,
    args: argparse.Namespace,
    db: NeurDBModelHandler,
    epoch: int,
    train_batch_num: int,
    eval_batch_num: int,
    test_batch_num: int,
    features: List[str],
    target: str,
) -> int:
    s = Setup(model_name, training_libsvm, args, db)

    model_id, err = await s.train(epoch, train_batch_num, eval_batch_num, test_batch_num)

    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1
    print(f"train done. model_id: {model_id}")
    s.register_model(model_id, "armnet", features, target)

    return model_id
