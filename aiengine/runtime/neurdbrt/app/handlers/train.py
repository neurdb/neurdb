import argparse
from typing import List

from neurdbrt.app.handlers.setup import Setup
from neurdbrt.dataloader.stream_libsvm_dataset import StreamingDataSet
from neurdbrt.log import logger
from neurdbrt.repo import ModelRepository


async def train(
    model_name: str,
    training_libsvm: StreamingDataSet,
    args: argparse.Namespace,
    db: ModelRepository,
    epoch: int,
    train_batch_num: int,
    eval_batch_num: int,
    test_batch_num: int,
    features: List[str],
    target: str,
) -> int:
    s = Setup(model_name, training_libsvm, args, db)

    model_id, err = await s.train(
        epoch, train_batch_num, eval_batch_num, test_batch_num
    )

    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1
    print(f"train done. model_id: {model_id}")
    s.register_model(model_id, "armnet", features, target)

    return model_id
