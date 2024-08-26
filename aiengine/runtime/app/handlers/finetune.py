from logger.logger import logger
import argparse
from connection import NeurDBModelHandler
from app.handlers.setup import Setup
from dataloader.steam_libsvm_dataset import StreamingDataSet


async def finetune(
    model_name: str,
    finetune_libsvm: StreamingDataSet,
    args: argparse.Namespace,
    db: NeurDBModelHandler,
    model_id: int,
    epoch: int,
    train_batch_num: int,
    eva_batch_num: int,
    test_batch_num: int,
) -> int:
    s = Setup(model_name, finetune_libsvm, args, db)

    model_id, err = await s.finetune(
        model_id,
        start_layer_id=5,
        epoch=epoch,
        train_batch_num=train_batch_num,
        eva_batch_num=eva_batch_num,
        test_batch_num=test_batch_num,
    )
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    print(f"finetune done. model_id: {model_id}")
    return model_id
