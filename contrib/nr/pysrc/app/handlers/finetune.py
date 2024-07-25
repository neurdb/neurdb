from logger.logger import logger
import argparse
from connection import NeurDBModelHandler
from app.handlers.setup import Setup


def finetune(
        model_name: str,
        finetune_libsvm: str,
        args: argparse.Namespace,
        db: NeurDBModelHandler,
        model_id: int,
        batch_size: int,
        epochs: int,
        train_batch_num: int,
        eva_batch_num: int,
        test_batch_num: int
) -> int:
    s = Setup(model_name, finetune_libsvm, args, db)

    model_id, err = s.finetune(model_id, batch_size, start_layer_id=5, epochs=epochs,
                               train_batch_num=train_batch_num,
                               eva_batch_num=eva_batch_num,
                               test_batch_num=test_batch_num)
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    print(f"finetune done. model_id: {model_id}")
    return model_id