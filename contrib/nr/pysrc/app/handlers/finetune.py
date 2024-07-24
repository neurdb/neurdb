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
        batch_per_epoch: int
) -> int:
    s = Setup(model_name, finetune_libsvm, args, db)

    model_id, err = s.finetune(model_id, batch_size, start_layer_id=5, epochs=epochs, batch_per_epoch=batch_per_epoch)
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    print(f"finetune done. model_id: {model_id}")
    return model_id
