from logger.logger import logger
import argparse
from connection import NeurDBModelHandler
from app.handlers.setup import Setup


def train(
        model_name: str,
        training_libsvm: str,
        args: argparse.Namespace,
        db: NeurDBModelHandler,
        batch_size: int,
) -> int:
    s = Setup(model_name, training_libsvm, args, db)

    model_id, err = s.train(batch_size)
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    print(f"train done. model_id: {model_id}")
    return model_id
