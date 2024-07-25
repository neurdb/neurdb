from logger.logger import logger
from typing import List
import argparse
from connection import NeurDBModelHandler
from app.handlers.setup import Setup
import numpy as np


def inference(
        model_name: str,
        inference_libsvm: str,
        args: argparse.Namespace,
        db: NeurDBModelHandler,
        model_id: int,
        batch_size: int,
        inf_batch_num: int
) -> List[np.ndarray]:
    s = Setup(model_name, inference_libsvm, args, db)
    response, err = s.inference(model_id, batch_size, inf_batch_num)
    if err is not None:
        logger.error(f"inference failed with error: {err}")
        return []
    logger.debug(f"inference done. response[0,:100]:")
    logger.debug(response[0][:100] if len(response[0]) >= 100 else response[0])

    return response
