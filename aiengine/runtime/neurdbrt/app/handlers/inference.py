import argparse
from typing import List

import numpy as np
from neurdbrt.app.handlers.setup import Setup
from neurdbrt.repo import ModelRepository
from neurdbrt.dataloader.stream_libsvm_dataset import StreamingDataSet
from neurdbrt.log import logger


async def inference(
    model_name: str,
    inference_libsvm: StreamingDataSet,
    args: argparse.Namespace,
    db: ModelRepository,
    model_id: int,
    inf_batch_num: int,
) -> List[np.ndarray]:
    s = Setup(model_name, inference_libsvm, args, db)
    response, err = await s.inference(model_id, inf_batch_num)
    if err is not None:
        logger.error(f"inference failed with error: {err}")
        return []
    logger.debug(f"inference done. response")

    return response
