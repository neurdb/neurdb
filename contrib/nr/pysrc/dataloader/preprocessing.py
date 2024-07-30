import torch
from logger.logger import logger


def libsvm_batch_preprocess(data: str, max_nfileds: int):
    """
    Preprocess a batch of data from LibSVM format.
    :param data: The data in LibSVM format.
    :param max_nfileds: nfileds
    :return: A dictionary with processed 'id', 'value', and 'y' tensors.
    """
    logger.debug(f"[Data Preprocessing]: Preprocessing started...")
    data = data.split("\n")

    sample_lines = 0
    ids_list = []
    values_list = []
    labels_list = []

    for line in data:
        if not line:
            continue  # skip empty lines
        columns = line.strip().split(" ")
        pairs = [list(map(int, pair.split(":"))) for pair in columns[1:]]
        ids, values = zip(*pairs) if pairs else ([], [])
        ids_list.append(ids)
        values_list.append(values)
        labels_list.append(float(columns[0]))
        sample_lines += 1

    nsamples = sample_lines
    feat_id = torch.zeros((nsamples, max_nfileds), dtype=torch.long)
    feat_value = torch.zeros((nsamples, max_nfileds), dtype=torch.float)
    y = torch.tensor(labels_list, dtype=torch.float)
    # logger.debug(f"[Data Preprocessing]: Creating tensors...")

    for i in range(nsamples):
        try:
            ids = ids_list[i]
            values = values_list[i]
            feat_id[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            feat_value[i, :len(values)] = torch.tensor(values, dtype=torch.float)
        except Exception as e:
            logger.debug(f"[Data Preprocessing]: Incorrect data format in sample {i}! Error: {e}")
    logger.debug(f"[Data Preprocessing]: # {nsamples} data samples loaded successfully.")

    return {"id": feat_id, "value": feat_value, "y": y}
