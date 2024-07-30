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

    # Split data into lines and filter out any empty lines
    lines = [line.strip() for line in data.split("\n") if line.strip()]

    # Initialize lists for ids, values, and labels
    ids_list = []
    values_list = []
    labels_list = []

    # Parse each line into ids, values, and label
    for line in lines:
        columns = line.split()
        label = float(columns[0])
        pairs = [pair.split(":") for pair in columns[1:]]
        ids = [int(pair[0]) for pair in pairs]
        values = [float(pair[1]) for pair in pairs]

        ids_list.append(ids)
        values_list.append(values)
        labels_list.append(label)

    nsamples = len(lines)
    feat_id = torch.zeros((nsamples, max_nfileds), dtype=torch.long)
    feat_value = torch.zeros((nsamples, max_nfileds), dtype=torch.float)
    y = torch.tensor(labels_list, dtype=torch.float)

    for i in range(nsamples):
        ids = ids_list[i]
        values = values_list[i]
        feat_id[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        feat_value[i, :len(values)] = torch.tensor(values, dtype=torch.float)

    logger.debug(f"[Data Preprocessing]: # {nsamples} data samples loaded successfully.")

    return {"id": feat_id, "value": feat_value, "y": y}
