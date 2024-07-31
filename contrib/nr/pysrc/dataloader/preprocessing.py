import torch
from logger.logger import logger
import time


def libsvm_batch_preprocess(data: str, max_nfileds: int):
    """
    Preprocess a batch of data from LibSVM format.
    :param data: The data in LibSVM format.
    :param max_nfileds: nfileds
    :return: A dictionary with processed 'id', 'value', and 'y' tensors.
    """
    logger.debug(f"[Data Preprocessing]: Preprocessing started...")

    # Split data into lines and filter out any empty lines
    lines = []
    for line in data.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    # Initialize lists for ids, values, and labels
    ids_list = []
    values_list = []
    labels_list = []

    # Timing for line traversal and parsing
    parse_start_time = time.time()
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
    parse_end_time = time.time()

    # Timing for tensor creation
    tensor_start_time = time.time()
    nsamples = len(lines)
    feat_id = torch.zeros((nsamples, max_nfileds), dtype=torch.long)
    feat_value = torch.zeros((nsamples, max_nfileds), dtype=torch.float)
    y = torch.tensor(labels_list, dtype=torch.float)
    tensor_end_time = time.time()

    # Timing for tensor population
    populate_start_time = time.time()
    for i in range(nsamples):
        ids = torch.tensor(ids_list[i], dtype=torch.long)
        values = torch.tensor(values_list[i], dtype=torch.float)
        feat_id[i, :len(ids)] = ids
        feat_value[i, :len(values)] = values
    populate_end_time = time.time()

    # Log timings
    logger.debug(f"[Data Preprocessing, Timing]: Parsing time: {parse_end_time - parse_start_time:.4f} seconds")
    logger.debug(
        f"[Data Preprocessing, Timing]: Tensor creation time: {tensor_end_time - tensor_start_time:.4f} seconds")
    logger.debug(
        f"[Data Preprocessing, Timing]: Tensor population time: {populate_end_time - populate_start_time:.4f} seconds")

    logger.debug(f"[Data Preprocessing]: # {nsamples} data samples loaded successfully.")

    return {"id": feat_id, "value": feat_value, "y": y}


def libsvm_batch_preprocess_id_only(data: str, max_nfileds: int):
    """
    Preprocess a batch of data from a simplified format where the data contains only labels and IDs.
    :param data: The data in the format "label id id id...".
    :param max_nfileds: The maximum number of fields (IDs) per sample.
    :return: A dictionary with processed 'id', 'value', and 'y' tensors.
    """
    logger.debug(f"[Data Preprocessing]: Preprocessing started...")

    # Split data into lines and filter out any empty lines
    lines = []
    for line in data.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    # Initialize lists for ids and labels
    ids_list = []
    labels_list = []

    # Timing for line traversal and parsing
    parse_start_time = time.time()
    # Parse each line into ids and label
    for line in lines:
        columns = line.split()
        label = float(columns[0])
        ids = [int(id) for id in columns[1:]]

        ids_list.append(ids)
        labels_list.append(label)
    parse_end_time = time.time()

    # Timing for tensor creation
    tensor_start_time = time.time()
    nsamples = len(lines)
    feat_id = torch.zeros((nsamples, max_nfileds), dtype=torch.long)
    feat_value = torch.ones((nsamples, max_nfileds), dtype=torch.float)  # All values set to 1.0
    y = torch.tensor(labels_list, dtype=torch.float)
    tensor_end_time = time.time()

    # Timing for tensor population
    populate_start_time = time.time()
    for i in range(nsamples):
        ids = torch.tensor(ids_list[i], dtype=torch.long)
        feat_id[i, :len(ids)] = ids
    populate_end_time = time.time()

    # Log timings
    logger.debug(f"[Data Preprocessing, Timing]: Parsing time: {parse_end_time - parse_start_time:.4f} seconds")
    logger.debug(
        f"[Data Preprocessing, Timing]: Tensor creation time: {tensor_end_time - tensor_start_time:.4f} seconds")
    logger.debug(
        f"[Data Preprocessing, Timing]: Tensor population time: {populate_end_time - populate_start_time:.4f} seconds")

    logger.debug(f"[Data Preprocessing]: # {nsamples} data samples loaded successfully.")

    return {"id": feat_id, "value": feat_value, "y": y}


if __name__ == "__main__":
    from clients.client_socket import generate_dataset, generate_id_only_dataset
    import time

    begin_time = time.time()
    data = generate_id_only_dataset(1000000)
    libsvm_batch_preprocess_id_only(data, 10)
    print("Total execution time:", time.time() - begin_time)
