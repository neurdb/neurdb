import csv
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch


def set_global_seed(seed=2024):
    """
    Set the random seed for all RNGs to ensure reproducibility.

    Args:
        seed (int): The seed value to use (default: 42).
    """
    # Python's random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Sets seed for CUDA RNG
    torch.cuda.manual_seed_all(
        seed
    )  # Sets seed for all CUDA devices (if multiple GPUs)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True  # Makes CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Disables benchmarking for reproducibility


def read_sql_file(filename, encoding="utf-8") -> str:
    """Read SQL file, remove comments, and return a list of sql statements as a string"""
    with open(filename, encoding=encoding) as f:
        file = f.read()
    statements = file.split("\n")
    return "\n".join(filter(lambda line: not line.startswith("--"), statements))


def generate_histogram_csv(data, output_path):
    rows = []
    for entry in data:
        table = entry["table"]
        column = entry["column"]
        freq = entry["freq"]
        bins = entry["bins"]

        # Convert frequency to hexadecimal string
        freq_hex = freq.tobytes().hex()

        # Combine table and column name
        table_alias = "".join([tok[0] for tok in table.split("_")])
        if table == "movie_info_idx":
            table_alias = "mi_idx"
        table_column = f"{table_alias}.{column}"

        # Prepare the row
        row = {
            "table": table,
            "column": column,
            "freq": freq_hex,
            "bins": str(bins),
            "table_column": table_column,
        }
        rows.append(row)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def get_hist_file(hist_path, bin_number=50):
    hist_file = pd.read_csv(hist_path)
    for i in range(len(hist_file)):
        freq = hist_file["freq"][i]
        freq_np = np.frombuffer(bytes.fromhex(freq), dtype=float)
        hist_file["freq"][i] = freq_np

    table_column = []
    for i in range(len(hist_file)):
        table = hist_file["table"][i]
        col = hist_file["column"][i]
        table_alias = "".join([tok[0] for tok in table.split("_")])
        if table == "movie_info_idx":
            table_alias = "mi_idx"
        combine = ".".join([table_alias, col])
        table_column.append(combine)
    hist_file["table_column"] = table_column

    for rid in range(len(hist_file)):
        hist_file["bins"][rid] = [
            int(i) for i in hist_file["bins"][rid][1:-1].split(" ") if len(i) > 0
        ]

    if bin_number != 50:
        hist_file = re_bin(hist_file, bin_number)

    return hist_file


# However, the number of bins for each column is not constant. To
# unify encoding, we need to re-group the histograms for all columns
# to the same 𝑁 number of bins.
def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file["freq"][i]
        bins = freq2bin(freq, target_number)
        hist_file["bins"][i] = bins
    return hist_file


# linear interpolation
def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq) - 1

    step = 1.0 / target_number
    mini = 0
    while freq[mini + 1] == 0:
        mini += 1
    pointer = mini + 1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi + 1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1

    if len(res_pos) == target_number:
        res_pos.append(maxi)

    return res_pos


def get_job_table_sample(workload_file_name, num_materialized_samples=1000):
    tables = []
    samples = []

    # Load queries
    with open(workload_file_name + ".csv", "r") as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter="#"))
        for row in data_raw:
            tables.append(row[0].split(","))

            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)

    print("Loaded queries with len ", len(tables))

    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(workload_file_name + ".bitmaps", "rb") as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder="little")
            bitmaps = np.empty(
                (num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8
            )
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")
    table_sample = []
    for ts, ss in zip(tables, samples):
        d = {}
        for t, s in zip(ts, ss):
            tf = t.split(" ")[0]  # remove alias
            d[tf] = s
        table_sample.append(d)

    return table_sample


def save_per_train_plot_data(
    embeddings, latencies, predictions, query_ids, output_file
):
    """
    Save embeddings, latencies, predictions, and query IDs to a single .npz file.

    Parameters:
    - embeddings: numpy array of embeddings
    - latencies: numpy array of true latencies
    - predictions: numpy array of predicted latencies
    - query_ids: list of query identifiers
    - output_file: path to save the .npz file
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save all data to a single .npz file
    np.savez(
        output_file,
        embeddings=embeddings,
        latencies=latencies,
        predictions=predictions,
        query_ids=np.array(query_ids, dtype=object),  # Convert list to array for saving
    )
    print(f"All data saved to {output_file}")


def load_per_train_plot_data(input_file):
    """
    Load embeddings, latencies, predictions, and query IDs from a single .npz file.

    Parameters:
    - input_file: path to the .npz file

    Returns:
    - embeddings: numpy array of embeddings
    - latencies: numpy array of true latencies
    - predictions: numpy array of predicted latencies
    - query_ids: list of query identifiers
    """
    # Load data from the .npz file
    data = np.load(
        input_file, allow_pickle=True
    )  # allow_pickle=True for object arrays (query_ids)

    embeddings = data["embeddings"]
    latencies = data["latencies"]
    predictions = data["predictions"]
    query_ids = data["query_ids"].tolist()  # Convert back to list from object array

    print(f"Data loaded from {input_file}:")
    print(f" - Embeddings shape: {embeddings.shape}")
    print(f" - Latencies shape: {latencies.shape}")
    print(f" - Predictions shape: {predictions.shape}")
    print(f" - Query IDs length: {len(query_ids)}")

    return embeddings, latencies, predictions, query_ids


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def save_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(text)
        f.write("\n")
    os.replace(path + ".tmp", path)
    return path
