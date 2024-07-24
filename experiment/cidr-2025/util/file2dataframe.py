"""
This script converts a libsvm file to a csv file.
"""
import argparse

import numpy as np
import pandas as pd

from neurdb.logger import configure_logging
from neurdb.logger import logger

configure_logging(None)


def _parse_libsvm_line(line: str, num_features: int) -> list:
    """
    Parse a line from a libsvm file
    :param line: a line from a libsvm file
    :return: a list [label, feature1, feature2, ...]
    """
    parts = line.strip().split(" ")
    label = int(parts[0])
    features = [0.0] * num_features
    for part in parts[1:]:
        index, value = part.split(":")
        features[int(index) - 1] = float(value)
    return [label] + features


def _read_libsvm_file(input_file: str) -> list:
    """
    Read a libsvm file
    :param input_file: path to the input file
    """
    with open(input_file, "r") as f:
        number_of_features = len(f.readline().strip().split(" ")) - 1
        f.seek(0)
        index = 0
        for line in f:
            if line.strip():
                yield _parse_libsvm_line(line, number_of_features)
                index += 1
                if index % 100000 == 0:
                    logger.debug(f"Read {index} lines...")


def libsvm2csv(input_file: str) -> pd.DataFrame:
    logger.debug(f"Converting libsvm file {input_file} to dataframe...")
    with open(input_file, "r") as f:
        number_of_features = len(f.readline().strip().split(" ")) - 1
    col_names = ["label"] + [f"feature{i}" for i in range(1, number_of_features + 1)]
    df = pd.DataFrame(_read_libsvm_file(input_file), columns=col_names)
    logger.debug(f"Number of samples: {len(df)}")
    logger.debug("Done converting...")
    return df


def npy2csv(input_file: str) -> pd.DataFrame:
    logger.debug(f"Converting npy file {input_file} to dataframe...")
    data = np.load(input_file)
    columns = ["label"] + [f"feature{i}" for i in range(1, data.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    logger.debug(f"Number of samples: {len(df)}")
    logger.debug("Done converting...")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="Type of input file: libsvm or npy")
    parser.add_argument("--input_file", type=str, help="Input libsvm file")
    args = parser.parse_args()
    input_path = args.input_file
    libsvm2csv(input_path)
