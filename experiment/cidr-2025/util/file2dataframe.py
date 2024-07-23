"""
This script converts a libsvm file to a csv file.
"""
import argparse

import numpy as np
import pandas as pd

from neurdb.logger import configure_logging
from neurdb.logger import logger

configure_logging(None)


def libsvm2csv(input_file: str, output_file: str) -> pd.DataFrame:
    logger.debug(f"Converting libsvm file {input_file} to csv file {output_file}...")
    with open(input_file, "r") as f:
        lines = f.readlines()
    data = []
    line = lines[0].strip().split(" ")
    feature_names = [f"feature{i}" for i in range(1, len(line))]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ")
        label = int(parts[0])  # an integer
        features = parts[1:]
        features = [int(feature.split(":")[0]) for feature in features]
        data.append((label, features))

    df = pd.DataFrame()
    df["label"] = [label for label, _ in data]
    for i, feature_name in enumerate(feature_names):
        df[feature_name] = [features[i] for _, features in data]
    logger.debug(f"Number of samples: {len(df)}...")
    logger.debug("Done converting...")
    return df


def npy2csv(input_file: str, output_file: str) -> pd.DataFrame:
    logger.debug(f"Converting npy file {input_file} to csv file {output_file}...")
    data = np.load(input_file)
    columns = ["label"] + [f"feature{i}" for i in range(1, data.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    logger.debug(f"Number of samples: {len(df)}")
    logger.debug("Done converting...")
    df.to_csv(output_file, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="Type of input file: libsvm or npy")
    parser.add_argument("--input_file", type=str, help="Input libsvm file")
    parser.add_argument("--output_file", type=str, help="Output csv file")
    args = parser.parse_args()
    input_path = args.input_file
    output_path = args.output_file
    libsvm2csv(input_path, output_path)
