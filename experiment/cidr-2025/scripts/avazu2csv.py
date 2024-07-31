"""
This script combine three avazu npy files into one csv file.
"""

import argparse
import os

import pandas as pd
import numpy as np


def avazu2csv(input_files: list, output_file: str):
    """
    Combine three avazu npy files into one csv file.
    :param input_files: list of paths to the input files
    :param output_file: path to the output file
    """
    data = np.concatenate([np.load(file) for file in input_files])
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine three avazu npy (train, val, split) files into one csv file.")
    parser.add_argument("--avazue_path", help="path to the directory containing the avazu npy files")
    parser.add_argument("--output_file", help="path to the output file")
    args = parser.parse_args()

    train_path = os.path.join(args.avazue_path, "train.npy")
    val_path = os.path.join(args.avazue_path, "val.npy")
    split_path = os.path.join(args.avazue_path, "test.npy")
    paths = [train_path, val_path, split_path]

    avazu2csv(paths, args.output_file)
