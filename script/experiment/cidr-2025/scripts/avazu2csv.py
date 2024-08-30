"""
This script combine three avazu npy files into one csv file.
"""

import argparse
import os

import numpy as np
import pandas as pd


def avazu2csv(input_file: str, output_file: str):
    """
    Combine three avazu npy files into one csv file.
    :param input_files: list of paths to the input files
    :param output_file: path to the output file
    """
    data = np.load(input_file)
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine three avazu npy (train, val, split) files into one csv file."
    )
    parser.add_argument(
        "--avazue_path", help="path to the directory containing the avazu npy file"
    )
    parser.add_argument("--output_file", help="path to the output file")
    args = parser.parse_args()

    avazu2csv(args.avazue_path, args.output_file)
