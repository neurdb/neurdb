import psycopg2
from psycopg2 import extras
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import argparse
import pandas as pd
from util.file2dataframe import libsvm2csv, npy2csv
from neurdb.logger import configure_logging
from neurdb.logger import logger

configure_logging(None)

DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "host": "127.0.0.1",
    "port": "5432",
}

register_adapter(np.int64, AsIs)


def _connect_to_db() -> tuple:
    """
    Connect to the database
    :return connection and cursor
    """
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    return conn, cursor


def _create_table(cursor, conn, table_name: str, data: pd.DataFrame):
    """
    Create a table in the database
    :param cursor: current cursor
    :param conn: current connection
    :param table_name: name of the table
    :param data: pandas DataFrame (this function only need the columns, not the data)
    :return None
    """

    # Drop the old table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()

    # Create the new table
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    query += "id SERIAL PRIMARY KEY,"
    query += "label INTEGER,"
    for i in range(1, len(data.columns)):
        data_type = "INTEGER" if data[data.columns[i]].dtype == "int64" else "FLOAT"
        query += f"feature{i} {data_type},"
    query = query[:-1] + ")"
    cursor.execute(query)
    conn.commit()


def create_table_for_dataset(data: pd.DataFrame, table_name: str, random_state: int):
    """
    We create six tables for each dataset:
    - raw: the original data, this contains 50% of the data
    - test1, 2, 3, 4, 5: these tables contain 10% of the data each
    :param data: pandas DataFrame
    :param table_name: name of the table
    :param random_state: random state used to shuffle the data
    :return None
    """
    logger.debug(f"Creating tables for dataset {table_name}...")
    conn, cursor = _connect_to_db()
    data = data.sample(frac=1, random_state=random_state)
    datasets = {
        f"{table_name}_raw": data[: int(0.5 * len(data))],
        f"{table_name}_test1": data[int(0.5 * len(data)): int(0.6 * len(data))],
        f"{table_name}_test2": data[int(0.6 * len(data)): int(0.7 * len(data))],
        f"{table_name}_test3": data[int(0.7 * len(data)): int(0.8 * len(data))],
        f"{table_name}_test4": data[int(0.8 * len(data)): int(0.9 * len(data))],
        f"{table_name}_test5": data[int(0.9 * len(data)):],
    }

    for table_name, data in datasets.items():
        _create_table(cursor, conn, table_name, data)

    insert_query = (
        f"INSERT INTO {table_name} "
        f"(label, {', '.join([f'feature{i}' for i in range(1, len(data.columns))])}) "
        f"VALUES %s"
    )
    extras.execute_values(cursor, insert_query, data.values)
    conn.commit()

    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    logger.debug(
        f"Successfully created table {table_name} with {cursor.fetchone()[0]} rows, random state {random_state}..."
    )
    conn.close()
    logger.debug("Done creating tables...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="Random state to shuffle the dataset"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the dataset file, e.g., /path/to/data.csv"
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="csv",
        help="Type of the input file, csv, npy or libsvm"
    )
    args = parser.parse_args()
    input_file = args.input_file
    dataset_name = args.dataset_name
    file_type = args.file_type

    if args.random_state is None:
        random_state = 10
    else:
        random_state = args.random_state

    if file_type == "csv":
        data = pd.read_csv(input_file)
    elif file_type == "npy":
        data = npy2csv(input_file, "temp.csv")
    elif file_type == "libsvm":
        data = libsvm2csv(input_file, "temp.csv")
    else:
        raise ValueError("Invalid file type. Please use csv, npy or libsvm.")
    create_table_for_dataset(data, dataset_name, random_state)
    logger.debug(f"Preparation for dataset {dataset_name} is done.")
