"""
This scrit puts a csv file into a table in a database.
"""

import argparse

import pandas as pd
from util.database import connect_db

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Put a csv file into a table in a database."
    )
    parser.add_argument("--csv_file", help="Path to the csv file")
    parser.add_argument("--table_name", help="Name of the table")
    args = parser.parse_args()
    if args.table_name is None:
        raise ValueError("--table_name has to be provided.")

    pd = pd.read_csv(args.csv_file)
    # count the number of columns
    num_columns = len(pd.columns)

    # first column is the label, the rest are feature1, feature2, ...
    conn, cursor = connect_db()
    feature_columns = ", ".join([f"feature{i} INT" for i in range(1, num_columns)])
    create_table_query = (
        f"CREATE TABLE {args.table_name} (label INT, {feature_columns})"
    )
    cursor.execute(create_table_query)
    conn.commit()

    with open(args.csv_file, "r") as f:
        cursor.copy_from(
            f,
            args.table_name,
            sep=",",
            columns=[f"label"] + [f"feature{i}" for i in range(1, num_columns)],
        )
    conn.commit()
    cursor.close()
    conn.close()
