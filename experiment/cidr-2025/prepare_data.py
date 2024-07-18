import psycopg2
from psycopg2 import extras
import argparse
import pandas as pd

DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'host': 'localhost',
    'port': '5432'
}


def connect_to_db() -> tuple:
    """
    Connect to the database
    :return connection and cursor
    """
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    return conn, cursor


def create_table() -> None:
    """
    Create the frappe table if it does not exist
    :return None
    """
    conn, cursor = connect_to_db()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS frappe ("
        "id SERIAL PRIMARY KEY,"
        "label INTEGER,"
        "feature1 INTEGER,"
        "feature2 INTEGER,"
        "feature3 INTEGER,"
        "feature4 INTEGER,"
        "feature5 INTEGER,"
        "feature6 INTEGER,"
        "feature7 INTEGER,"
        "feature8 INTEGER,"
        "feature9 INTEGER,"
        "feature10 INTEGER"
        ")"
    )
    conn.commit()
    conn.close()


def prepare_data(number_of_rows) -> None:
    """
    Prepare training data for the experiment
    :param number_of_rows: number of rows in frappe table
    :return None
    """
    conn, cursor = connect_to_db()
    cursor.execute('SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = \'frappe\')')
    table_exists = cursor.fetchone()[0]
    if not table_exists:
        raise Exception('Table frappe does not exist')

    cursor.execute('SELECT COUNT(*) FROM frappe')
    current_row_num = cursor.fetchone()[0]

    if current_row_num > number_of_rows:
        raise Exception(f'Number of rows in frappe table is {current_row_num}, greater than {number_of_rows}')

    # insert data into the frappe table
    data = pd.read_csv('dataset/frappe.csv')
    needed_rows = data.sample(n=number_of_rows - current_row_num)
    insert_query = "INSERT INTO frappe (label, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10) VALUES %s"
    extras.execute_values(cursor, insert_query, needed_rows.values)

    conn.commit()
    cursor.execute('SELECT COUNT(*) FROM frappe')
    print(f"Number of rows in frappe table: {cursor.fetchone()[0]}")
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_table", action="store_true", help="Create the frappe table if it does not exist")
    parser.add_argument("--num_rows", type=int, default=10000, help="Number of rows in the frappe table")
    args = parser.parse_args()
    if args.create_table:
        create_table()
    prepare_data(args.num_rows)
