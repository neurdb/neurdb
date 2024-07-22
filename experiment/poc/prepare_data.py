import psycopg2
from psycopg2 import extras
import argparse

NUM_ROWS = 150

# Database connection parameters
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "host": "localhost",
    "port": "5432",
}


def connect_to_db() -> tuple:
    """
    Connect to the database
    :return: connection and cursor
    """
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    return conn, cursor


def prepare(number_of_rows: int = 150) -> None:
    """
    Prepare data for the experiment
    :param number_of_rows: number of rows in iris table
    :return: None
    """
    conn, cursor = connect_to_db()
    # check if iris table exists
    cursor.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'iris')"
    )
    table_exists = cursor.fetchone()[0]
    if not table_exists:
        raise Exception("Table iris does not exist")

    # get number of rows in iris table
    cursor.execute("SELECT COUNT(*) FROM iris")
    current_row_num = cursor.fetchone()[0]

    if current_row_num > number_of_rows:
        raise Exception(
            f"Number of rows in iris table is {current_row_num}, greater than {number_of_rows}"
        )

    cursor.execute("SELECT * FROM iris")
    existing_rows = cursor.fetchall()
    insert_query = (
        "INSERT INTO iris (sepal_l, sepal_w, petal_l, petal_w, class) VALUES %s"
    )

    batch_size = len(existing_rows)
    batch_num = (number_of_rows - current_row_num) // batch_size
    rest = (number_of_rows - current_row_num) % batch_size

    for _ in range(batch_num):
        extras.execute_values(cursor, insert_query, existing_rows)
        conn.commit()
    extras.execute_values(cursor, insert_query, existing_rows[:rest])

    # check the number of rows after insertion
    cursor.execute("SELECT COUNT(*) FROM iris")
    current_row_num = cursor.fetchone()[0]
    print(f"Number of rows in iris table: {current_row_num}")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for the experiment")
    parser.add_argument(
        "--num_rows", type=int, default=150, help="Number of rows in iris table"
    )
    args = parser.parse_args()
    NUM_ROWS = args.num_rows
    prepare(NUM_ROWS)
