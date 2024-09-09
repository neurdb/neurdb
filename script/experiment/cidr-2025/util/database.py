"""
This script provides some simple functions to connect to a database.
"""

import psycopg2
from config import DB_CONFIG


def connect_db() -> tuple:
    """
    Connect to the database
    :return connection and cursor
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    return conn, cursor
