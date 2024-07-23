"""
This script provides some simple functions to connect to a database.
"""
import psycopg2

from ..config import Configuration


def connect_db() -> tuple:
    """
    Connect to the database
    :return connection and cursor
    """
    conn = psycopg2.connect(**Configuration.DB_PARAMS)
    cursor = conn.cursor()
    return conn, cursor
