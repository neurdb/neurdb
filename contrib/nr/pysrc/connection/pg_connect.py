import psycopg2
import pandas as pd
from utils.parse_sql import parse_conditions
from sklearn.model_selection import train_test_split
import torch
from collections import namedtuple

LoadedDataset = namedtuple(
    "Dataset", ["X_train", "X_test", "y_train", "y_test", "num_classes"]
)


class DatabaseModelHandler:
    def __init__(self, db_params):
        self.db_params = db_params
        self.conn = None
        self.cursor = None

    def connect_to_db(self):
        self.conn = psycopg2.connect(**self.db_params)
        self.cursor = self.conn.cursor()

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def read_data(self, table: str, where_condition: str, label: str) -> LoadedDataset:
        if where_condition != "":
            cond = parse_conditions(where_condition)
            query = f"SELECT * FROM {table} WHERE {cond}"
        else:
            query = f"SELECT * FROM {table}"

        df = pd.read_sql(query, self.conn)
        features = df.columns.tolist()
        features.remove(label)
        X = df[features].values
        y, _ = pd.factorize(df[label].values)

        num_classes = len(pd.unique(y))  # Ensure this is calculated after factorization

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        return LoadedDataset(X_train, X_test, y_train, y_test, num_classes)

    def create_model_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model (
                model_id SERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_path TEXT,
                model_byte BYTEA
            )
        """
        )
        self.conn.commit()

    def insert_model_binary(
        self, model_path: str, model_name: str, model_binary: bytes
    ) -> str:
        self.cursor.execute(
            "INSERT INTO model (model_name, model_path, model_byte) VALUES (%s, %s, %s) RETURNING model_id",
            (model_name, model_path, psycopg2.Binary(model_binary)),
        )
        # Fetch the ID of the inserted row
        inserted_model_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return inserted_model_id

    def get_model_binary(self, model_id: str) -> (bytes, str):
        self.cursor.execute(
            "SELECT model_name, model_byte FROM model WHERE model_id = %s", (model_id,)
        )
        result = self.cursor.fetchone()
        if result is None:
            return None
        return result[0], result[1]
