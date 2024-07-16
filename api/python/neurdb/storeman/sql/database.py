from datetime import datetime
from pprint import pprint
from typing import List, Optional

import psycopg2
from ..config import Configuration
from ..common.storage import ModelStorage
from .entity import ModelEntity, LayerEntity
from neurdb.logger import logger


class NeurDB:
    """
    NeurDB manages the database connection and provides methods to save, load, update and delete models
    """

    def __init__(
        self,
        db_name: str = Configuration.DB_NAME,
        db_user: str = Configuration.DB_USER,
        db_host: str = Configuration.DB_HOST,
        db_port: str = Configuration.DB_PORT,
    ):
        self.database = Database(db_name, db_user, db_host, db_port)
        self._init_neurdb()

    def save_model(self, model_storage_pickle: ModelStorage.Pickled) -> int:
        """
        Save the model to the database
        @param ModelStorage.Pickled model_storage_pickle: pickled representation of the model
        @return: model_id
        """
        model_entity = ModelEntity(model_storage_pickle.model_meta_pickled)
        model_id = self._save_new_model_entity(model_entity)

        if model_id is None:
            raise ValueError("Failed to insert model into the database.")

        # save the layers
        for index, layer_pickled in enumerate(
            model_storage_pickle.layer_sequence_pickled
        ):
            layer_entity = LayerEntity(model_id, index, datetime.now(), layer_pickled)
            self._save_layer_entity(layer_entity)
        return model_id

    def load_model(self, model_id: int) -> ModelStorage.Pickled:
        """
        Load the model from the database
        @param int model_id: model id
        @return: ModelStorage.Pickled
        """
        try:
            model_meta = self.database.select(
                "model", ["model_meta"], ["model_id = %s"], [model_id]
            )[0][0].tobytes()
        except IndexError:
            raise FileNotFoundError(f"cannot find model id: {model_id}")

        layers = self.database.select(
            "layer",
            ["model_id", "layer_id", "create_time", "layer_data"],
            ["model_id = %s"],
            [model_id],
        )

        # if two layers have the same layer_id, the one with the latest create_time is kept,
        # all layers are sorted by layer_id from smallest to largest
        layers = sorted(
            layers, key=lambda x: (x[1], datetime.now() - x[2]), reverse=False
        )  # sort in ascending order

        selected_layers = [layers[0]]
        for i in range(1, len(layers)):
            if layers[i][1] != layers[i - 1][1]:
                selected_layers.append(layers[i])

        [
            logger.debug(
                "select layer", model_id=x[0], layer_id=x[1], create_time=str(x[2])
            )
            for x in selected_layers
        ]

        layer_sequence_pickled = [layer[3] for layer in selected_layers]

        return ModelStorage.Pickled(model_meta, layer_sequence_pickled)

    def update_model(self, model_id: int, layer_id: int, layer_data: bytes) -> None:
        """
        Update one layer of the model in the database
        @param model_id: the model id
        @param layer_id: the layer id
        @param layer_data: the new layer data in bytes
        @return: None
        """
        # check if the model_id exists
        if not self.database.select(
            "model", ["model_id"], ["model_id = %s"], [model_id]
        ):
            raise ValueError("The model_id does not exist in the database.")

        self.database.insert(
            "layer",
            ["model_id", "layer_id", "create_time", "layer_data"],
            [model_id, layer_id, datetime.now().isoformat(), layer_data],
        )

    def delete_model(self, model_id: int):
        """
        Delete the model from the database
        @param model_id: the model id
        @return: None
        """
        self.database.delete("model", ["model_id = %s"], [model_id])
        self.database.delete("layer", ["model_id = %s"], [model_id])

    def close(self):
        """
        Close the database connection
        @return: None
        """
        self.database.close()

    # ******************** Private methods ********************

    def _init_neurdb(self):
        self.database.create_table(
            "model", ["model_id SERIAL PRIMARY KEY", "model_meta BYTEA"]
        )  # model table

        self.database.create_table(
            "layer",
            [
                "model_id INT",
                "layer_id INT",
                "create_time TIMESTAMP",
                "layer_data BYTEA",
            ],
        )  # layer table

    def _save_new_model_entity(self, model_entity: ModelEntity) -> int:
        """
        Save a new model entity to the database, 'new' means that the model_id is not known
        @param model_entity: model entity
        @return: model_id
        """
        self.database.insert("model", ["model_meta"], [model_entity.model_meta])

        # model_id
        return self.database.select(
            "model",
            ["model_id"],
            ["model_meta = %s"],
            [model_entity.model_meta],
            order_by="model_id",
            ascending=False,
        )[0][0]

    def _save_layer_entity(self, layer_entity: LayerEntity) -> None:
        """
        Save the layer entity to the database
        @param layer_entity: layer entity
        @return: None
        """
        self.database.insert(
            "layer",
            ["model_id", "layer_id", "create_time", "layer_data"],
            [
                layer_entity.model_id,
                layer_entity.layer_id,
                layer_entity.create_time,
                layer_entity.layer_data,
            ],
        )


class Database:
    def __init__(
        self,
        db_name: str = Configuration.DB_NAME,
        db_user: str = Configuration.DB_USER,
        db_host: str = Configuration.DB_HOST,
        db_port: str = Configuration.DB_PORT,
    ):
        self.db_name = db_name
        self.db_user = db_user
        self.db_host = db_host
        self.db_port = db_port
        self.connection = psycopg2.connect(
            dbname=self.db_name, user=self.db_user, host=self.db_host, port=self.db_port
        )
        self.cursor = self.connection.cursor()

    def select(
        self,
        table: str,
        columns: List[str],
        conditions: Optional[List[str]] = None,
        condition_values: Optional[List] = None,
        order_by: Optional[str] = None,
        ascending: bool = True,
    ) -> list:
        """
        Selects data from the database
        @param table: table name
        @param columns: list of columns to select, e.g. ['column1', 'column2']
        @param conditions: list of conditions, e.g. ['column1 = %s', 'column2 = %s']
        @param condition_values: list of values for the conditions, e.g. [value1, value2]
        @return: list of tuples with the selected data
        """
        columns_str = ", ".join(columns)
        if conditions:
            conditions_str = " AND ".join(conditions)
        else:
            conditions_str = "1=1"
            condition_values = []

        query = f"SELECT {columns_str} FROM {table} WHERE {conditions_str}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if not ascending:
            query += " DESC"

        self.cursor.execute(query, condition_values)
        return self.cursor.fetchall()

    def insert(self, table: str, columns: List[str], values: List) -> None:
        """
        Inserts data into the database
        @param table: table name
        @param columns: list of columns to insert, e.g. ['column1', 'column2']
        @param values: list of values to insert, e.g. ['value1', 'value2']
        @return: None
        """
        columns_str = ", ".join(columns)
        values_placeholder = ", ".join(["%s"] * len(values))
        query = f"INSERT INTO {table} ({columns_str}) VALUES ({values_placeholder})"
        self.cursor.execute(query, values)
        self.connection.commit()

    def create_table(self, table: str, columns: List[str]) -> None:
        """
        Creates a table in the database if it does not exist
        @param table: table name
        @param columns: list of columns to create, e.g. ['column1 INT', 'column2 TEXT']
        @return None
        """
        columns_str = ", ".join(columns)
        query = f"CREATE TABLE IF NOT EXISTS {table} ({columns_str})"
        self.cursor.execute(query)
        self.connection.commit()

    def delete(
        self,
        table: str,
        conditions: Optional[List[str]] = None,
        condition_values: Optional[List] = None,
    ) -> None:
        """
        Deletes data from the database
        @param table: table name
        @param conditions: list of conditions, e.g. ['column1 = %s', 'column2 = %s']
        @param condition_values: list of values for the conditions, e.g. [value1, value2]
        @return None
        """
        if conditions:
            conditions_str = " AND ".join(conditions)
        else:
            conditions_str = "1=1"

        query = f"DELETE FROM {table} WHERE {conditions_str}"
        self.cursor.execute(query, condition_values)
        self.connection.commit()

    def close(self) -> None:
        """
        Closes the database connection
        @return: None
        """
        self.cursor.close()
        self.connection.close()
