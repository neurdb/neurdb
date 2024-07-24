from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

import pandas as pd
import psycopg2
import torch


class TableDataset(Dataset):
    def __init__(self, database_config, table_name: str, batch_size: int):
        """
        Initialize the DatabaseDataset
        :param cursor: The cursor to the database
        :param table_name: The name of the table
        :param batch_size: The batch size, i.e., the number of rows to load at once
        """
        self.connection = psycopg2.connect(**database_config)
        self.cursor = self.connection.cursor()
        self.table_name = table_name
        self.batch_size = batch_size
        self.batch_cache_x = pd.DataFrame()
        self.batch_cache_y = pd.DataFrame()
        self.batch_id = 0
        self.length = self.get_length()
        self.nfeat = self.get_nfeat()
        self.nfield = self.get_nfield()

    def get_length(self):
        """
        Get the number of rows in the table
        """
        self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return self.cursor.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.batch_size * self.batch_id < idx:
            # load a new batch from the database if the current batch is exhausted
            batch_cache = pd.read_sql(
                f"SELECT * FROM {self.table_name} "
                f"LIMIT {self.batch_size} "
                f"OFFSET {self.batch_id * self.batch_size}",
                self.connection
            )
            self.batch_cache_x = batch_cache.drop(columns=['label'])
            self.batch_cache_y = batch_cache['label']
            self.batch_id += 1

        x = torch.tensor(self.batch_cache_x.iloc[idx % self.batch_size].values, dtype=torch.float)
        y = torch.tensor(self.batch_cache_y.iloc[idx % self.batch_size].values, dtype=torch.long)
        return x, y

    def get_nfeat(self) -> int:
        """
        Get the number of features
        """
        self.cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 1")
        nfeat_in_line = len(self.cursor.fetchone())  # TODO: Not sure if this is correct
        return nfeat_in_line

    def get_nfield(self) -> int:
        """
        Get the number of fields
        """
        self.cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 1")
        nfield_in_line = len(self.cursor.fetchone()) - 1
        return nfield_in_line

    def close(self):
        self.connection.close()


def table_dataloader(database_config, table_name: str, batch_size: int):
    """
    Mirror from pysrc/utils/dataset.py
    """
    dataset = TableDataset(database_config, table_name, batch_size)
    val_split = 0.1
    test_split = 0.1
    nfield = dataset.nfield
    nfeat = dataset.nfeat

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # TODO: change this to args specified
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # TODO: change this to args specified
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # TODO: change this to args specified
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, nfield, nfeat
