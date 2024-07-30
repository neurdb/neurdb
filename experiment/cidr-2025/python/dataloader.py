from typing import List
import warnings
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader, random_split

import pandas as pd
import psycopg2
import torch
import config
from config import logger


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
        # self.nfeat = self.get_nfeat()
        # self.nfield = self.get_nfield()

        # avazue_test1
        self.nfeat = 1544272
        self.nfield = 22
        
        self._fetch_batch(0) # cache first batch

    def get_length(self):
        """
        Get the number of rows in the table
        """
        self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return self.cursor.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        lb = self.batch_size * self.batch_id
        ub = self.batch_size * (self.batch_id + 1)
        if idx < lb or idx >= ub:
            self._fetch_batch(idx)

        # x = torch.tensor(
        #     self.batch_cache_x.iloc[idx % self.batch_size].values, dtype=torch.long
        # )
        # y = torch.tensor(
        #     self.batch_cache_y.iloc[idx % self.batch_size], dtype=torch.float
        # )
        x = self.batch_cache_x[idx % self.batch_size]
        y = self.batch_cache_y[idx % self.batch_size]

        return {
            "id": x,
            # FIXME: The current table schema suppose every id have value 1.
            # How to get the real value from the table? May need to change the
            # data storage format.
            "value": torch.tensor([1.0] * x.shape[0], dtype=torch.float),
            "y": y,
        }

    def _fetch_batch(self, idx):
        """
        load a new batch from the database if the current batch is exhausted
        """
        batch_id = idx // self.batch_size

        with warnings.catch_warnings():
            # suppress "not tested for psycopg" warning
            warnings.simplefilter("ignore", UserWarning)

            batch_cache = pd.read_sql(
                f"SELECT * FROM {self.table_name} "
                f"LIMIT {self.batch_size} "
                f"OFFSET {batch_id * self.batch_size}",
                self.connection,
            )

        # self.batch_cache_x = batch_cache.drop(columns=["id", "label"])
        # self.batch_cache_y = batch_cache["label"]
        self.batch_cache_x = torch.tensor(batch_cache.drop(columns=["id", "label"]).values, dtype=torch.long)
        self.batch_cache_y = torch.tensor(batch_cache["label"].values, dtype=torch.float)
        self.batch_id = batch_id

        # logger.debug(
        #     "new batch fetched from DB",
        #     idx=idx,
        #     batch_id=self.batch_id,
        #     x_shape=self.batch_cache_x.shape,
        #     y_shape=self.batch_cache_y.shape,
        # )

    def get_column_names(self) -> List[str]:
        self.cursor.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = N'{self.table_name}'"
        )
        column_names: List[str] = [r[0] for r in self.cursor.fetchall()]
        column_names = [r for r in column_names if r not in ["id", "label"]]
        logger.debug("get column names", column_names=column_names)
        
        return column_names

    def get_nfeat(self) -> int:
        """
        Get the number of features
        """
        self.cursor.execute(
            f"SELECT MAX(GREATEST({','.join(self.get_column_names())})) FROM {self.table_name}"
        )
        result = self.cursor.fetchone()[0] + 1
        logger.debug("get nfeat", nfeat=result)

        # self.cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 1")
        # nfeat_in_line = len(self.cursor.fetchone()) - 2  # TODO: Not sure if this is correct
        return result

    def get_nfield(self) -> int:
        """
        Get the number of fields
        """
        result = len(self.get_column_names())
        logger.debug("get nfield", nfield=result)
        return result

    def close(self):
        self.connection.close()


def table_dataloader(database_config, table_name: str, batch_size: int):
    """
    Mirror from pysrc/utils/dataset.py
    """
    dataset = TableDataset(database_config, table_name, batch_size)
    logger.debug("TableDataset initialized")
    val_split = 0.1
    test_split = 0.1
    nfield = dataset.nfield
    nfeat = dataset.nfeat

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size
    # train_dataset, val_dataset, test_dataset = random_split(
    #     dataset, [train_size, val_size, test_size]
    # )
    train_dataset, val_dataset, test_dataset = (
        Subset(dataset, range(0, train_size)),
        Subset(dataset, range(train_size, train_size+val_size)),
        Subset(dataset, range(train_size+val_size))
    )

    logger.debug("TableDataset splited")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.DATALOADER_NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.DATALOADER_NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.DATALOADER_NUM_WORKERS,
        pin_memory=True,
    )
    logger.debug("data loader created")

    return train_loader, val_loader, test_loader, nfield, nfeat
