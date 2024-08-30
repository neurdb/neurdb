import os

import torch
from log import logger
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class LibsvmDataset(Dataset):
    """Dataset loader for Libsvm data format"""

    def __init__(self, data: str):
        self.data = data.split("\n")
        self._load_data()

    def _load_data(self):
        max_nfields = 0
        sample_lines = 0
        ids_list = []
        values_list = []
        labels_list = []
        unique_features = set()
        for line in self.data:
            if not line:
                continue  # skip empty lines
            columns = line.strip().split(" ")
            nfields_in_line = len(columns) - 1
            max_nfields = max(max_nfields, nfields_in_line)

            pairs = [list(map(int, pair.split(":"))) for pair in columns[1:]]
            ids, values = zip(*pairs) if pairs else ([], [])
            ids_list.append(ids)
            values_list.append(values)
            labels_list.append(float(columns[0]))
            unique_features.update(ids)
            sample_lines += 1

        self.nfields = max_nfields
        # this is start from 0, max dimension is +1
        self.nfeat = max(unique_features) + 1
        self.nsamples = sample_lines
        self.feat_id = torch.zeros((self.nsamples, self.nfields), dtype=torch.long)
        self.feat_value = torch.zeros((self.nsamples, self.nfields), dtype=torch.float)
        self.y = torch.tensor(labels_list, dtype=torch.float)

        with tqdm(total=self.nsamples) as pbar:
            for i in range(self.nsamples):
                try:
                    self.feat_id[i, : len(ids_list[i])] = torch.tensor(
                        ids_list[i], dtype=torch.long
                    )
                    self.feat_value[i, : len(values_list[i])] = torch.tensor(
                        values_list[i], dtype=torch.float
                    )
                except Exception as e:
                    print(f"Incorrect data format in sample {i}! Error: {e}")
                pbar.update(1)

        logger.debug("data samples loaded", n_samples=self.nsamples)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {
            "id": self.feat_id[idx],
            "value": self.feat_value[idx],
            "y": self.y[idx],
        }


def libsvm_dataloader(
    batch_size: int,
    data_loader_worker: int,
    data: str,
    train_batch_num: int,
    eva_batch_num: int,
    test_batch_num: int,
):
    val_split = 0.1
    test_split = 0.1
    dataset = LibsvmDataset(data)
    nfields = dataset.nfields
    nfeat = dataset.nfeat
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    test_size = int(total_samples * test_split)
    train_size = total_samples - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_worker,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_worker,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_worker,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, nfields, nfeat


def build_inference_loader(
    data_loader_worker: int, data: str, batch_size=0, inf_batch_num: int = -1
):
    dataset = LibsvmDataset(data)
    nfields = dataset.nfields
    nfeat = dataset.nfeat
    total_samples = len(dataset)
    if batch_size == 0:
        batch_size = total_samples
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_worker,
        pin_memory=True,
    )
    return loader, nfields, nfeat


if __name__ == "__main__":
    dataset = """0 204:1 4798:1 5041:1 5046:1 5053:1 5055:1 5058:1 5060:1 5073:1 5183:1\n
    1 42:1 1572:1 5042:1 5047:1 5053:1 5055:1 5058:1 5060:1 5070:1 5150:1\n
    1 282:1 2552:1 5044:1 5052:1 5054:1 5055:1 5058:1 5060:1 5072:1 5244:1\n
    0 215:1 1402:1 5039:1 5051:1 5054:1 5055:1 5058:1 5063:1 5069:1 5149:1\n
    0 346:1 2423:1 5043:1 5051:1 5054:1 5055:1 5058:1 5063:1 5088:1 5149:1\n
    0 391:1 2081:1 5039:1 5050:1 5054:1 5055:1 5058:1 5060:1 5088:1 5268:1\n
    0 164:1 3515:1 5042:1 5052:1 5053:1 5055:1 5058:1 5062:1 5074:1 5149:1\n
    0 4:1 1177:1 5044:1 5049:1 5054:1 5057:1 5058:1 5060:1 5071:1 5152:1"""

    _loader, _nfields, _nfeat = build_inference_loader(1, dataset, 4)
    for _ in range(2):
        for batch_idx, batch in enumerate(_loader):
            print(batch_idx, batch)
