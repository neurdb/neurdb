import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split


class LibsvmDataset(Dataset):
    """ Dataset loader for Libsvm data format """

    def __init__(self, data: str):
        self.data = data.split('\n')
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
                continue    # skip empty lines
            columns = line.strip().split(' ')
            nfields_in_line = len(columns) - 1
            max_nfields = max(max_nfields, nfields_in_line)

            pairs = [list(map(int, pair.split(':'))) for pair in columns[1:]]
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
                    self.feat_id[i, :len(ids_list[i])] = torch.tensor(ids_list[i], dtype=torch.long)
                    self.feat_value[i, :len(values_list[i])] = torch.tensor(values_list[i], dtype=torch.float)
                except Exception as e:
                    print(f'Incorrect data format in sample {i}! Error: {e}')
                pbar.update(1)
        print(f'# {self.nsamples} data samples loaded...')

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {'id': self.feat_id[idx], 'value': self.feat_value[idx], 'y': self.y[idx]}


def libsvm_dataloader(batch_size: int, data_loader_worker: int, data: str):
    val_split = 0.1
    test_split = 0.1
    dataset = LibsvmDataset(data)
    nfields = dataset.nfields
    nfeat = dataset.nfeat
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    test_size = int(total_samples * test_split)
    train_size = total_samples - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=data_loader_worker,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=data_loader_worker,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=data_loader_worker,
                             pin_memory=True)

    return train_loader, val_loader, test_loader, nfields, nfeat


def build_inference_loader(data_loader_worker: int, data: str):
    dataset = LibsvmDataset(data)
    nfields = dataset.nfields
    nfeat = dataset.nfeat
    total_samples = len(dataset)
    loader = DataLoader(dataset,
                        batch_size=total_samples,
                        shuffle=False,
                        num_workers=data_loader_worker,
                        pin_memory=True)
    return loader, nfields, nfeat
