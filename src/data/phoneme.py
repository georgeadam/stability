import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .augmented import AugmentedDataset
from .creation import datasets
from .data_module import DataModule
from .tabular import TabularDataset


class PhonemeDataModule(DataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int,
                 random_state: int, noise: float):
        super().__init__(data_dir, train_size, val_size, extra_size, batch_size, random_state, noise)

        self.prepare_data()
        self.setup(None)

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.train_data:
            x, y = fetch_openml(data_id=1489, data_home=self.data_dir, return_X_y=True, as_frame=True)
            y = pd.factorize(y)[0]
            x = x.to_numpy()
            x = process_data(x)
            x = x.astype("float32")

            full_data = TabularDataset(x, y)

            if self.random_state is not None:
                r = np.random.RandomState(self.random_state)
                all_indices = r.choice(np.arange(len(full_data)),
                                       size=self.train_size + self.val_size + self.extra_size,
                                       replace=False)
            else:
                all_indices = np.random.choice(np.arange(len(full_data)),
                                               size=self.train_size + self.val_size + self.extra_size,
                                               replace=False)

            if self.val_size == 0:
                train_indices = all_indices
                val_indices = np.array([]).astype(int)
            else:
                train_indices, val_indices = train_test_split(all_indices, test_size=self.val_size,
                                                              random_state=self.random_state)

            train_data = copy.deepcopy(full_data)
            val_data = copy.deepcopy(full_data)

            train_data = AugmentedDataset(train_data, train_indices, 0)
            train_data.data = train_data.data[train_indices]
            train_data.targets = torch.tensor(train_data.targets)
            train_data.targets = train_data.targets[train_indices]

            val_data = AugmentedDataset(val_data, val_indices, 0)
            val_data.data = val_data.data[val_indices]
            val_data.targets = torch.tensor(val_data.targets)
            val_data.targets = val_data.targets[val_indices]

            if self.extra_size == 0:
                train_indices = np.arange(len(train_data))
                extra_indices = np.array([]).astype(int)
            else:
                train_indices, extra_indices = train_test_split(np.arange(len(train_data)), test_size=self.extra_size,
                                                                random_state=self.random_state)

            extra_data = copy.deepcopy(train_data)

            train_data.data = train_data.data[train_indices]
            train_data.targets = train_data.targets[train_indices]
            train_data.indices = train_data.indices[train_indices]

            extra_data.data = extra_data.data[extra_indices]
            extra_data.targets = extra_data.targets[extra_indices]
            extra_data.indices = extra_data.indices[extra_indices]
            extra_data.source = 1

            self.train_data = train_data
            self.val_data = val_data
            self.extra_data = extra_data
            self.orig_train_data = copy.deepcopy(self.train_data)

            test_indices = np.setdiff1d(np.arange(len(full_data)), all_indices)
            test_data = copy.deepcopy(full_data)
            test_data = AugmentedDataset(test_data, test_indices, 0)
            test_data.data = test_data.data[test_indices]
            test_data.targets = test_data.targets[test_indices]
            self.test_data = test_data
            self.predict_data = copy.deepcopy(test_data)

    def merge_train_and_extra_data(self):
        self.train_data.data = np.concatenate([self.train_data.data, self.extra_data.data])
        self.train_data.targets = torch.cat([self.train_data.targets, self.extra_data.targets])
        self.train_data.indices = np.concatenate([self.train_data.indices, self.extra_data.indices])

    @property
    def num_classes(self):
        return 2

    @property
    def num_features(self):
        return 5

    @property
    def stats(self):
        return {"num_features": self.num_features}

    @property
    def train_labels(self):
        return np.array(self.orig_train_data.targets)

    @property
    def test_labels(self):
        return np.array(self.test_data.targets)

    @property
    def val_labels(self):
        return np.array(self.val_data.targets)


def process_data(x):
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(x)
    x = standard_scaler.transform(x)

    return x


datasets.register_builder("phoneme", PhonemeDataModule)
