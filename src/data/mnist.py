import copy
import os
from typing import Optional

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from settings import ROOT_DIR
from .augmented import AugmentedDataset
from .creation import datasets


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int,
                 random_state: int):
        super().__init__()

        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.train_size = train_size
        self.val_size = val_size
        self.extra_size = extra_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_data = None
        self.orig_train_data = None
        self.extra_data = None
        self.test_data = None
        self.predict_data = None

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if (stage == "fit" or stage is None) and not self.train_data:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            all_indices = np.random.choice(np.arange(len(mnist_full)),
                                           size=self.train_size + self.val_size + self.extra_size,
                                           replace=False)
            train_indices, val_indices = train_test_split(all_indices, test_size=self.val_size,
                                                          random_state=self.random_state)
            mnist_train = copy.deepcopy(mnist_full)
            mnist_val = copy.deepcopy(mnist_full)

            mnist_train = AugmentedDataset(mnist_train, train_indices)
            mnist_train.data = mnist_train.data[train_indices]
            mnist_train.targets = mnist_train.targets[train_indices]

            mnist_val = AugmentedDataset(mnist_val, val_indices)
            mnist_val.data = mnist_val.data[val_indices]
            mnist_val.targets = mnist_val.targets[val_indices]

            if self.extra_size == 0:
                train_indices = np.arange(len(mnist_train))
                extra_indices = np.array([]).astype(int)
            else:
                train_indices, extra_indices = train_test_split(np.arange(len(mnist_train)), test_size=self.extra_size,
                                                                random_state=self.random_state)
            mnist_extra = copy.deepcopy(mnist_train)

            mnist_train.data = mnist_train.data[train_indices]
            mnist_train.targets = mnist_train.targets[train_indices]
            mnist_train.indices = mnist_train.indices[train_indices]

            mnist_extra.data = mnist_extra.data[extra_indices]
            mnist_extra.targets = mnist_extra.targets[extra_indices]
            mnist_extra.indices = mnist_extra.indices[extra_indices]

            self.train_data = mnist_train
            self.val_data = mnist_val
            self.extra_data = mnist_extra
            self.orig_train_data = copy.deepcopy(self.train_data)

            test_data = MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_data = AugmentedDataset(test_data, np.arange(len(test_data)))
            predict_data = MNIST(self.data_dir, train=False, transform=self.transform)
            self.predict_data = AugmentedDataset(predict_data, np.arange(len(predict_data)))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def train_dataloader_ordered(self):
        return DataLoader(self.orig_train_data, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False)

    def extra_dataloader(self):
        return DataLoader(self.extra_data, batch_size=self.batch_size, shuffle=False)

    def merge_train_and_extra_data(self):
        self.train_data.data = torch.cat([self.train_data.data, self.extra_data.data])
        self.train_data.targets = torch.cat([self.train_data.targets, self.extra_data.targets])
        self.train_data.indices = np.concatenate([self.train_data.indices, self.extra_data.indices])

    @property
    def num_classes(self):
        return 10

    @property
    def train_labels(self):
        return np.array(self.orig_train_data.targets)

    @property
    def test_labels(self):
        return np.array(self.test_data.targets)


datasets.register_builder("mnist", MNISTDataModule)
