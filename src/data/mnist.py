import copy
import os
from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from settings import ROOT_DIR
from .creation import datasets


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int):
        super().__init__()

        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.train_size = train_size
        self.val_size = val_size
        self.extra_size = extra_size
        self.batch_size = batch_size
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
            train_indices, val_indices = train_test_split(np.arange(len(mnist_full)), test_size=self.val_size)
            mnist_train = copy.deepcopy(mnist_full)
            mnist_val = copy.deepcopy(mnist_full)

            mnist_train.data = mnist_train.data[train_indices]
            mnist_train.targets = [mnist_train.targets[i] for i in train_indices]

            mnist_val.data = mnist_val.data[val_indices]
            mnist_val.targets = [mnist_val.targets[i] for i in val_indices]

            train_indices, extra_indices = train_test_split(np.arange(len(mnist_train)), test_size=self.extra_size)
            mnist_extra = copy.deepcopy(mnist_train)

            mnist_train.data = mnist_train.data[train_indices]
            mnist_train.targets = [mnist_train.targets[i] for i in train_indices]

            mnist_extra.data = mnist_extra.data[extra_indices]
            mnist_extra.targets = [mnist_extra.targets[i] for i in extra_indices]

            self.train_data = mnist_train
            self.val_data = mnist_val
            self.extra_data = mnist_extra
            self.orig_train_data = self.train_data

            self.test_data = MNIST(self.data_dir, train=False, transform=self.transform)
            self.predict_data = MNIST(self.data_dir, train=False, transform=self.transform)

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

    def merge_train_and_extra_data(self):
        self.train_data = ConcatDataset([self.train_data, self.extra_data])

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
