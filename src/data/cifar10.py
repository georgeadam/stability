import copy
import os
from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from settings import ROOT_DIR
from .creation import datasets


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int):
        super().__init__()

        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.train_size = train_size
        self.val_size = val_size
        self.extra_size = extra_size
        self.batch_size = batch_size
        dataset_mean = [0.491, 0.482, 0.447]
        dataset_std = [0.247, 0.243, 0.262]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self.train_data = None
        self.orig_train_data = None
        self.extra_data = None
        self.test_data = None
        self.predict_data = None

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if (stage == "fit" or stage is None) and not self.train_data:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.train_transform)
            train_indices, val_indices = train_test_split(np.arange(len(cifar_full)), test_size=self.val_size)
            cifar_train = copy.deepcopy(cifar_full)
            cifar_val = copy.deepcopy(cifar_full)

            cifar_train.data = cifar_train.data[train_indices]
            cifar_train.targets = [cifar_train.targets[i] for i in train_indices]

            cifar_val.data = cifar_val.data[val_indices]
            cifar_val.targets = [cifar_val.targets[i] for i in val_indices]

            train_indices, extra_indices = train_test_split(np.arange(len(cifar_train)), test_size=self.extra_size)
            cifar_extra = copy.deepcopy(cifar_train)

            cifar_train.data = cifar_train.data[train_indices]
            cifar_train.targets = [cifar_train.targets[i] for i in train_indices]

            cifar_extra.data = cifar_extra.data[extra_indices]
            cifar_extra.targets = [cifar_extra.targets[i] for i in extra_indices]

            self.train_data = cifar_train
            self.val_data = cifar_val
            self.extra_data = cifar_extra
            self.val_data.transform = self.val_transform
            self.orig_train_data = self.train_data

            self.test_data = CIFAR10(self.data_dir, train=False, transform=self.val_transform)
            self.predict_data = CIFAR10(self.data_dir, train=False, transform=self.val_transform)

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


datasets.register_builder("cifar10", CIFAR10DataModule)
