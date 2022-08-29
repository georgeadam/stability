import copy
import os
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from settings import ROOT_DIR
from .augmented import AugmentedDataset
from .creation import datasets


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int,
                 random_state: int):
        super().__init__()

        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.train_size = train_size
        self.val_size = val_size
        self.extra_size = extra_size
        self.batch_size = batch_size
        self.random_state = random_state
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
        self._sampler = None

        self.prepare_data()
        self.setup(None)

    def prepare_data(self):
        # download
        if not self.train_data:
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if not self.train_data:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.train_transform)

            if self.random_state:
                r = np.random.RandomState(self.random_state)
                all_indices = r.choice(np.arange(len(cifar_full)),
                                       size=self.train_size + self.val_size + self.extra_size,
                                       replace=False)
            else:
                all_indices = np.random.choice(np.arange(len(cifar_full)),
                                               size=self.train_size + self.val_size + self.extra_size,
                                               replace=False)

            train_indices, val_indices = train_test_split(all_indices, test_size=self.val_size,
                                                          random_state=self.random_state)
            cifar_train = copy.deepcopy(cifar_full)
            cifar_val = copy.deepcopy(cifar_full)

            cifar_train = AugmentedDataset(cifar_train, train_indices, 0)
            cifar_train.data = cifar_train.data[train_indices]
            cifar_train.targets = torch.tensor(cifar_train.targets)
            cifar_train.targets = cifar_train.targets[train_indices]

            cifar_val = AugmentedDataset(cifar_val, val_indices, 0)
            cifar_val.data = cifar_val.data[val_indices]
            cifar_val.targets = torch.tensor(cifar_val.targets)
            cifar_val.targets = cifar_val.targets[val_indices]

            if self.extra_size == 0:
                train_indices = np.arange(len(cifar_train))
                extra_indices = np.array([]).astype(int)
            else:
                train_indices, extra_indices = train_test_split(np.arange(len(cifar_train)), test_size=self.extra_size,
                                                                random_state=self.random_state)

            cifar_extra = copy.deepcopy(cifar_train)

            cifar_train.data = cifar_train.data[train_indices]
            cifar_train.targets = cifar_train.targets[train_indices]
            cifar_train.indices = cifar_train.indices[train_indices]

            cifar_extra.data = cifar_extra.data[extra_indices]
            cifar_extra.targets = cifar_extra.targets[extra_indices]
            cifar_extra.indices = cifar_extra.indices[extra_indices]
            cifar_extra.source = 1

            self.train_data = cifar_train
            self.val_data = cifar_val
            self.extra_data = cifar_extra
            self.val_data.transform = self.val_transform
            self.orig_train_data = copy.deepcopy(self.train_data)

            test_data = CIFAR10(self.data_dir, train=False, transform=self.val_transform)
            self.test_data = AugmentedDataset(test_data, np.arange(len(test_data)), 0)
            predict_data = CIFAR10(self.data_dir, train=False, transform=self.val_transform)
            self.predict_data = AugmentedDataset(predict_data, np.arange(len(predict_data)), 0)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def train_dataloader_curriculum(self, sampler):
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=sampler)

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
        self.train_data.data = np.concatenate([self.train_data.data, self.extra_data.data])
        self.train_data.targets = torch.cat([self.train_data.targets, self.extra_data.targets])
        self.train_data.indices = np.concatenate([self.train_data.indices, self.extra_data.indices])

    def sort_samples_by_score(self, scorer):
        indices, scores = scorer.generate_scores()

        if indices is None:
            return

        sorted_indices = np.argsort(scores)
        sorted_indices, sorted_scores = indices[sorted_indices], scores[sorted_indices]

        mapping = np.where(sorted_indices.reshape(sorted_indices.size, 1) == self.train_data.indices)[1]

        self.train_data.data = self.train_data.data[mapping]
        self.train_data.targets = self.train_data.targets[mapping]
        self.train_data.indices = self.train_data.indices[mapping]

    @property
    def num_classes(self):
        return 10

    @property
    def num_channels(self):
        return 3

    @property
    def height(self):
        return 32

    @property
    def train_labels(self):
        return np.array(self.orig_train_data.targets)

    @property
    def test_labels(self):
        return np.array(self.test_data.targets)


datasets.register_builder("cifar10", CIFAR10DataModule)
