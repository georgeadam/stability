import copy
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from .augmented import AugmentedDataset
from .contrastive import ContrastiveLearningViewGenerator
from .creation import datasets
from .data_module import DataModule
from .transforms.gaussian_blur import GaussianBlur


class CIFAR10DataModule(DataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int,
                 random_state: int):
        super().__init__(data_dir, train_size, val_size, extra_size, batch_size, random_state)

        dataset_mean = [0.491, 0.482, 0.447]
        dataset_std = [0.247, 0.243, 0.262]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.supervised_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        simclr_transform_pipeline = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomApply(
                                                            [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                                        transforms.RandomGrayscale(p=0.2),
                                                        GaussianBlur(kernel_size=int(0.1 * 32)),
                                                        transforms.ToTensor()])

        self.contrastive_transform = ContrastiveLearningViewGenerator(simclr_transform_pipeline)

        self.prepare_data()
        self.setup(None)

    def prepare_data(self):
        # download
        if not self.train_data:
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if not self.train_data:
            cifar_full = CIFAR10(self.data_dir, train=True)

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
            self.orig_train_data = copy.deepcopy(self.train_data)

            test_data = CIFAR10(self.data_dir, train=False)
            self.test_data = AugmentedDataset(test_data, np.arange(len(test_data)), 0)
            predict_data = CIFAR10(self.data_dir, train=False)
            self.predict_data = AugmentedDataset(predict_data, np.arange(len(predict_data)), 0)

    def merge_train_and_extra_data(self):
        self.train_data.data = np.concatenate([self.train_data.data, self.extra_data.data])
        self.train_data.targets = torch.cat([self.train_data.targets, self.extra_data.targets])
        self.train_data.indices = np.concatenate([self.train_data.indices, self.extra_data.indices])

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
