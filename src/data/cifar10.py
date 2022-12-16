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
from .utils import add_label_noise


class CIFAR10DataModule(DataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int,
                 random_state: int, noise: float):
        super().__init__(data_dir, train_size, val_size, extra_size, batch_size, random_state, noise)

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
            full_data = CIFAR10(self.data_dir, train=True)

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
            train_data.targets = add_label_noise(train_data.targets, self.noise)
            train_data.indices = train_data.indices[train_indices]

            extra_data.data = extra_data.data[extra_indices]
            extra_data.targets = extra_data.targets[extra_indices]
            extra_data.targets = add_label_noise(extra_data.targets, self.noise)
            extra_data.indices = extra_data.indices[extra_indices]
            extra_data.source = 1

            self.train_data = train_data
            self.val_data = val_data
            self.extra_data = extra_data
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

    @property
    def val_labels(self):
        return np.array(self.val_data.targets)


datasets.register_builder("cifar10", CIFAR10DataModule)
