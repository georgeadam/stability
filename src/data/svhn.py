import copy
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import SVHN
from torchvision.transforms import transforms

from .augmented import AugmentedDataset
from .creation import datasets
from .data_module import DataModule

from PIL import Image


class MySVHN(SVHN):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform = None,
        target_transform = None,
        download = False,
    ) -> None:
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)

        self.targets = self.labels
        del self.labels

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHNDataModule(DataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int,
                 random_state: int):
        super().__init__(data_dir, train_size, val_size, extra_size, batch_size, random_state)

        dataset_mean = [0.438, 0.444, 0.473]
        dataset_std = [0.198, 0.201, 0.197]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.supervised_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.tensor_transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.constrastive_transform = transforms.Compose(
            [transforms.ToTensor(), normalize])

        self.prepare_data()
        self.setup(None)

    def prepare_data(self):
        # download
        if not self.train_data:
            MySVHN(self.data_dir, split="train", download=True)
            MySVHN(self.data_dir, split="test", download=True)

    def setup(self, stage: Optional[str] = None):
        if not self.train_data:
            svhn_full = MySVHN(self.data_dir, split="train")

            if self.random_state is not None:
                r = np.random.RandomState(self.random_state)
                all_indices = r.choice(np.arange(len(svhn_full)),
                                       size=self.train_size + self.val_size + self.extra_size,
                                       replace=False)
            else:
                all_indices = np.random.choice(np.arange(len(svhn_full)),
                                               size=self.train_size + self.val_size + self.extra_size,
                                               replace=False)

            train_indices, val_indices = train_test_split(all_indices, test_size=self.val_size,
                                                          random_state=self.random_state)
            svhn_train = copy.deepcopy(svhn_full)
            svhn_val = copy.deepcopy(svhn_full)

            svhn_train = AugmentedDataset(svhn_train, train_indices, 0)
            svhn_train.data = svhn_train.data[train_indices]
            svhn_train.targets = svhn_train.targets[train_indices]

            svhn_val = AugmentedDataset(svhn_val, val_indices, 0)
            svhn_val.data = svhn_val.data[val_indices]
            svhn_val.targets = svhn_val.targets[val_indices]

            if self.extra_size == 0:
                train_indices = np.arange(len(svhn_train))
                extra_indices = np.array([]).astype(int)
            else:
                train_indices, extra_indices = train_test_split(np.arange(len(svhn_train)), test_size=self.extra_size,
                                                                random_state=self.random_state)
            svhn_extra = copy.deepcopy(svhn_train)

            svhn_train.data = svhn_train.data[train_indices]
            svhn_train.targets = svhn_train.targets[train_indices]
            svhn_train.indices = svhn_train.indices[train_indices]

            svhn_extra.data = svhn_extra.data[extra_indices]
            svhn_extra.targets = svhn_extra.targets[extra_indices]
            svhn_extra.indices = svhn_extra.indices[extra_indices]
            svhn_extra.source = 1

            self.train_data = svhn_train
            self.val_data = svhn_val
            self.extra_data = svhn_extra
            self.orig_train_data = copy.deepcopy(self.train_data)

            test_data = MySVHN(self.data_dir, split="test")
            self.test_data = AugmentedDataset(test_data, np.arange(len(test_data)), 0)
            predict_data = MySVHN(self.data_dir, split="test")
            self.predict_data = AugmentedDataset(predict_data, np.arange(len(predict_data)), 0)

    def merge_train_and_extra_data(self):
        self.train_data.data = np.concatenate([self.train_data.data, self.extra_data.data])
        self.train_data.targets = np.concatenate([self.train_data.targets, self.extra_data.targets])
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


datasets.register_builder("svhn", SVHNDataModule)
