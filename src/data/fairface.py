import copy
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from settings import ROOT_DIR
from .augmented import AugmentedDataset
from .creation import datasets
from .data_module import DataModule
from .utils import add_label_noise


class FairFace(Dataset):
    def __init__(self, data_dir, split="train"):
        if split == "train":
            files = os.listdir(os.path.join(ROOT_DIR, data_dir, "train"))
            files = sorted(files, key=lambda x: int(x.split(".")[0]))
            labels = pd.read_csv(os.path.join(ROOT_DIR, data_dir, "processed", "fairface_label_train.csv"))
            labels = labels["race"].to_numpy().astype(int)
        else:
            files = os.listdir(os.path.join(ROOT_DIR, data_dir, "val"))
            files = sorted(files, key=lambda x: int(x.split(".")[0]))
            labels = pd.read_csv(os.path.join(ROOT_DIR, data_dir, "processed", "fairface_label_val.csv"))
            labels = labels["race"].to_numpy().astype(int)

        self.data_dir = os.path.join(ROOT_DIR, data_dir, split)
        self.data = np.array(files)
        self.targets = labels
        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = Image.open(os.path.join(self.data_dir, self.data[index])), self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FairFaceDataModule(DataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int,
                 random_state: int, noise: float):
        super().__init__(data_dir, train_size, val_size, extra_size, batch_size, random_state, noise)
        dataset_mean = [0.482, 0.358, 0.305]
        dataset_std = [0.255, 0.222, 0.217]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.supervised_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        simclr_transform_pipeline = None

        self.prepare_data()
        self.setup(None)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.train_data:
            full_data = FairFace(self.data_dir, split="train")

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
            train_data.targets = train_data.targets[train_indices]

            val_data = AugmentedDataset(val_data, val_indices, 0)
            val_data.data = val_data.data[val_indices]
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

            test_data = FairFace(self.data_dir, split="val")
            self.test_data = AugmentedDataset(test_data, np.arange(len(test_data)), 0)
            predict_data = FairFace(self.data_dir, split="val")
            self.predict_data = AugmentedDataset(predict_data, np.arange(len(predict_data)), 0)

    def merge_train_and_extra_data(self):
        self.train_data.data = np.concatenate([self.train_data.data, self.extra_data.data])
        self.train_data.targets = np.concatenate([self.train_data.targets, self.extra_data.targets])
        self.train_data.indices = np.concatenate([self.train_data.indices, self.extra_data.indices])

    @property
    def num_classes(self):
        return 7

    @property
    def num_channels(self):
        return 3

    @property
    def height(self):
        return 224

    @property
    def stats(self):
        return {"height": self.height,
                "num_channels": self.num_channels}

    @property
    def train_labels(self):
        return np.array(self.orig_train_data.targets)

    @property
    def test_labels(self):
        return np.array(self.test_data.targets)

    @property
    def val_labels(self):
        return np.array(self.val_data.targets)


datasets.register_builder("fairface", FairFaceDataModule)
