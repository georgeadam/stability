import abc
import os

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from settings import ROOT_DIR


class DataModule(LightningDataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int,
                 random_state: int, noise: float):
        super().__init__()

        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.train_size = train_size
        self.val_size = val_size
        self.extra_size = extra_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.noise = noise

        self.supervised_transform = None
        self.tensor_transform = None
        self.contrastive_transform = None

        self.train_data = None
        self.val_data = None
        self.orig_train_data = None
        self.extra_data = None
        self.test_data = None
        self.predict_data = None
        self._sampler = None

    def train_dataloader(self):
        self.train_data.transform = self.supervised_transform
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def train_dataloader_contrastive(self):
        self.train_data.transform = self.contrastive_transform
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def train_dataloader_curriculum(self, sampler):
        self.train_data.transform = self.supervised_transform
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=sampler)

    def train_dataloader_ordered(self):
        self.orig_train_data.transform = self.supervised_transform
        return DataLoader(self.orig_train_data, batch_size=self.batch_size, shuffle=False)

    def train_dataloader_inference(self, batch_size):
        self.train_data.transform = self.tensor_transform
        return DataLoader(self.train_data, batch_size=batch_size, shuffle=False)

    def val_dataloader(self):
        self.val_data.transform = self.tensor_transform
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def val_dataloader_contrastive(self):
        self.val_data.transform = self.contrastive_transform
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        self.test_data.transform = self.tensor_transform
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        self.predict_data.transform = self.tensor_transform
        return DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False)

    def extra_dataloader(self):
        self.extra_data.transform = self.tensor_transform
        return DataLoader(self.extra_data, batch_size=self.batch_size, shuffle=False)

    @abc.abstractmethod
    def merge_train_and_extra_data(self):
        raise NotImplementedError

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
    @abc.abstractmethod
    def num_classes(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def stats(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def train_labels(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def test_labels(self):
        raise NotImplementedError
