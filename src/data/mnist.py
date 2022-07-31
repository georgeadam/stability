import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
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

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            if not self.train_data:
                # If this is the first time using the DataModule, we exclude the extra data from training
                mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
                mnist_sub, mnist_rest = random_split(mnist_full,
                                                     [self.train_size + self.extra_size + self.val_size,
                                                      len(mnist_full) - self.train_size - self.extra_size - self.val_size])
                self.combined_data, self.val_data = random_split(mnist_sub,
                                                                 [self.train_size + self.extra_size, self.val_size])
                self.train_data, _ = random_split(self.combined_data, [self.train_size, self.extra_size])
            else:
                # If this is the second time using the DataModule, we include the extra data for training to
                # see effect on churn
                self.train_data = self.combined_data

        if stage == "test" or stage is None:
            self.test_data = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.predict_data = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False)

    @property
    def output_dim(self):
        return 10

    @property
    def labels(self):
        return self.predict_data.targets


datasets.register_builder("mnist", MNISTDataModule)
