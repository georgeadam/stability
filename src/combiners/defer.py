import abc
import copy

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.lightning_modules import lightning_modules
from src.models import models
from src.utils.inference import extract_predictions
from .creation import combiners


class Defer(metaclass=abc.ABCMeta):
    def __init__(self, base_model, new_model, dataset, callback_args, model_args, module_args, trainer_args):
        self.base_model = base_model
        self.new_model = new_model
        self.dataset = dataset
        self.defer_model = None
        self.defer_module = None
        self.trainer = None

        self._setup(callback_args, model_args, module_args, trainer_args)

    def predict(self, dataloader):
        base_preds = extract_predictions(dataloader, self.base_model)
        new_preds = extract_predictions(dataloader, self.base_model)
        combined_preds = np.stack([base_preds, new_preds], axis=1)

        choices = extract_predictions(dataloader, self.defer_model)
        sample_indices = np.arange(len(combined_preds))
        meta_preds = combined_preds[sample_indices, choices]

        return meta_preds

    def _setup(self, callback_args, model_args, module_args, trainer_args):
        wandb_logger = WandbLogger(project="stability", prefix="combiner")
        self.defer_model = models.create(model_args.name, num_classes=2, num_channels=self.dataset.num_channels,
                                         height=self.dataset.height, **model_args.params)
        self.defer_module = lightning_modules.create(module_args.name, model=self.defer_model, **module_args.params)
        self.trainer = Trainer(logger=wandb_logger, callbacks=EarlyStopping("val/loss", **callback_args.early_stopping),
                               log_every_n_steps=1, deterministic=True, gpus=1, **trainer_args)

        train_dataloader, val_dataloader = self._create_defer_dataloaders()
        self.trainer.fit(self.defer_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    def _create_defer_dataloaders(self):
        original_val_preds = extract_predictions(self.dataset.val_dataloader(), self.base_model)
        new_val_preds = extract_predictions(self.dataset.val_dataloader(), self.new_model)

        original_correct = torch.tensor(original_val_preds == self.dataset.val_labels).int()
        new_correct = torch.tensor(new_val_preds == self.dataset.val_labels).int()

        targets = torch.stack([original_correct, new_correct], dim=1).float()

        train_x, val_x, train_y, val_y = train_test_split(self.dataset.val_data.data, targets, test_size=0.2)
        train_dataloader = self._create_defer_dataloader(self.dataset.val_data, train_x, train_y,
                                                         self.dataset.tensor_transform, self.dataset.batch_size)
        val_dataloader = self._create_defer_dataloader(self.dataset.val_data, val_x, val_y,
                                                       self.dataset.tensor_transform, self.dataset.batch_size)

        return train_dataloader, val_dataloader

    def _create_defer_dataloader(self, original_data, x, y, transform, batch_size):
        defer_data = copy.deepcopy(original_data)
        defer_data.transform = transform
        defer_data.data = x
        defer_data.targets = y
        dataloader = DataLoader(defer_data, batch_size=batch_size)

        return dataloader


combiners.register_builder("defer", Defer)
