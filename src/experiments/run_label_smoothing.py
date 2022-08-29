import logging
import os

import hydra
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from settings import ROOT_DIR
from src.callbacks import trackers
from src.data import datasets
from src.label_smoothers import label_smoothers
from src.lightning_modules import lightning_modules
from src.models import models
from src.utils.logging import log_final_metrics
from src.utils.save import save_predictions

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")


def fit_and_predict_original(args, dataset, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = create_model(args, dataset.num_classes, dataset.num_channels, dataset.height)
    module = create_module_original(args, model)
    callbacks = create_callbacks_original(args)
    trainer = create_trainer(args, list(callbacks.values()), logger)

    trainer.fit(module, datamodule=dataset)

    train_logits = trainer.predict(module, dataloaders=dataset.train_dataloader_ordered())
    train_logits = torch.cat(train_logits)
    train_preds = torch.argmax(train_logits, dim=1).detach().cpu().numpy()

    test_logits = trainer.predict(module, dataloaders=dataset.test_dataloader())
    test_logits = torch.cat(test_logits)
    test_preds = torch.argmax(test_logits, dim=1).detach().cpu().numpy()

    extra_logits = trainer.predict(module, dataloaders=dataset.extra_dataloader())
    extra_logits = torch.cat(extra_logits)

    return model, train_preds, test_preds, train_logits, extra_logits


def fit_and_predict_new(args, dataset, original_model, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = create_model(args, dataset.num_classes, dataset.num_channels, dataset.height)
    module = create_module_new(args, model, original_model)
    callbacks = create_callbacks_new(args)
    trainer = create_trainer(args, list(callbacks.values()), logger)

    trainer.fit(module, datamodule=dataset)

    train_logits = trainer.predict(module, dataloaders=dataset.train_dataloader_ordered())
    train_logits = torch.cat(train_logits)
    train_preds = torch.argmax(train_logits, dim=1).detach().cpu().numpy()

    test_logits = trainer.predict(module, dataloaders=dataset.test_dataloader())
    test_logits = torch.cat(test_logits)
    test_preds = torch.argmax(test_logits, dim=1).detach().cpu().numpy()

    return train_preds, test_preds


def create_model(args, num_classes, num_channels, height):
    return models.create(args.model.name, num_classes=num_classes, num_channels=num_channels, height=height,
                         **args.model.params)


def create_module_original(args, model):
    return lightning_modules.create(args.lightning_module.name, model=model, original_model=None,
                                    **args.lightning_module.params)


def create_module_new(args, model, original_model):
    return lightning_modules.create(args.lightning_module.name, model=model, original_model=original_model,
                                    **args.lightning_module.params)


def create_trainer(args, callbacks, logger):
    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      callbacks=callbacks,
                      deterministic=True,
                      gpus=1,
                      enable_checkpointing=False,
                      **args.trainer)

    return trainer


def create_callbacks_original(args):
    return {"early_stopping": EarlyStopping("val/loss", **args.callbacks.early_stopping),
            "flip_tracker": trackers.create("flip")}


def create_callbacks_new(args):
    return {"early_stopping": EarlyStopping("val/loss", **args.callbacks.early_stopping),
            "churn_tracker": trackers.create("churn")}


def smooth_labels(dataset, logits, smoother):
    smoothed_targets = smoother(dataset.targets, logits=logits)
    dataset.targets = smoothed_targets


def reset_labels(modified_dataset, original_dataset):
    modified_dataset.targets = original_dataset.targets


@hydra.main(config_path=config_path, config_name="label_smoothing")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    seed_everything(args.misc.seed, workers=True)
    dataset = datasets.create(args.data.name, **args.data.params)

    cfg = OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True
    )

    # Initial training
    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="stability", prefix="initial",
                               name="{}_{}_label_smoothing-{}".format(args.data.name,
                                                                      args.model.name,
                                                                      args.misc.seed))
    wandb_logger.experiment.config.update(cfg)
    smoother = label_smoothers.create(args.base_smoother.name, **args.base_smoother.params,
                                      num_classes=dataset.num_classes)
    smooth_labels(dataset.train_data, None, smoother)
    original_model, original_train_preds, original_test_preds, original_train_logits, original_extra_logits = fit_and_predict_original(
        args, dataset, wandb_logger)
    reset_labels(dataset.train_data, dataset.orig_train_data)

    # Training on combined data
    wandb_logger = WandbLogger(project="stability", prefix="combined")
    dataset.merge_train_and_extra_data()
    smoother = label_smoothers.create(args.new_smoother.name, **args.new_smoother.params,
                                      num_classes=dataset.num_classes)
    smooth_labels(dataset.train_data, torch.cat([original_train_logits, original_extra_logits]), smoother)
    new_train_preds, new_test_preds = fit_and_predict_new(args, dataset, original_model, wandb_logger)

    log_final_metrics(dataset, new_test_preds, new_train_preds, original_test_preds, original_train_preds)

    # Save predictions
    if args.misc.save_predictions:
        save_predictions(original_train_preds, original_test_preds, new_train_preds, new_test_preds)


if __name__ == "__main__":
    main()