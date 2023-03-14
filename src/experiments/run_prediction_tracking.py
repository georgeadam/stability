import logging
import os

import hydra
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from settings import ROOT_DIR
from src.callbacks import callbacks
from src.data import datasets
from src.lightning_modules import lightning_modules
from src.models import models
from src.trainers import Trainer

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")


def fit_and_predict_original(args, dataset, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = create_model(args, dataset.num_classes, dataset.num_channels, dataset.height)
    module = create_module_original(args, model)
    callbacks = create_callbacks_original(args)
    trainer = create_trainer(args, list(callbacks.values()), logger, "orig")

    trainer.fit(module, datamodule=dataset)
    logger.log_table("train/predictions", dataframe=callbacks["prediction_tracker"].training_predictions)
    logger.log_table("val/predictions", dataframe=callbacks["prediction_tracker"].validation_predictions)

    train_logits = trainer.predict(module, dataloaders=dataset.train_dataloader_ordered())
    train_logits = torch.cat(train_logits)
    train_preds = torch.argmax(train_logits, dim=1).detach().cpu().numpy()

    test_logits = trainer.predict(module, dataloaders=dataset.test_dataloader())
    test_logits = torch.cat(test_logits)
    test_preds = torch.argmax(test_logits, dim=1).detach().cpu().numpy()

    return model, train_preds, test_preds


def create_model(args, num_classes, num_channels, height):
    return models.create(args.model.name, num_classes=num_classes, num_channels=num_channels, height=height,
                         **args.model.params)


def create_module_original(args, model):
    return lightning_modules.create(args.lightning_module.name, model=model, original_model=None,
                                    **args.lightning_module.params)


def create_trainer(args, callbacks, logger, split):
    trainer = Trainer(split=split,
                      logger=logger,
                      log_every_n_steps=1,
                      callbacks=callbacks,
                      deterministic=True,
                      gpus=1,
                      **args.trainer)

    return trainer


def create_callbacks_original(args):
    callbacks_dict = {key: callbacks.create(value.name, **value.params) for key, value in args.callbacks.items()}
    callbacks_dict["flip_tracker"] = callbacks.create("flip_tracker")
    callbacks_dict["prediction_tracker"] = callbacks.create("prediction_tracker")

    return callbacks_dict


@hydra.main(config_path=config_path, config_name="prediction_tracking")
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
                               name="{}_{}_{}-{}".format(args.data.name,
                                                         args.model.name,
                                                         args.experiment_name,
                                                         args.misc.seed))
    wandb_logger.experiment.config.update(cfg)
    original_model, original_train_preds, original_test_preds = fit_and_predict_original(args, dataset, wandb_logger)


if __name__ == "__main__":
    main()
