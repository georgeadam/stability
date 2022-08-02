import logging
import os

import numpy as np
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
from src.data import datasets
from src.lightning_modules import lightning_modules
from src.models import models
from src.utils.save import save_predictions

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")


def fit_and_predict_original(args, dataset, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = create_model(args, dataset.num_classes)
    module = create_module_original(args, model)
    trainer = create_trainer(args, logger)
    trainer.fit(module, datamodule=dataset)

    test_logits = trainer.predict(module, dataloaders=dataset.test_dataloader())
    test_logits = torch.cat(test_logits)
    test_preds = torch.argmax(test_logits, dim=1).detach().cpu().numpy()

    train_logits = trainer.predict(module, dataloaders=dataset.train_dataloader())
    train_logits = torch.cat(train_logits)
    train_probs = torch.nn.Softmax(dim=1)(train_logits).detach().cpu().numpy()

    return train_probs, test_preds


def fit_and_predict_distill(args, dataset, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = create_model(args, dataset.num_classes)
    module = create_module_distill(args, model)
    trainer = create_trainer(args, logger)
    trainer.fit(module, datamodule=dataset)

    test_logits = trainer.predict(module, dataloaders=dataset.test_dataloader())
    test_logits = torch.cat(test_logits)
    test_preds = torch.argmax(test_logits, dim=1).detach().cpu().numpy()

    return test_preds


def create_model(args, num_classes):
    return models.create(args.model.name, num_classes=num_classes, **args.model.params)


def create_module_original(args, model):
    return lightning_modules.create(args.orig_module.name, model=model, **args.orig_module.params)

def create_module_distill(args, model):
    return lightning_modules.create(args.distill_module.name, model=model, **args.distill_module.params)

def create_trainer(args, logger):
    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      callbacks=[EarlyStopping("val/loss", **args.callbacks.early_stopping)],
                      deterministic=True,
                      gpus=1,
                      enable_checkpointing=False,
                      **args.trainer)

    return trainer


@hydra.main(config_path=config_path, config_name="distill")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    seed_everything(args.misc.seed)
    dataset = datasets.create(args.data.name, **args.data.params)

    cfg = OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True
    )

    # Initial training
    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="stability", prefix="initial",
                               name="{}_{}_distillation_{}-{}".format(args.data.name,
                                                                      args.model.name,
                                                                      args.distill_module.params.alpha,
                                                                      args.misc.seed))
    wandb_logger.experiment.config.update(cfg)
    original_train_probs, original_test_preds = fit_and_predict_original(args, dataset, wandb_logger)

    # Training on combined data
    dataset.augment_train_data(original_train_probs)
    dataset.augment_extra_data(np.zeros_like(original_train_probs))
    dataset.merge_train_and_extra_data()
    wandb_logger = WandbLogger(project="stability", prefix="combined")
    new_test_preds = fit_and_predict_distill(args, dataset, wandb_logger)

    # Compute churn
    labels = dataset.labels
    overall_churn = (original_test_preds != new_test_preds).astype(float).mean()
    relevant_churn = ((original_test_preds == labels) & (new_test_preds != original_test_preds)).astype(float).mean()

    # Compute accuracy
    original_accuracy = (original_test_preds == labels).astype(float).mean()
    new_accuracy = (new_test_preds == labels).astype(float).mean()

    wandb_logger = WandbLogger(project="stability")
    wandb_logger.log_metrics({"overall_churn": overall_churn,
                              "relevant_churn": relevant_churn,
                              "original_accuracy": original_accuracy,
                              "new_accuracy": new_accuracy})

    # Save predictions
    # save_predictions(original_preds, new_preds)


if __name__ == "__main__":
    main()
