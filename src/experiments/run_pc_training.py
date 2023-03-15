import logging
import os

import hydra
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from settings import ROOT_DIR
from src.callbacks import callbacks
from src.data import datasets
from src.lightning_modules import lightning_modules
from src.models import models
from src.trainers import Trainer
from src.utils.logging import log_final_metrics
from src.utils.save import save_predictions

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")


def fit_and_predict_original(args, dataset, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = create_model(args, dataset.num_classes, dataset.stats)
    module = create_module_original(args, model)
    callbacks = create_callbacks_original(args)
    trainer = create_trainer_orig(args, list(callbacks.values()), logger, "orig")

    trainer.fit(module, datamodule=dataset)

    train_logits = trainer.predict(module, dataloaders=dataset.train_dataloader_ordered())
    train_logits = torch.cat(train_logits)
    train_preds = torch.argmax(train_logits, dim=1).detach().cpu().numpy()

    test_logits = trainer.predict(module, dataloaders=dataset.test_dataloader())
    test_logits = torch.cat(test_logits)
    test_preds = torch.argmax(test_logits, dim=1).detach().cpu().numpy()

    extra_logits = trainer.predict(module, dataloaders=dataset.extra_dataloader())
    extra_logits = torch.cat(extra_logits)

    return model, callbacks, train_preds, test_preds, train_logits, extra_logits


def fit_and_predict_distill(args, dataset, original_model, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = create_model(args, dataset.num_classes, dataset.stats)
    module = create_module_distill(args, model, original_model)
    callbacks = create_callbacks_distill(args)
    trainer = create_trainer_distill(args, list(callbacks.values()), logger, "new")

    trainer.fit(module, train_dataloaders=dataset.train_dataloader(),
                val_dataloaders=dataset.val_dataloader())

    train_logits = trainer.predict(module, dataloaders=dataset.train_dataloader_ordered())
    train_logits = torch.cat(train_logits)
    train_preds = torch.argmax(train_logits, dim=1).detach().cpu().numpy()

    test_logits = trainer.predict(module, dataloaders=dataset.test_dataloader())
    test_logits = torch.cat(test_logits)
    test_preds = torch.argmax(test_logits, dim=1).detach().cpu().numpy()

    return train_preds, test_preds


def create_model(args, num_classes, dataset_stats):
    return models.create(args.model.name, num_classes=num_classes, **dataset_stats,
                         **args.model.params)


def create_module_original(args, model):
    return lightning_modules.create(args.orig_module.name, model=model, original_model=None, **args.orig_module.params)


def create_module_distill(args, model, original_model):
    return lightning_modules.create(args.distill_module.name, model=model, original_model=original_model,
                                    **args.distill_module.params)


def create_trainer_orig(args, callbacks, logger, split):
    trainer = Trainer(split=split,
                      logger=logger,
                      log_every_n_steps=1,
                      callbacks=callbacks,
                      deterministic=True,
                      gpus=1,
                      **args.orig_trainer)

    return trainer


def create_trainer_distill(args, callbacks, logger, split):
    trainer = Trainer(split=split,
                      logger=logger,
                      log_every_n_steps=1,
                      callbacks=callbacks,
                      deterministic=True,
                      gpus=1,
                      **args.distill_trainer)

    return trainer


def create_callbacks_original(args):
    callbacks_dict = {key: callbacks.create(value.name, **value.params) for key, value in args.callbacks.items()}
    callbacks_dict["flip_tracker"] = callbacks.create("flip_tracker")

    return callbacks_dict


def create_callbacks_distill(args):
    callbacks_dict = {key: callbacks.create(value.name, **value.params) for key, value in args.callbacks.items()}
    callbacks_dict["churn_tracker"] = callbacks.create("churn_tracker")

    return callbacks_dict


@hydra.main(config_path=config_path, config_name="pc_training")
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
    original_model, original_callbacks, original_train_preds, original_test_preds, original_train_logits, original_extra_logits = fit_and_predict_original(
        args, dataset, wandb_logger)
    dataset.train_data.targets = dataset.orig_train_data.targets

    # Combine train and extra data
    dataset.merge_train_and_extra_data()
    wandb_logger = WandbLogger(project="stability", prefix="combined")
    # Training on combined data
    new_train_preds, new_test_preds = fit_and_predict_distill(args, dataset, original_model, wandb_logger)

    log_final_metrics(dataset, new_test_preds, new_train_preds, original_test_preds, original_train_preds)

    # Save predictions
    if args.misc.save_predictions:
        save_predictions(original_train_preds, original_test_preds, new_train_preds, new_test_preds)


if __name__ == "__main__":
    main()
