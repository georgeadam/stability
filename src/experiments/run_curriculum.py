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
from src.samplers import samplers
from src.trainers import Trainer
from src.utils.hydra import get_wandb_run
from src.utils.logging import log_final_metrics
from src.utils.save import save_predictions

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


def fit_and_predict_original(args, dataset, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = create_model(args, dataset.num_classes, dataset.stats)
    module = create_module_original(args, model)
    callbacks = create_callbacks_original(args)
    trainer = create_trainer(args, list(callbacks.values()), logger, "orig")

    trainer.fit(module, datamodule=dataset)

    train_logits = trainer.predict(module, dataloaders=dataset.train_dataloader_ordered())
    train_logits = torch.cat(train_logits)
    train_preds = torch.argmax(train_logits, dim=1).detach().cpu().numpy()

    test_logits = trainer.predict(module, dataloaders=dataset.test_dataloader())
    test_logits = torch.cat(test_logits)
    test_preds = torch.argmax(test_logits, dim=1).detach().cpu().numpy()

    return model, callbacks, train_preds, test_preds


def fit_and_predict_curriculum(args, dataset, original_model, original_acc, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = create_model(args, dataset.num_classes, dataset.stats)
    module = create_module_curriculum(args, model, original_model)
    sampler = create_sampler(args, len(dataset.train_data), original_size=len(dataset.orig_train_data),
                             update_size=len(dataset.extra_data), correct_percentage=original_acc)
    callbacks = create_callbacks_curriculum(args)
    trainer = create_trainer(args, list(callbacks.values()), logger, "new")

    trainer.fit(module, train_dataloaders=dataset.train_dataloader_curriculum(sampler),
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
    return lightning_modules.create(args.orig_module.name, model=model, original_model=None,
                                    **args.orig_module.params)


def create_module_curriculum(args, model, original_model):
    return lightning_modules.create(args.new_module.name, model=model, original_model=original_model,
                                    **args.new_module.params)


def create_trainer(args, callbacks, logger, split):
    trainer = Trainer(split=split,
                      logger=logger,
                      log_every_n_steps=1,
                      callbacks=callbacks,
                      deterministic=False,
                      gpus=1,
                      **args.trainer)

    return trainer


def create_callbacks_original(args):
    callbacks_dict = {key: callbacks.create(value.name, **value.params) for key, value in args.callbacks.items()}
    callbacks_dict["flip_tracker"] = callbacks.create("flip_tracker")
    callbacks_dict["scorer"] = callbacks.create(args.scorer.name, **args.scorer.params)

    return callbacks_dict


def create_callbacks_curriculum(args):
    callbacks_dict = {key: callbacks.create(value.name, **value.params) for key, value in args.callbacks.items()}
    callbacks_dict["early_stopping"] = callbacks.create("curriculum_early_stopping", monitor="val/loss",
                                                        epochs_until_full=float("inf"),
                                                        **args.callbacks.early_stopping.params)
    callbacks_dict["churn_tracker"] = callbacks.create("churn_tracker")

    return callbacks_dict


def create_sampler(args, dataset_size, **extra_args):
    return samplers.create(args.sampler.name, dataset_size=dataset_size, **extra_args, **args.sampler.params)


@hydra.main(config_path=config_path, config_name="curriculum")
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
    original_model, original_callbacks, original_train_preds, original_test_preds = fit_and_predict_original(args,
                                                                                                             dataset,
                                                                                                             wandb_logger)

    # Training on combined data
    acc = (original_train_preds == dataset.train_labels).mean()
    dataset.sort_samples_by_score(original_callbacks["scorer"])
    dataset.merge_train_and_extra_data()
    wandb_logger = WandbLogger(project="stability", prefix="combined")
    new_train_preds, new_test_preds = fit_and_predict_curriculum(args, dataset, original_model, acc, wandb_logger)

    log_final_metrics(dataset, new_test_preds, new_train_preds, original_test_preds, original_train_preds)

    # Save predictions
    if args.misc.save_predictions:
        save_predictions(original_train_preds, original_test_preds, new_train_preds, new_test_preds)


if __name__ == "__main__":
    main()
