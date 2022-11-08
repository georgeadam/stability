import logging
import os

import hydra
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from pytorch_lightning.utilities.seed import seed_everything

from settings import ROOT_DIR
from src.callbacks import trackers
from src.data import datasets
from src.ensemblers import ensemblers
from src.lightning_modules import lightning_modules
from src.models import models
from src.utils.logging import log_final_metrics
from src.utils.save import save_predictions

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")


def fit_and_predict_original(args, dataset, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    models = []

    for i in range(args.num_models):
        model = create_model(args, dataset.num_classes, dataset.num_channels, dataset.height)
        module = create_module_original(args, model)
        callbacks = create_callbacks_original(args)
        trainer = create_trainer(args, list(callbacks.values()), logger)

        trainer.fit(module, datamodule=dataset)
        models.append(model)

    ensemble = ensemblers.create(args.ensembler.name, models=models, **args.ensembler.params)

    train_preds = ensemble.predict(dataset.train_dataloader_ordered()).detach().cpu().numpy()
    test_preds = ensemble.predict(dataset.test_dataloader()).detach().cpu().numpy()

    del ensemble
    garbage_collection_cuda()

    return train_preds, test_preds


def fit_and_predict_new(args, dataset, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    models = []

    for i in range(args.num_models):
        model = create_model(args, dataset.num_classes, dataset.num_channels, dataset.height)
        module = create_module_new(args, model)
        callbacks = create_callbacks_new(args)
        trainer = create_trainer(args, list(callbacks.values()), logger)

        trainer.fit(module, datamodule=dataset)
        models.append(model)

    ensemble = ensemblers.create(args.ensembler.name, models=models, **args.ensembler.params)

    train_preds = ensemble.predict(dataset.train_dataloader_ordered()).detach().cpu().numpy()
    test_preds = ensemble.predict(dataset.test_dataloader()).detach().cpu().numpy()

    del ensemble
    garbage_collection_cuda()

    return train_preds, test_preds


def create_model(args, num_classes, num_channels, height):
    return models.create(args.model.name, num_classes=num_classes, num_channels=num_channels, height=height,
                         **args.model.params)


def create_module_original(args, model):
    return lightning_modules.create(args.orig_module.name, model=model, original_model=None,
                                    **args.orig_module.params)


def create_module_new(args, model):
    return lightning_modules.create(args.new_module.name, model=model, original_model=None,
                                    **args.new_module.params)


def create_trainer(args, callbacks, logger):
    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      callbacks=callbacks,
                      deterministic=True,
                      gpus=1,
                      **args.trainer)

    return trainer


def create_callbacks_original(args):
    return {"early_stopping": EarlyStopping("val/loss", **args.callbacks.early_stopping),
            "flip_tracker": trackers.create("flip")}


def create_callbacks_new(args):
    return {"early_stopping": EarlyStopping("val/loss", **args.callbacks.early_stopping)}


@hydra.main(config_path=config_path, config_name="ensemble")
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
    original_train_preds, original_test_preds = fit_and_predict_original(args, dataset, wandb_logger)

    # Training on combined data
    dataset.merge_train_and_extra_data()
    wandb_logger = WandbLogger(project="stability", prefix="combined")
    new_train_preds, new_test_preds = fit_and_predict_new(args, dataset, wandb_logger)

    log_final_metrics(dataset, new_test_preds, new_train_preds, original_test_preds, original_train_preds)

    # Save predictions
    if args.misc.save_predictions:
        save_predictions(original_train_preds, original_test_preds, new_train_preds, new_test_preds)


if __name__ == "__main__":
    main()