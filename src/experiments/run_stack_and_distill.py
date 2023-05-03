import copy
import logging
import os

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from settings import ROOT_DIR
from src.callbacks import callbacks
from src.data import datasets
from src.data.stacking import StackingDataset
from src.data.transforms import transforms
from src.inferers import Prediction, StackingPrediction
from src.lightning_modules import lightning_modules
from src.models import models
from src.utils.hydra import get_wandb_run
from src.utils.load import get_checkpoints, load_model, remove_prefix
from src.utils.logging import log_final_metrics_defer

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


def predict(args, dataset, split):
    inferer = Prediction()

    logging.info("split: {}".format(split))
    run = get_run(args.model_name, args.data_name, args.random_state, args.seed)
    config = get_config(run)

    if dataset is None:
        dataset = datasets.create(config.data.name, **config.data.params)

    model = get_model(config, run, dataset, split)
    val_logits, _, _ = inferer.make_predictions(model, dataset.val_dataloader())
    test_logits, _, _ = inferer.make_predictions(model, dataset.test_dataloader())
    test_preds = np.argmax(test_logits, axis=1)

    return val_logits, test_logits, test_preds, dataset


def get_model(config, run, dataset, split):
    checkpoints_dir = os.path.join(ROOT_DIR, "results/{}/{}/{}/{}/checkpoints".format(
        config.data.name, config.model.name, config.experiment_name, run.id))
    checkpoints = get_checkpoints(checkpoints_dir)

    checkpoint = [checkpoint for checkpoint in checkpoints if checkpoint.startswith(split)][0]
    model = load_model(checkpoints_dir, checkpoint, config, dataset)

    return model


def get_run(model_name, data_name, random_state, seed):
    api = wandb.Api(timeout=6000, api_key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    runs = api.runs('georgeadam/stability', {"$and": [
        {
            "config.experiment_name.value": "cold_start_checkpoint",
            "config.misc.seed": seed,
            "config.data.name": data_name,
            "config.data.params.random_state": random_state,
            "config.model.name": model_name}]})

    runs_processed = 0
    run = None

    for run in runs:
        runs_processed += 1

        if runs_processed > 1:
            print("Found more than one unique run!")

    return run


def get_config(run):
    return OmegaConf.create(run.config)


@hydra.main(config_path=config_path, config_name="stack_and_distill")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    seed_everything(args.seed, workers=True)

    cfg = OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True
    )

    # Initial training
    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="stability", prefix="initial",
                               name="{}_{}_{}-{}".format(args.data_name,
                                                         args.model_name,
                                                         args.experiment_name,
                                                         args.random_state))
    wandb_logger.experiment.config.update(cfg)
    original_val_logits, original_test_logits, original_test_preds, dataset = predict(args, None, "orig")

    # Training on combined data
    dataset.merge_train_and_extra_data()
    new_val_logits, new_test_logits, new_test_preds, _ = predict(args, dataset, "new")

    if args.normalize:
        base_transform = transforms.create("scaler")
        base_transform.fit(original_val_logits)
        new_transform = transforms.create("scaler")
        new_transform.fit(new_val_logits)
    else:
        base_transform = None
        new_transform = None

    stacking_train_dataset = StackingDataset(original_val_logits, new_val_logits, dataset.val_labels)
    stacking_train_dataset.base_transform = base_transform
    stacking_train_dataset.new_transform = new_transform

    stacking_val_dataset = copy.deepcopy(stacking_train_dataset)
    train_indices, val_indices = train_test_split(np.arange(len(stacking_train_dataset)),
                                                  test_size=0.2,
                                                  random_state=args.random_state)
    stacking_train_dataset.logits_base = stacking_train_dataset.logits_base[train_indices]
    stacking_train_dataset.logits_new = stacking_train_dataset.logits_new[train_indices]
    stacking_train_dataset.labels = stacking_train_dataset.labels[train_indices]

    stacking_val_dataset.logits_base = stacking_val_dataset.logits_base[val_indices]
    stacking_val_dataset.logits_new = stacking_val_dataset.logits_new[val_indices]
    stacking_val_dataset.labels = stacking_val_dataset.labels[val_indices]

    stacking_test_dataset = StackingDataset(original_test_logits, new_test_logits, dataset.test_labels)
    stacking_test_dataset.base_transform = base_transform
    stacking_test_dataset.new_transform = new_transform

    stacking_train_dataloader = torch.utils.data.DataLoader(stacking_train_dataset, batch_size=32, shuffle=True)
    stacking_val_dataloader = torch.utils.data.DataLoader(stacking_val_dataset, batch_size=32, shuffle=False)
    stacking_test_dataloader = torch.utils.data.DataLoader(stacking_test_dataset, batch_size=32, shuffle=False)

    model = models.create("mlp", num_classes=dataset.num_classes, num_layers=args.num_layers)
    module = lightning_modules.create("stack_and_distill", model=model, alpha=args.alpha)
    model_checkpoint = ModelCheckpoint(save_top_k=1, monitor="val/loss", mode="min", save_weights_only=True)
    callbacks_list = [callbacks.create("early_stopping", patience=5),
                      model_checkpoint]
    trainer = Trainer(callbacks=callbacks_list,
                      max_epochs=100,
                      gpus=1,
                      enable_checkpointing=True)
    trainer.fit(module, train_dataloaders=stacking_train_dataloader, val_dataloaders=stacking_val_dataloader)

    inferer = StackingPrediction()
    model.load_state_dict(remove_prefix(torch.load(model_checkpoint.best_model_path)["state_dict"]))
    meta_test_preds = inferer.make_predictions(model, stacking_test_dataloader)

    log_final_metrics_defer(original_test_preds, new_test_preds, meta_test_preds, dataset.test_labels)


if __name__ == "__main__":
    main()
