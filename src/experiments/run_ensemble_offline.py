import logging
import os

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from settings import ROOT_DIR
from src.data import datasets
from src.inferers import Prediction
from src.utils.hydra import get_wandb_run
from src.utils.load import get_checkpoints, load_model
from src.utils.logging import log_final_metrics

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


def predict(args, dataset, split):
    all_train_logits = []
    all_test_logits = []

    inferer = Prediction()

    for i in range(args.num_models):
        run = get_run(args.model_name, args.data_name, args.random_state, i)
        config = get_config(run)

        if dataset is None:
            dataset = datasets.create(config.data.name, **config.data.params)

        model = get_model(config, run, dataset, split)
        train_logits, _, _ = inferer.make_predictions(model, dataset.train_dataloader_ordered())
        test_logits, _, _ = inferer.make_predictions(model, dataset.test_dataloader())

        all_train_logits.append(train_logits)
        all_test_logits.append(test_logits)

    train_logits = np.stack(all_train_logits)
    train_logits = np.mean(train_logits, axis=0)
    train_preds = np.argmax(train_logits, axis=1)

    test_logits = np.stack(all_test_logits)
    test_logits = np.mean(test_logits, axis=0)
    test_preds = np.argmax(test_logits, axis=1)

    return train_preds, test_preds, dataset

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

    for run in runs:
        runs_processed += 1

        if runs_processed > 1:
            print("Found more than one unique run!")

    return run


def get_config(run):
    return OmegaConf.create(run.config)


@hydra.main(config_path=config_path, config_name="ensemble_offline")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

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
    original_train_preds, original_test_preds, dataset= predict(args, None, "orig")

    # Training on combined data
    dataset.merge_train_and_extra_data()
    new_train_preds, new_test_preds, _ = predict(args, dataset, "new")

    log_final_metrics(dataset, new_test_preds, new_train_preds, original_test_preds, original_train_preds)


if __name__ == "__main__":
    main()
