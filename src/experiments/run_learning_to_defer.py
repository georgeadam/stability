import logging
import os
from collections import OrderedDict

import hydra
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from settings import ROOT_DIR
from src.combiners import combiners
from src.data import datasets
from src.models import models
from src.utils.hydra import get_wandb_run
from src.utils.inference import extract_predictions
from src.utils.logging import log_final_metrics_defer
from src.utils.wandb import WANDB_API_KEY

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


def remove_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "original_" in k:
            continue

        name = k.replace("model.", "")
        new_state_dict[name] = v

    return new_state_dict


@hydra.main(config_path=config_path, config_name="learning_to_defer")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    experiment_dir = os.path.join(ROOT_DIR, args.experiment_dir)
    base_checkpoint_path = os.path.join(experiment_dir, args.base_checkpoint_path)
    new_checkpoint_path = os.path.join(experiment_dir, args.new_checkpoint_path)

    experiment_config = OmegaConf.load(os.path.join(experiment_dir, "config.yaml"))

    cfg = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Initial training
    wandb.login(key=WANDB_API_KEY)
    wandb_logger = WandbLogger(project="stability")
    wandb_logger.experiment.config.update(cfg)

    seed_everything(experiment_config.misc.seed, workers=True)
    dataset = datasets.create(experiment_config.data.name, **experiment_config.data.params)

    model_base = models.create(experiment_config.model.name, num_classes=dataset.num_classes,
                               **dataset.stats,
                               **experiment_config.model.params)
    model_new = models.create(experiment_config.model.name, num_classes=dataset.num_classes,
                              **dataset.stats,
                              **experiment_config.model.params)

    checkpoint_base = torch.load(os.path.join(experiment_dir, base_checkpoint_path))
    checkpoint_new = torch.load(os.path.join(experiment_dir, new_checkpoint_path))

    state_dict_base = remove_prefix(checkpoint_base["state_dict"])
    state_dict_new = remove_prefix(checkpoint_new["state_dict"])

    model_base.load_state_dict(state_dict_base)
    model_base.eval()

    model_new.load_state_dict(state_dict_new)
    model_new.eval()

    base_test_preds = extract_predictions(dataset.test_dataloader(), model_base)
    new_test_preds = extract_predictions(dataset.test_dataloader(), model_new)

    combiner = combiners.create(args.combiner.name, base_model=model_base, new_model=model_new, dataset=dataset,
                                **args.combiner.params)
    meta_test_preds = combiner.predict(dataset.test_dataloader())

    log_final_metrics_defer(base_test_preds, new_test_preds, meta_test_preds, dataset.test_labels)


if __name__ == "__main__":
    main()
