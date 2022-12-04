import logging
import os
from collections import OrderedDict

import hydra
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from settings import ROOT_DIR
from src.combiners import combiners
from src.data import datasets
from src.lightning_modules import lightning_modules
from src.models import models
from src.utils.hydra import get_wandb_run
from src.utils.logging import log_final_metrics_frankenstein

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


def extract_predictions(dataloader, model):
    module = lightning_modules.create("prediction", model=model)
    trainer = Trainer(gpus=1, enable_checkpointing=False, enable_progress_bar=False)

    test_logits = trainer.predict(module, dataloaders=dataloader)
    test_logits = torch.cat(test_logits)
    test_preds = torch.argmax(test_logits, dim=1)

    return test_preds


@hydra.main(config_path=config_path, config_name="frankenstein")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    experiment_dir = os.path.join(ROOT_DIR, args.experiment_dir)
    base_checkpoint_path = os.path.join(experiment_dir, args.base_checkpoint_path)
    new_checkpoint_path = os.path.join(experiment_dir, args.new_checkpoint_path)

    experiment_config = OmegaConf.load(os.path.join(experiment_dir, "config.yaml"))

    cfg = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Initial training
    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="stability")
    wandb_logger.experiment.config.update(cfg)

    seed_everything(experiment_config.misc.seed, workers=True)
    dataset = datasets.create(experiment_config.data.name, **experiment_config.data.params)

    model_base = models.create(experiment_config.model.name, num_classes=dataset.num_classes,
                               num_channels=dataset.num_channels, height=dataset.height,
                               **experiment_config.model.params)
    model_new = models.create(experiment_config.model.name, num_classes=dataset.num_classes,
                              num_channels=dataset.num_channels, height=dataset.height,
                              **experiment_config.model.params)

    checkpoint_base = torch.load(os.path.join(experiment_dir, base_checkpoint_path))
    checkpoint_new = torch.load(os.path.join(experiment_dir, new_checkpoint_path))

    state_dict_base = remove_prefix(checkpoint_base["state_dict"])
    state_dict_new = remove_prefix(checkpoint_new["state_dict"])

    model_base.load_state_dict(state_dict_base)
    model_base.eval()

    model_new.load_state_dict(state_dict_new)
    model_new.eval()

    original_test_preds = extract_predictions(dataset.test_dataloader(), model_base).numpy()
    new_test_preds = extract_predictions(dataset.test_dataloader(), model_new).numpy()

    frankenstein_model = combiners.create(args.combiner.name,
                                          base_model=model_base,
                                          new_model=model_new,
                                          dataset=dataset,
                                          **args.combiner.params)

    print("Making Meta Preds")
    meta_test_preds = frankenstein_model.predict(dataset.test_dataloader())
    frankenstein_test_preds = frankenstein_model.predict(dataset.test_dataloader())

    log_final_metrics_frankenstein(dataset.test_labels, meta_test_preds, frankenstein_test_preds, new_test_preds,
                      original_test_preds)


if __name__ == "__main__":
    main()
