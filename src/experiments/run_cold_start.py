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
from src.data import datasets
from src.lightning_modules import lightning_modules
from src.models import models
from src.utils.save import save_predictions

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")


def fit_and_predict(args, dataset, logger):
    if args.misc.reset_random_state:
        seed_everything(args.misc.seed)

    model = models.create(args.model.name, num_classes=dataset.num_classes, **args.model.params)
    lightning_module = lightning_modules.create(args.lightning_module.name, model=model, **args.lightning_module.params)
    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      gpus=1,
                      callbacks=[EarlyStopping("val/loss", **args.callbacks.early_stopping)],
                      deterministic=True,
                      enable_checkpointing=False,
                      **args.trainer)
    trainer.fit(lightning_module, datamodule=dataset)
    logits = trainer.predict(lightning_module, datamodule=dataset)
    logits = torch.cat(logits)
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

    return preds


@hydra.main(config_path=config_path, config_name="cold_start")
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
    wandb_logger = WandbLogger(project="stability", prefix="initial", name="{}_{}_cold_start_shift_seed-{}".format(args.data.name,
                                                                                                        args.model.name,
                                                                                                        args.misc.seed))
    wandb_logger.experiment.config.update(cfg)
    original_preds = fit_and_predict(args, dataset, wandb_logger)

    # Training on combined data
    dataset.merge_train_and_extra_data()
    wandb_logger = WandbLogger(project="stability", prefix="combined")
    new_preds = fit_and_predict(args, dataset, wandb_logger)

    # Compute churn
    labels = dataset.labels
    overall_churn = (original_preds != new_preds).astype(float).mean()
    relevant_churn = ((original_preds == labels) & (new_preds != original_preds)).astype(float).mean()

    # Compute accuracy
    original_accuracy = (original_preds == labels).astype(float).mean()
    new_accuracy = (new_preds == labels).astype(float).mean()

    wandb_logger = WandbLogger(project="stability")
    wandb_logger.log_metrics({"overall_churn": overall_churn,
                              "relevant_churn": relevant_churn,
                              "original_accuracy": original_accuracy,
                              "new_accuracy": new_accuracy})

    # Save predictions
    save_predictions(original_preds, new_preds)


if __name__ == "__main__":
    main()
