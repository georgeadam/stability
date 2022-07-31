import logging
import os

import hydra
import torch
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
    seed_everything(args.misc.seed)
    model = models.create(args.model.name, output_dim=dataset.output_dim, **args.model.params)
    lightning_module = lightning_modules.create(args.lightning_module.name, model=model, **args.lightning_module.params)
    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      callbacks=[EarlyStopping("val_loss", **args.callbacks.early_stopping)],
                      deterministic=True,
                      **args.trainer)
    trainer.fit(lightning_module, datamodule=dataset)
    logits = trainer.predict(lightning_module, datamodule=dataset)
    preds = torch.argmax(logits, dim=1)
    preds = torch.cat(preds).detach().cpu().numpy()

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
    wandb_logger = WandbLogger(project="stability", prefix="initial", name="{}_{}_cold_start-{}".format(args.data.name,
                                                                                                        args.model.name,
                                                                                                        args.misc.seed))
    wandb_logger.experiment.config.update(cfg)
    original_preds = fit_and_predict(args, dataset, wandb_logger)

    # Training on combined data
    wandb_logger = WandbLogger(project="stability", prefix="combined")
    new_preds = fit_and_predict(args, dataset, wandb_logger)

    # Compute churn
    labels = dataset.labels
    overall_churn = (original_preds != new_preds).astype(float).mean()
    relevant_churn = ((original_preds == labels) & (new_preds != original_preds)).astype(float).mean()

    wandb_logger = WandbLogger(project="stability")
    wandb_logger.log_metrics({"overall_churn": overall_churn,
                              "relevant_churn": relevant_churn})

    # Save predictions
    save_predictions(original_preds, new_preds)


if __name__ == "__main__":
    main()
