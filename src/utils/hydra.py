import wandb
from pytorch_lightning.loggers import WandbLogger

from .wandb import WANDB_API_KEY


def get_wandb_run():
    wandb.login(key=WANDB_API_KEY)
    WandbLogger(project="stability")

    if wandb.run is None:
        return "missing_wandb"
    else:
        return wandb.run.id
