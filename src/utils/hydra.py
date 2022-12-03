import wandb
from pytorch_lightning.loggers import WandbLogger


def get_wandb_run():
    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    WandbLogger(project="stability")

    return wandb.run.id
