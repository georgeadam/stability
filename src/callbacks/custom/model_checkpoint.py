from pytorch_lightning.callbacks import ModelCheckpoint as LightningModelCheckpoint

from ..creation import callbacks


class ModelCheckpoint(LightningModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, monitor="val/loss", mode="min")

    def setup(self, trainer, pl_module, stage=None) -> None:
        self.dirpath = "checkpoints"
        self.filename = "{}".format(trainer.split) + "-{epoch}"
        self.CHECKPOINT_NAME_LAST = "{}".format(trainer.split) + "-{epoch}"
        super().setup(trainer, pl_module, stage)


callbacks.register_builder("model_checkpoint", ModelCheckpoint)
