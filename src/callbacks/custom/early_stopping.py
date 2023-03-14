from pytorch_lightning.callbacks import EarlyStopping

from ..creation import callbacks


def early_stopping(**kwargs):
    return EarlyStopping("val/loss", **kwargs)


callbacks.register_builder("early_stopping", early_stopping)