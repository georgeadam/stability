import abc

from pytorch_lightning import Trainer


class Inferer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(enable_progress_bar=False, gpus=1, **kwargs)

    @abc.abstractmethod
    def make_predictions(self, *args):
        raise NotImplementedError