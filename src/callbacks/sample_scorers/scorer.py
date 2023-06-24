import abc

from pytorch_lightning.callbacks import Callback


class Scorer(Callback, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def on_train_epoch_end(self, trainer, pl_module):
        raise NotImplementedError

    @abc.abstractmethod
    def generate_scores(self):
        raise NotImplementedError
