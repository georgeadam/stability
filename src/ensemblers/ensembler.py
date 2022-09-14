import abc


class Ensembler(metaclass=abc.ABCMeta):
    def __init__(self, models):
        self.models = models

    @abc.abstractmethod
    def predict(self, dataloader):
        raise NotImplementedError
