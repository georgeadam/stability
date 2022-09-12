import abc


class Combiner(metaclass=abc.ABCMeta):
    def __init__(self, base_model, new_model, dataset):
        self.base_model = base_model
        self.new_model = new_model
        self.dataset = dataset

    @abc.abstractmethod
    def predict(self, dataloader):
        raise NotImplementedError

    def _setup(self):
        raise NotImplementedError
