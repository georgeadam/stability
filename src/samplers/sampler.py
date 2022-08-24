import abc


class Sampler:
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size

    @property
    @abc.abstractmethod
    def indices(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self):
        raise NotImplementedError
