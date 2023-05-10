import abc


class Sampler:
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size

    @property
    @abc.abstractmethod
    def indices(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def epochs_until_full(self):
        raise NotImplementedError


    @abc.abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError
