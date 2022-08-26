import abc


class Annealer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, initial, epoch, max_epochs, *args):
        raise NotImplementedError
