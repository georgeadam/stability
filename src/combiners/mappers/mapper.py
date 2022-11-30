import abc


class MapperInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, x, y):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError


