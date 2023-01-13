import abc


class LossInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, out, y):
        raise NotImplementedError