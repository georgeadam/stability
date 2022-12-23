import abc


class ProjectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def project(self, model, x, y, loss_fn):
        raise NotImplementedError
