import abc


class ScorerInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_scores(self, predictions):
        raise NotImplementedError
