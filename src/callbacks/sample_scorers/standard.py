from ..creation import callbacks
from .scorer import Scorer


class StandardScorer(Scorer):
    def __init__(self, **args):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def generate_scores(self):
        return None, None


callbacks.register_builder("standard_scorer", StandardScorer)
