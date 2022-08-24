from .creation import scorers
from .scorer import Scorer


class StandardScorer(Scorer):
    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def generate_scores(self):
        return None, None


scorers.register_builder("standard", StandardScorer)
