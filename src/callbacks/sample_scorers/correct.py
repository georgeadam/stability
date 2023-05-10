import pandas as pd

from .scorer import Scorer
from ..creation import callbacks


class CorrectScorer(Scorer):
    def __init__(self, **kwargs):
        self.predictions = pd.DataFrame({"preds": [], "y": [],
                                         "correct": [], "epoch": [], "index": []})
        self.predictions = self.predictions.astype(int)

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = pl_module.training_outputs

        outputs = {k: outputs[k] for k in self.predictions.keys()}
        outputs = pd.DataFrame(outputs)
        self.predictions = pd.concat([self.predictions, outputs])

    def generate_scores(self):
        final_preds = self.predictions.loc[self.predictions["epoch"] == self.predictions["epoch"].max()]["preds"]
        labels = self.predictions.loc[self.predictions["epoch"] == self.predictions["epoch"].max()]["y"]

        unique_indices = self.predictions.loc[self.predictions["epoch"] == self.predictions["epoch"].max()]["index"].astype(int)
        incorrect = (final_preds != labels).astype(int)

        return unique_indices.to_numpy(), incorrect.to_numpy()


callbacks.register_builder("correct_scorer", CorrectScorer)
