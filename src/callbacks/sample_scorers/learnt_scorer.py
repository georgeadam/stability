import numpy as np
import pandas as pd

from ..creation import callbacks
from .scorer import Scorer


class LearntScorer(Scorer):
    def __init__(self, order, **kwargs):
        self.predictions = pd.DataFrame({"preds": [], "y": [],
                                         "correct": [], "epoch": [], "index": []})
        self.predictions = self.predictions.astype(int)
        self.order = order

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = pl_module.training_outputs

        outputs = {k: outputs[k] for k in self.predictions.keys()}
        outputs = pd.DataFrame(outputs)
        self.predictions = pd.concat([self.predictions, outputs])

    def generate_scores(self):
        self.predictions["adjusted"] = 1 - self.predictions["correct"]
        correct_to_end_indices = self.predictions[::-1].astype(int).groupby(["index"])["adjusted"].cumsum().eq(0)
        learnt = self.predictions[correct_to_end_indices.values[::-1]].groupby(["index"])["epoch"].min()

        # now we have the epoch at which a sample is first permanently learned, but this only works for samples that
        # are permanently learned, so we have to add the rest manually with the max value

        unique_indices = np.array(self.predictions["index"].unique())
        not_learnt = np.setdiff1d(unique_indices, learnt.index)
        max_epoch = self.predictions["epoch"].max()
        not_learnt = pd.Series(np.ones(len(not_learnt)) * max_epoch, index=not_learnt)

        combined = pd.concat([learnt, not_learnt])

        if self.order == "anticurriculum":
            return np.array(combined.index), - combined.values
        else:
            return np.array(combined.index), combined.values


callbacks.register_builder("learnt_scorer", LearntScorer)
