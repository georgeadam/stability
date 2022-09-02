import numpy as np
import pandas as pd

from .creation import scorers
from .scorer import Scorer


class SelfTaughtScorer(Scorer):
    def __init__(self, order, **kwargs):
        self.predictions = pd.DataFrame({"preds": [], "y": [], "probs_gt": [],
                                         "correct": [], "epoch": [], "index": []})
        self.predictions = self.predictions.astype(int)
        self.predictions = self.predictions.astype({"probs": float})
        self.order = order

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = pl_module.training_outputs

        outputs = {k: outputs[k] for k in self.predictions.keys()}
        outputs = pd.DataFrame(outputs)
        self.predictions = pd.concat([self.predictions, outputs])

    def generate_scores(self):
        last_epoch_preds = self.predictions.loc[self.predictions["epoch"] == self.predictions["epoch"].max()]
        indices = last_epoch_preds["index"].to_numpy()
        probs = last_epoch_preds["probs_gt"].to_numpy()

        if self.order == "anticurriculum":
            return indices, probs
        else:
            return indices, 1 - probs


scorers.register_builder("self_taught", SelfTaughtScorer)