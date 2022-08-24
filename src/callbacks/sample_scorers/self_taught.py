import numpy as np
import pandas as pd

from .creation import scorers
from .scorer import Scorer


class SelfTaughtScorer(Scorer):
    def __init__(self):
        self.predictions = pd.DataFrame({"preds": [], "y": [], "probs": [],
                                         "correct": [], "epoch": [], "index": []})
        self.predictions = self.predictions.astype(int)
        self.predictions = self.predictions.astype({"probs": float})

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = pl_module.training_outputs

        outputs = {k: outputs[k] for k in self.predictions.keys()}
        outputs = pd.DataFrame(outputs)
        self.predictions = pd.concat([self.predictions, outputs])

    def generate_scores(self):
        last_epoch_preds = self.predictions.loc[self.predictions["epoch"] == self.predictions["epoch"].max()]
        indices = last_epoch_preds["index"].to_numpy()
        probs = last_epoch_preds["probs"].to_numpy()

        return indices, 1 - probs


scorers.register_builder("self_taught", SelfTaughtScorer)