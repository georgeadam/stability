import numpy as np
import pandas as pd

from ..creation import callbacks
from .scorer import Scorer


class FlipScorer(Scorer):
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
        num_classes = self.predictions["y"].max() + 1
        unique_indices = self.predictions["index"].unique().astype(int)
        horizontal = np.zeros((len(unique_indices), self.predictions["epoch"].max() + 1))

        unique_indices = self.predictions["index"].unique().astype(int)
        index_to_ordered = {unique_indices[i]: i for i in range(len(unique_indices))}

        horizontal[self.predictions["index"].apply(lambda x: index_to_ordered[x]),
                   self.predictions["epoch"]] = self.predictions["preds"]

        shifted = np.roll(horizontal, -1, axis=1)
        shifted[:, -1] = -num_classes

        diff = horizontal - shifted
        diff *= (diff < num_classes)

        flips = np.sum(diff != 0, axis=1)

        if self.order == "anticurriculum":
            return unique_indices, - flips
        else:
            return unique_indices, flips


callbacks.register_builder("flip_scorer", FlipScorer)
