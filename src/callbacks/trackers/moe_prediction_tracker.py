import numpy as np
from pytorch_lightning.callbacks import Callback

from .creation import trackers


class MOEPredictionTracker(Callback):
    def __init__(self):
        self.training_predictions = {"preds": [], "probs_full": []}
        self.validation_predictions = {"preds": [], "probs_full": []}

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.state.stage.name == "SANITY_CHECKING":
            return

        outputs = pl_module.training_outputs
        outputs = {k: outputs[k] for k in self.training_predictions.keys()}

        for k in self.training_predictions.keys():
            self.training_predictions[k].append(outputs[k])

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.state.stage.name == "SANITY_CHECKING":
            return

        outputs = pl_module.validation_outputs
        outputs = {k: outputs[k] for k in self.validation_predictions.keys()}

        for k in self.validation_predictions.keys():
            self.validation_predictions[k].append(outputs[k])

    def get_training_predictions(self):
        training_predictions = {key: np.stack(self.training_predictions[key], axis=1) for key in
                                self.training_predictions.keys()}

        return training_predictions

    def get_validation_predictions(self):
        validation_predictions = {key: np.stack(self.validation_predictions[key], axis=1) for key in
                                  self.validation_predictions.keys()}

        return validation_predictions


trackers.register_builder("moe_prediction", MOEPredictionTracker)
