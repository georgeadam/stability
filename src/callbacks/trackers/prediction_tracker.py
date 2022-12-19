import pandas as pd
from pytorch_lightning.callbacks import Callback

from .creation import trackers


class PredictionTracker(Callback):
    def __init__(self):
        self.training_predictions = pd.DataFrame({"preds": [], "original_preds": [], "y": [], "probs": [],
                                         "correct": [], "epoch": [], "index": [], "source": []})
        self.validation_predictions = pd.DataFrame({"preds": [], "original_preds": [], "y": [], "probs": [],
                                         "correct": [], "epoch": [], "index": [], "source": []})

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.state.stage.name == "SANITY_CHECKING":
            return

        outputs = pl_module.training_outputs

        outputs = {k: outputs[k] for k in self.training_predictions.keys()}
        outputs = pd.DataFrame(outputs)
        self.training_predictions = pd.concat([self.training_predictions, outputs])

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.state.stage.name == "SANITY_CHECKING":
            return

        outputs = pl_module.validation_outputs

        outputs = {k: outputs[k] for k in self.validation_predictions.keys()}
        outputs = pd.DataFrame(outputs)
        self.validation_predictions = pd.concat([self.validation_predictions, outputs])


trackers.register_builder("prediction", PredictionTracker)
