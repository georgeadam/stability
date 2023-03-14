import pandas as pd
from pytorch_lightning.callbacks import Callback

from ..creation import callbacks


class FlipTracker(Callback):
    def __init__(self):
        self.prev_training_outputs = None
        self.prev_validation_outputs = None
        self.keys = ["preds", "y", "index"]

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = pl_module.training_outputs

        if self.prev_training_outputs is None:
            outputs = {k: outputs[k] for k in self.keys}
            self.prev_training_outputs = pd.DataFrame(outputs)
            return

        outputs = {k: outputs[k] for k in self.keys}
        outputs = pd.DataFrame(outputs)

        combined = pd.merge(self.prev_training_outputs, outputs, on="index", suffixes=["_prev", "_current"])
        flips = (combined["preds_prev"] == combined["y_prev"]) & (combined["preds_prev"] != combined["preds_current"])
        flips = flips.mean()
        self.prev_training_outputs = outputs

        pl_module.log('train/negative_flips', flips, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = pl_module.validation_outputs

        if self.prev_validation_outputs is None:
            outputs = {k: outputs[k] for k in self.keys}
            self.prev_validation_outputs = pd.DataFrame(outputs)
            return

        outputs = {k: outputs[k] for k in self.keys}
        outputs = pd.DataFrame(outputs)

        combined = pd.merge(self.prev_validation_outputs, outputs, on="index", suffixes=["_prev", "_current"])
        flips = (combined["preds_prev"] == combined["y_prev"]) & (combined["preds_prev"] != combined["preds_current"])
        flips = flips.mean()
        self.prev_validation_outputs = outputs

        pl_module.log('val/negative_flips', flips, on_step=False, on_epoch=True)


callbacks.register_builder("flip_tracker", FlipTracker)
