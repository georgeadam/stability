from pytorch_lightning.callbacks import Callback

import pandas as pd


class PredictionTracker(Callback):
    def __init__(self):
        self.predictions = pd.DataFrame({"preds": [], "original_preds": [], "y": [],
                                         "correct": [], "epoch": [], "index": [], "source": []})

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = pl_module.outputs

        outputs = {k: outputs[k] for k in self.predictions.keys()}
        outputs = pd.DataFrame(outputs)
        self.predictions = pd.concat([self.predictions, outputs])

