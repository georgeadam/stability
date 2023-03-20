import torch
from pytorch_lightning import LightningModule

from .inferer import Inferer


class StackingPrediction(Inferer):
    def make_predictions(self, model, dataloaders):
        module = Module(model)
        results = self.predict(module, dataloaders=dataloaders)

        preds = torch.cat(results)

        return preds.numpy()


class Module(LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def predict_step(self, batch, batch_idx):
        logits_base, logits_new, _ = batch
        logits_combined = torch.concat([logits_base, logits_new], dim=1)

        out = self.model(logits_combined)
        preds = torch.argmax(out, dim=1)

        return preds