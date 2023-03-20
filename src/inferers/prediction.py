import torch
from pytorch_lightning import LightningModule

from .inferer import Inferer


class Prediction(Inferer):
    def make_predictions(self, model, dataloaders):
        module = Module(model)
        results = self.predict(module, dataloaders=dataloaders)

        logits = [results[i][0] for i in range(len(results))]
        logits = torch.cat(logits)

        probs = [results[i][1] for i in range(len(results))]
        probs = torch.cat(probs)

        preds = [results[i][2] for i in range(len(results))]
        preds = torch.cat(preds)

        return logits.numpy(), probs.numpy(), preds.numpy()


class Module(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)

    def predict_step(self, batch, batch_idx):
        x, y, _, _ = batch
        logits = self.model(x)
        probs = self.softmax(logits)
        preds = torch.argmax(logits, dim=1)

        return logits, probs, preds

