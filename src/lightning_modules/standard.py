import pandas as pd
import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from .creation import lightning_modules


class Standard(LightningModule):
    def __init__(self, model, lr):
        super().__init__()

        self.model = model
        self.lr = lr
        self.loss = torch.nn.CrossEntropyLoss()

        self.predictions = pd.DataFrame({"preds": [], "y": [], "correct": [], "epoch": [], "index": []})

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y, idx = batch

        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        metrics = self._get_metrics(batch)

        # Log loss and metric
        self.log('train/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('train/accuracy', metrics["acc"], on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def training_epoch_end(self, outputs):
        stacked_outputs = {k: [] for k in self.predictions.columns}
        for out in outputs:
            for k in self.predictions.columns:
                stacked_outputs[k].append(out[k])

        stacked_outputs = {k: torch.concat(v) for k, v in stacked_outputs.items()}
        stacked_outputs = pd.DataFrame(stacked_outputs)
        self.predictions = pd.concat([self.predictions, stacked_outputs])

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val/accuracy', summary='max')

        metrics = self._get_metrics(batch)

        # Log loss and metric
        self.log('val/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('val/accuracy', metrics["acc"], on_step=False, on_epoch=True)
        self.log('val/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_metrics(self, batch):
        x, y, index = batch
        logits = self.model(x)
        loss = self.loss(logits, y)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y)
            correct = preds == y

        epoch = torch.tensor([self.current_epoch] * len(preds))

        return {"preds": preds.cpu(), "y": y.cpu(), "correct": correct.cpu(),
                "loss": loss, "acc": acc, "index": index.cpu(), "epoch": epoch}


lightning_modules.register_builder("standard", Standard)
