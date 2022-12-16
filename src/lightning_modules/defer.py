import torch
import wandb
from pytorch_lightning import LightningModule

from src.lr_schedulers import lr_schedulers
from src.optimizers import optimizers
from .creation import lightning_modules


class Defer(LightningModule):
    def __init__(self, model, optimizer, lr_scheduler):
        super().__init__()

        self.model = model
        self.loss = torch.nn.BCEWithLogitsLoss()
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        self.training_outputs = None
        self.validation_outputs = None

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y = batch

        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        metrics = self._get_all_metrics(batch)

        # Log loss and metric
        self.log('train/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('train/accuracy', metrics["acc"], on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val/accuracy', summary='max')

        metrics = self._get_all_metrics(batch)

        # Log loss and metric
        self.log('val/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('val/accuracy', metrics["acc"], on_step=False, on_epoch=True)
        self.log('val/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def configure_optimizers(self):
        optimizer = optimizers.create(self._optimizer.name, parameters=self.model.parameters(),
                                      **self._optimizer.params)
        lr_scheduler = lr_schedulers.create(self._lr_scheduler.name, optimizer=optimizer, **self._lr_scheduler.params)

        return [optimizer], [lr_scheduler]

    def _get_all_metrics(self, batch):
        x, y, index, source = batch
        logits = self.model(x)
        loss = self._get_loss(logits, y)

        with torch.no_grad():
            stats = self._get_stats(x, logits, y)

        stats["loss"] = loss

        return stats

    def _get_loss(self, logits, y):
        total_loss = 0

        for c in range(y.shape[1]):
            total_loss += self.loss(logits[:, c], y[:, c])

        return total_loss / logits.shape[1]

    def _get_stats(self, x, logits, y):
        probs = torch.nn.Sigmoid()(logits)
        preds = (probs > 0.5).int()

        acc = torch.mean((preds == y).float())

        return {"acc": acc}


lightning_modules.register_builder("defer", Defer)
