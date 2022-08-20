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

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y, idx = batch

        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        self.log('train/accuracy', acc, on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val/accuracy', summary='max')

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        self.log('val/accuracy', acc, on_step=False, on_epoch=True)
        self.log('val/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        x, y, idx = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)

        return preds, loss, acc


lightning_modules.register_builder("standard", Standard)
