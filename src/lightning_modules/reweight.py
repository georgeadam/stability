import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from .creation import lightning_modules


class Reweight(LightningModule):
    def __init__(self, model, original_model, lr, weight, focal, normalize):
        super().__init__()

        self.model = model
        self.original_model = original_model
        self.lr = lr
        self.weight = weight
        self.focal = focal
        self.normalize = normalize
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

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
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        x, y, idx = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)

        # we have a few cases: flip, both right, new wrong
        original_preds = torch.argmax(self.original_model(x), dim=1)
        weights = torch.ones_like(loss)

        if self.focal:
            negative_flips = (original_preds == y) & (preds != y)
            weights[negative_flips] *= self.weight
        else:
            original_correct = (original_preds == y)
            weights[original_correct] *= self.weight

        if self.normalize:
            weights = weights / weights.mean()

        loss = (loss * weights).mean()

        return preds, loss, acc


lightning_modules.register_builder("reweight", Reweight)
