import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from .creation import lightning_modules


class Distill(LightningModule):
    def __init__(self, model, lr, alpha):
        super().__init__()

        self.model = model
        self.lr = lr
        self.alpha = alpha
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.distill_loss = DistillLoss()

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y = batch

        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        _, ce_loss, distill_loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train/loss_ce', ce_loss, on_step=False, on_epoch=True)
        self.log('train/loss_distill', distill_loss, on_step=False, on_epoch=True)
        self.log('train/accuracy', acc, on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return ((1 - self.alpha) * ce_loss) + (self.alpha * distill_loss)

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val_accuracy', summary='max')

        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        ce_loss = self.ce_loss(logits, y)
        acc = accuracy(preds, y)

        # Log loss and metric
        self.log('val/loss', ce_loss, on_step=False, on_epoch=True)
        self.log('val/accuracy', acc, on_step=False, on_epoch=True)
        self.log('val/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        x, y, knowledge = batch
        logits = self.model(x)
        probs = torch.nn.Softmax(dim=1)(logits)
        preds = torch.argmax(logits, dim=1)
        ce_loss = self.ce_loss(logits, y)
        distill_loss = self.distill_loss(probs, knowledge, y)
        acc = accuracy(preds, y)

        return preds, ce_loss, distill_loss, acc


class DistillLoss(object):
    def __init__(self):
        pass

    def __call__(self, y_pred, knowledge, y_true):
        loss = knowledge * (- torch.log(y_pred))
        loss = torch.sum(loss, dim=1)

        if (loss > 0).sum() == 0:
            return torch.zeros_like(loss).sum()
        else:
            return loss[loss > 0].mean()


lightning_modules.register_builder("distill", Distill)
