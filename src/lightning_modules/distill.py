import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from .creation import lightning_modules


class Distill(LightningModule):
    def __init__(self, model, original_model, lr, alpha):
        super().__init__()

        self.model = model
        self.original_model = original_model
        self.lr = lr
        self.alpha = alpha
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.distill_loss = DistillLoss()

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y, idx = batch

        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        _, ce_loss, distill_loss, acc = self._get_preds_loss_accuracy(batch)
        total_loss = ((1 - self.alpha) * ce_loss) + (self.alpha * distill_loss)

        # Log loss and metric
        self.log('train/loss', total_loss, on_step=False, on_epoch=True)
        self.log('train/loss_ce', ce_loss, on_step=False, on_epoch=True)
        self.log('train/loss_distill', distill_loss, on_step=False, on_epoch=True)
        self.log('train/accuracy', acc, on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val_accuracy', summary='max')

        preds, ce_loss, distill_loss, acc = self._get_preds_loss_accuracy(batch)
        total_loss = ((1 - self.alpha) * ce_loss) + (self.alpha * distill_loss)

        # Log loss and metric
        self.log('val/loss', total_loss, on_step=False, on_epoch=True)
        self.log('val/loss_ce', ce_loss, on_step=False, on_epoch=True)
        self.log('val/loss_distill', distill_loss, on_step=False, on_epoch=True)
        self.log('val/accuracy', acc, on_step=False, on_epoch=True)
        self.log('val/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        x, y, idx = batch
        logits = self.model(x)
        probs = torch.nn.Softmax(dim=1)(logits)

        with torch.no_grad():
            old_logits = self.original_model(x)

        old_probs = torch.nn.Softmax(dim=1)(old_logits)
        preds = torch.argmax(logits, dim=1)
        ce_loss = self.ce_loss(logits, y)
        distill_loss = self.distill_loss(probs, old_probs, y)
        acc = accuracy(preds, y)

        return preds, ce_loss, distill_loss, acc


class DistillLoss(object):
    def __init__(self):
        pass

    def __call__(self, probs, old_probs, y_true):
        loss = old_probs * (- torch.log(probs))
        loss = torch.sum(loss, dim=1)

        if (loss > 0).sum() == 0:
            return torch.zeros_like(loss).sum()
        else:
            return loss[loss > 0].mean()


lightning_modules.register_builder("distill", Distill)
