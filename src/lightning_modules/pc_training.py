import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from .creation import lightning_modules


class PCTraining(LightningModule):
    def __init__(self, model, original_model, lr, alpha, beta, gamma):
        super().__init__()

        self.model = model
        self.original_model = original_model
        self.lr = lr
        self.gamma = gamma
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.pc_loss = PCFocalLoss(alpha, beta)

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y, idx = batch

        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        _, ce_loss, pc_loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train/loss_ce', ce_loss, on_step=False, on_epoch=True)
        self.log('train/loss_pc', pc_loss, on_step=False, on_epoch=True)
        self.log('train/accuracy', acc, on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return ce_loss + (self.gamma * pc_loss)

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val_accuracy', summary='max')

        x, y, idx = batch
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
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        x, y, idx = batch
        logits = self.model(x)

        with torch.no_grad():
            old_logits = self.original_model(x)

        preds = torch.argmax(logits, dim=1)
        ce_loss = self.ce_loss(logits, y)
        pc_loss = self.pc_loss(logits, old_logits, y)
        acc = accuracy(preds, y)

        return preds, ce_loss, pc_loss, acc


class PCFocalLoss(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, new_logits, old_logits, y_true):
        logit_distance = torch.linalg.norm(new_logits - old_logits, ord=2, dim=1) ** 2
        logit_distance = logit_distance / 2
        old_pred = torch.argmax(old_logits, dim=1)

        indicator = (old_pred == y_true).int()
        loss = (self.alpha + self.beta * indicator) * logit_distance

        if (loss > 0).sum() == 0:
            return torch.zeros_like(loss).sum()
        else:
            return loss[loss > 0].mean()


lightning_modules.register_builder("pc_training", PCTraining)
