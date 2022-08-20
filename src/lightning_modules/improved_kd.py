import copy
import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from .creation import lightning_modules


class ImprovedKD(LightningModule):
    def __init__(self, model, original_model, lr, alpha, focal, warm_start, *args, **kwargs):
        super().__init__()

        if warm_start:
            self.model = copy.deepcopy(original_model)
        else:
            self.model = model

        self.original_model = original_model
        self.lr = lr
        self.alpha = alpha
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.distill_loss = KDLoss(focal)

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y, idx = batch

        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        _, ce_loss, distill_loss, acc = self._get_preds_loss_accuracy(batch)
        distill_zero = distill_loss == 0
        ce_loss[distill_zero] /= (1 - self.alpha)
        ce_loss = ce_loss.mean()
        distill_loss = distill_loss.mean()
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
        distill_zero = distill_loss == 0
        ce_loss[distill_zero] /= (1 - self.alpha)
        ce_loss = ce_loss.mean()
        distill_loss = distill_loss.mean()
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

        if len(y.shape) > 1:
            y = torch.argmax(y, dim=1)
        acc = accuracy(preds, y)

        return preds, ce_loss, distill_loss, acc


class KDLoss(object):
    def __init__(self, focal):
        self.focal = focal

    def __call__(self, probs, old_probs, y_true):
        loss = old_probs * (- torch.log(probs))
        loss = torch.sum(loss, dim=1)

        old_preds = torch.argmax(old_probs, dim=1)

        if len(y_true.shape) > 1:
            y_true = torch.argmax(y_true, dim=1)
        correct = (old_preds == y_true)

        if self.focal:
            loss[correct] *= 0

        return loss


lightning_modules.register_builder("improved_kd", ImprovedKD)
