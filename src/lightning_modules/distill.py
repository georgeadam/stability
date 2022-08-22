import numpy as np
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

        self.training_outputs = None
        self.validation_outputs = None

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y, idx, source = batch

        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        metrics = self._get_all_metrics(batch)

        # Log loss and metric
        self.log('train/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('train/loss_ce', metrics["ce_loss"], on_step=False, on_epoch=True)
        self.log('train/loss_distill', metrics["distill_loss"], on_step=False, on_epoch=True)
        self.log('train/accuracy', metrics["acc"], on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val_accuracy', summary='max')

        metrics = self._get_all_metrics(batch)

        # Log loss and metric
        self.log('val/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('val/loss_ce', metrics["ce_loss"], on_step=False, on_epoch=True)
        self.log('val/loss_distill', metrics["distill_loss"], on_step=False, on_epoch=True)
        self.log('val/accuracy', metrics["acc"], on_step=False, on_epoch=True)
        self.log('val/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def training_epoch_end(self, outputs):
        outputs = self._stack_outputs(outputs)
        self.training_outputs = outputs

    def validation_epoch_end(self, outputs):
        outputs = self._stack_outputs(outputs)
        self.validation_outputs = outputs

    def _stack_outputs(self, outputs):
        names = list(outputs[0].keys())
        stacked_outputs = {k: [] for k in names}

        for out in outputs:
            for k in names:
                if isinstance(out[k], torch.Tensor):
                    if len(out[k].shape) == 0:
                        stacked_outputs[k].append(out[k].unsqueeze(0).detach().cpu().numpy())
                    else:
                        stacked_outputs[k].append(out[k].cpu().numpy())
                elif isinstance(out[k], float):
                    stacked_outputs[k].append(np.array([out[k]]))
                else:
                    stacked_outputs[k].append(out[k])

        stacked_outputs = {k: np.concatenate(v) for k, v in stacked_outputs.items()}

        return stacked_outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _get_all_metrics(self, batch):
        x, y, index, source = batch
        logits = self.model(x)

        loss, ce_loss, distill_loss = self._get_loss(x, logits, y)

        with torch.no_grad():
            stats = self._get_stats(x, index, source, logits, y)

        stats["ce_loss"] = ce_loss
        stats["distill_loss"] = distill_loss
        stats["loss"] = loss

        return stats

    def _get_loss(self, x, logits, y):
        probs = torch.nn.Softmax(dim=1)(logits)

        with torch.no_grad():
            old_logits = self.original_model(x)

        old_probs = torch.nn.Softmax(dim=1)(old_logits)

        ce_loss = self.ce_loss(logits, y)
        distill_loss = self.distill_loss(probs, old_probs, y)
        loss = ((1 - self.alpha) * ce_loss) + (self.alpha * distill_loss)

        return loss, ce_loss, distill_loss

    def _get_stats(self, x, index, source, logits, y):
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        correct = preds == y

        if self.original_model:
            original_logits = self.original_model(x)
            original_preds = torch.argmax(original_logits, dim=1)
        else:
            original_preds = preds

        epoch = np.array([self.current_epoch] * len(preds))

        return {"preds": preds.cpu().numpy(), "y": y.cpu().numpy(), "correct": correct.cpu().numpy(),
                "index": index.cpu().numpy(), "epoch": epoch,
                "acc": acc, "original_preds": original_preds.cpu().numpy(), "source": source.cpu().numpy()}

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
