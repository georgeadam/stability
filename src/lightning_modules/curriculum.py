import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from src.lr_schedulers import lr_schedulers
from src.optimizers import optimizers
from .creation import lightning_modules


class Curriculum(LightningModule):
    def __init__(self, model, original_model, optimizer, lr_scheduler):
        super().__init__()

        self.model = model
        self.original_model = original_model
        self.loss = torch.nn.CrossEntropyLoss()
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

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
        self.log('train/accuracy', metrics["acc"], on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def training_epoch_end(self, outputs):
        outputs = self._stack_outputs(outputs)
        self.training_outputs = outputs

        acc = (outputs["preds"] == outputs["y"]).mean()
        self.trainer.train_dataloader.sampler.update(acc)

        for callback in self.trainer.callbacks:
            if "CurriculumEarlyStopping" in str(callback.__class__):
                callback.epochs_until_full = self.trainer.train_dataloader.sampler.epochs_until_full

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
                        stacked_outputs[k].append(out[k].unsqueeze(0).cpu().numpy())
                    else:
                        stacked_outputs[k].append(out[k].cpu().numpy())
                elif isinstance(out[k], float):
                    stacked_outputs[k].append(np.array([out[k]]))
                else:
                    stacked_outputs[k].append(out[k])

        stacked_outputs = {k: np.concatenate(v) for k, v in stacked_outputs.items()}

        return stacked_outputs

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
            stats = self._get_stats(x, index, source, logits, y)

        stats["loss"] = loss

        return stats

    def _get_loss(self, logits, y):
        return self.loss(logits, y)

    def _get_stats(self, x, index, source, logits, y):
        preds = torch.argmax(logits, dim=1)
        probs = torch.nn.Softmax(dim=1)(logits)

        if len(y.shape) > 1:
            y = torch.argmax(y, dim=1)

        probs_full = probs
        probs_gt = probs[torch.arange(len(probs)), y]
        probs = probs[torch.arange(len(probs)), preds]

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
                "acc": acc, "original_preds": original_preds.cpu().numpy(), "source": source.cpu().numpy(),
                "probs": probs.cpu().numpy(), "probs_gt": probs_gt.cpu().numpy(),
                "probs_full": probs_full.cpu().numpy()}


lightning_modules.register_builder("curriculum", Curriculum)
