import torch
import wandb
from pytorch_lightning import LightningModule

from src.optimizers import optimizers
from .creation import lightning_modules
from .improved_kd import KDLoss


class StackAndDistill(LightningModule):
    def __init__(self, model, alpha):
        super().__init__()
        self.model = model
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.distill_loss = KDLoss("all")
        self.softmax = torch.nn.Softmax(dim=1)
        self.alpha = alpha

    def forward(self, batch):
        logits_base, logits_new, labels = batch

        logits_combined = torch.concat([logits_base, logits_new], dim=1)

        out = self.model(logits_combined)

        return out

    def training_step(self, batch, batch_idx):
        logits_base, logits_new, labels = batch
        logits_combined = torch.concat([logits_base, logits_new], dim=1)
        out = self.model(logits_combined)

        probs_base = self.softmax(logits_base)
        distill_loss = self.distill_loss(out, probs_base, labels)

        distill_zero = distill_loss == 0
        ce_loss = self.ce_loss(out, labels)

        ce_loss[distill_zero] /= (1 - self.alpha)
        loss = ((1 - self.alpha) * ce_loss) + (self.alpha * distill_loss)

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val_accuracy', summary='max')

        logits_base, logits_new, labels = batch
        logits_combined = torch.concat([logits_base, logits_new], dim=1)
        out = self.model(logits_combined)

        probs_base = self.softmax(logits_base)
        distill_loss = self.distill_loss(out, probs_base, labels)

        distill_zero = distill_loss == 0
        ce_loss = self.ce_loss(out, labels)

        ce_loss[distill_zero] /= (1 - self.alpha)
        loss = ((1 - self.alpha) * ce_loss) + (self.alpha * distill_loss)

        self.log('val/loss', loss, on_step=False, on_epoch=True)

        return loss.mean()

    def configure_optimizers(self):
        optimizer = optimizers.create("adam", parameters=self.model.parameters(),
                                      lr=0.001, weight_decay=0)

        return optimizer


lightning_modules.register_builder("stack_and_distill", StackAndDistill)
