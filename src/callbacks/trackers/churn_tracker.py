from pytorch_lightning.callbacks import Callback

from src.utils.metrics import compute_overall_churn, compute_relevant_churn
from .creation import trackers


class ChurnTracker(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        self._log_churn(pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_churn(pl_module, "val")

    def _log_churn(self, pl_module, partition):
        if partition == "train":
            outputs = pl_module.training_outputs
        elif partition == "val":
            outputs = pl_module.validation_outputs

        no_extra = outputs["source"] == 0

        overall_churn = compute_overall_churn(outputs["original_preds"][no_extra], outputs["preds"][no_extra])
        relevant_churn = compute_relevant_churn(outputs["original_preds"][no_extra], outputs["preds"][no_extra],
                                                outputs["y"][no_extra])

        pl_module.log('{}/overall_churn'.format(partition), overall_churn, on_step=False, on_epoch=True)
        pl_module.log('{}/relevant_churn'.format(partition), relevant_churn, on_step=False, on_epoch=True)


trackers.register_builder("churn", ChurnTracker)
