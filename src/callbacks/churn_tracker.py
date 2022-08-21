from pytorch_lightning.callbacks import Callback

from src.utils.metrics import compute_overall_churn, compute_relevant_churn


class ChurnTracker(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        outputs = pl_module.outputs
        no_extra = outputs["source"] == 0

        overall_churn = compute_overall_churn(outputs["original_preds"][no_extra], outputs["preds"][no_extra])
        relevant_churn = compute_relevant_churn(outputs["original_preds"][no_extra], outputs["preds"][no_extra],
                                                outputs["y"][no_extra])

        pl_module.log('train/overall_churn', overall_churn, on_step=False, on_epoch=True)
        pl_module.log('train/relevant_churn', relevant_churn, on_step=False, on_epoch=True)
