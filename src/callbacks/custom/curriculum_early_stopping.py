from pytorch_lightning.callbacks import EarlyStopping

from ..creation import callbacks


class CurriculumEarlyStopping(EarlyStopping):
    def __init__(self, epochs_until_full, **kwargs):
        super().__init__(**kwargs)

        self.epochs_until_full = epochs_until_full

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch >= self.epochs_until_full:
            self._run_early_stopping_check(trainer)

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass


callbacks.register_builder("curriculum_early_stopping", CurriculumEarlyStopping)
