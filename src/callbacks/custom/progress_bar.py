from .creation import custom_callbacks
from pytorch_lightning.callbacks import TQDMProgressBar


class ProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate, process_position):
        super().__init__(refresh_rate, process_position)

    def print(self, *args, sep=" ", **kwargs):
        active_progress_bar = None

        if self._main_progress_bar is not None and not self.main_progress_bar.disable:
            active_progress_bar = self.main_progress_bar
        elif self._val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self._test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self._predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, **kwargs)

custom_callbacks.register_builder("progress_bar", ProgressBar)
