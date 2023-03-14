from pytorch_lightning import Trainer as LightningTrainer


class Trainer(LightningTrainer):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split