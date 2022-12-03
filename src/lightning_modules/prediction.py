from pytorch_lightning import LightningModule

from .creation import lightning_modules


class Prediction(LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y, idx, source = batch

        logits = self.model(x)

        return logits


lightning_modules.register_builder("prediction", Prediction)
