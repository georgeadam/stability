import torch.nn

from .creation import losses
from .loss import LossInterface


class CELoss(LossInterface):
    def __init__(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(self, out, y):
        return self.loss_fn(out, y)


losses.register_builder("ce", CELoss)