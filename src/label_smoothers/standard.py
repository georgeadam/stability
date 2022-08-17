import torch
from torch.nn.functional import one_hot

from .creation import label_smoothers


class StandardLabelSmoother:
    def __init__(self, alpha, num_classes):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, y, **kwargs):
        if len(y.shape) == 1:
            expanded_y = one_hot(y, self.num_classes)
        else:
            expanded_y = y

        return (1 - self.alpha) * expanded_y + (self.alpha / (self.num_classes)) * (torch.ones_like(expanded_y))


label_smoothers.register_builder("standard", StandardLabelSmoother)
