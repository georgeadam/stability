from torchvision.models import resnet18

from .creation import models


def resnet18_wrapper(num_channels, height, **kwargs):
    return resnet18(**kwargs)

models.register_builder("resnet18", resnet18_wrapper)
