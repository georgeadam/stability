from torchvision.models import resnet50

from .creation import models


def resnet50_wrapper(num_channels, height, **kwargs):
    return resnet50(**kwargs)

models.register_builder("resnet50", resnet50_wrapper)
