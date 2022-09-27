from torchvision.models import mobilenet_v2

from .creation import models


def mobilenet_wrapper(num_channels, height, **kwargs):
    return mobilenet_v2(**kwargs)

models.register_builder("mobilenet", mobilenet_wrapper)
