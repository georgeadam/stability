from torchvision.models import mobilenet_v3_small

from .creation import models


def mobilenet_wrapper(num_channels, height, **kwargs):
    return mobilenet_v3_small(**kwargs)

models.register_builder("mobilenet", mobilenet_wrapper)
