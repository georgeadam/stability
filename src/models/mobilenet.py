from torchvision.models import mobilenet_v2, MobileNetV2

from .creation import models
import torch


def forward_embedding(self, x):
    # See note [TorchScript super()]
    x = self.features(x)
    # Cannot use "squeeze" as batch-size can be 1
    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)

    return x


def forward_classifier(self, x):
    return self.classifier(x)


MobileNetV2.forward_embedding = forward_embedding
MobileNetV2.forward_classifier = forward_classifier


def mobilenet_wrapper(num_channels, height, **kwargs):
    return mobilenet_v2(**kwargs)

models.register_builder("mobilenet", mobilenet_wrapper)
