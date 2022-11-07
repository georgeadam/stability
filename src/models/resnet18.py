import torch
from torchvision.models import resnet18, ResNet

from .creation import models


def forward_embedding(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    return x


ResNet.forward_embedding = forward_embedding


def resnet18_wrapper(num_channels, height, **kwargs):
    return resnet18(**kwargs)

models.register_builder("resnet18", resnet18_wrapper)
