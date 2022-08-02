from torchvision.models import resnet18

from .creation import models

models.register_builder("resnet18", resnet18)
