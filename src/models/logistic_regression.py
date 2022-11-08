import torch.nn as nn

from .creation import models


class LogisticRegression(nn.Module):
    def __init__(self, num_classes, num_channels, height):
        super().__init__()

        in_features = height * height * num_channels

        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        return self.fc(x)


models.register_builder("logistic_regression", LogisticRegression)
