import torch

from .creation import models


class FC(torch.nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()

        self.fc1 = torch.nn.Linear(num_features, 100)
        self.fc2 = torch.nn.Linear(100, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def forward_embedding(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        return x

    def forward_classifier(self, x):
        x = self.fc2(x)

        return x

models.register_builder("fc", FC)
