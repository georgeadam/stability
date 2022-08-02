import torch.nn as nn

from .creation import models


class LeNet(nn.Module):
    def __init__(self, num_classes, size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16 * size, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16 * size, out_channels=24 * size, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=384 * size, out_features=128 * size)
        self.fc2 = nn.Linear(in_features=128 * size, out_features=84)
        self.fc3 = nn.Linear(84, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        return self.fc3(x)


models.register_builder("lenet", LeNet)
