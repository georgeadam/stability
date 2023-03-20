import torch

from .creation import models


class MLP(torch.nn.Module):
    def __init__(self, num_classes, num_layers):
        super().__init__()

        self.layers = self._create_layers(num_classes, num_layers)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return x

    def _create_layers(self, num_classes, num_layers):
        if num_layers == 0:
            layers = torch.nn.ModuleList([torch.nn.Linear(2 * num_classes, num_classes)])
        else:
            layers = torch.nn.ModuleList([torch.nn.Linear(2 * num_classes, 100),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(100, num_classes)])

        return layers


models.register_builder("mlp", MLP)