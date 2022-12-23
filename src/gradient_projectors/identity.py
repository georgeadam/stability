import torch

from .creation import gradient_projectors
from .projector import ProjectorInterface
from .utils import parameters_to_grad_vector


class Identity(ProjectorInterface):
    def __init__(self, normalize):
        self.normalize = normalize

    def project(self, model, x, y, loss_fn):
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        gradient = parameters_to_grad_vector(model.parameters()).cpu()

        if self.normalize:
            gradient = gradient / torch.linalg.norm(gradient, ord=2)

        return gradient


gradient_projectors.register_builder("identity", Identity)
