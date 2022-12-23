import numpy as np
import torch
from scipy.optimize import linprog

from .creation import gradient_projectors
from .projector import ProjectorInterface
from .utils import get_per_sample_gradients


class Linprog(ProjectorInterface):
    def __init__(self, normalize):
        self.normalize = normalize

    def project(self, model, x, y, loss_fn):
        per_sample_gradients = get_per_sample_gradients(model, x, y, loss_fn)

        coefficients = np.zeros(per_sample_gradients.shape[1])
        constraint_matrix = per_sample_gradients.detach().cpu().numpy()
        constraint_vector = np.zeros(len(constraint_matrix))

        gradient = linprog(coefficients, -constraint_matrix, constraint_vector).x
        gradient = torch.tensor(gradient).float().to(x.device)

        if self.normalize:
            gradient = gradient / torch.linalg.norm(gradient, ord=2)

        return gradient


gradient_projectors.register_builder("linprog", Linprog)
