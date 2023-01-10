import numpy as np
import torch
import quadprog

from .creation import gradient_projectors
from .projector import ProjectorInterface
from .utils import get_per_sample_gradients


class Quadprog(ProjectorInterface):
    def __init__(self, normalize, eps, margin):
        self.normalize = normalize
        self.eps = eps
        self.margin = margin

    def project(self, model, x, y, loss_fn):
        per_sample_gradients = get_per_sample_gradients(model, x, y, loss_fn)
        avg_gradient = torch.mean(per_sample_gradients, dim=0)

        memories_np = per_sample_gradients.detach().cpu().double().numpy()
        gradient_np = avg_gradient.detach().cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * self.eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np

        gradient = torch.tensor(x).float()

        if self.normalize:
            gradient = gradient / torch.linalg.norm(gradient, ord=2)

        return gradient


gradient_projectors.register_builder("quadprog", Quadprog)
