import numpy as np
import torch
from pyhessian import hessian

from .creation import gradient_projectors
from .projector import ProjectorInterface
from .utils import parameters_to_grad_vector


class Hessian(ProjectorInterface):
    def __init__(self, normalize, num_eigenvectors):
        self.normalize = normalize
        self.num_eigenvectors = num_eigenvectors

    def project(self, model, x, y, loss_fn):
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        gradient = parameters_to_grad_vector(model.parameters()).cpu()

        hessian_comp = hessian(model, loss_fn, data=(x, y), cuda=True)
        eigenvalues, eigenvectors = hessian_comp.eigenvalues(top_n=self.num_eigenvectors)
        eigenvectors = [flatten_pyhessian_eigenvector(eigenvector) for eigenvector in eigenvectors]
        eigenvectors = torch.stack(eigenvectors)
        eigenvectors = eigenvectors.T
        eigenvectors = eigenvectors.cpu()
        gradient = gradient - project_vec_onto_subspace(gradient, eigenvectors)

        if self.normalize:
            gradient = gradient / torch.linalg.norm(gradient, ord=2)

        return gradient


def project_vec_onto_subspace(vec, proj_basis):
    # this function assumes that proj_basis is orthonormal. not sure if this is the case for the eigenvectors returned
    # by pyhessian. We know they are orthgonal, need to check if orthonormal, if not we have to normalize
    if proj_basis.shape[1] > 0:  # param x basis_size
        dots = np.matmul(vec, proj_basis)  # basis_size
        # out = torch.matmul(proj_basis, dots)
        # TODO : Check !!!!
        out = np.dot(proj_basis, dots)

        return out
    else:
        return torch.zeros_like(vec)


def flatten_pyhessian_eigenvector(vec):
    flat = []
    for i in range(len(vec)):
        flat.append(vec[i].reshape(-1))

    return torch.cat(flat)


gradient_projectors.register_builder("hessian", Hessian)
