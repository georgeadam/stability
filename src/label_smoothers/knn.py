import torch
from sklearn.neighbors import NearestNeighbors
from torch.nn.functional import one_hot

from .creation import label_smoothers


class KNNLabelSmoother:
    def __init__(self, alpha, beta, num_classes, n_neighbors=10):
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.n_neighbors = n_neighbors

    def __call__(self, y, logits):
        if len(y.shape) == 1:
            expanded_y = one_hot(y, self.num_classes)
        else:
            expanded_y = y

        neighbors_structure = NearestNeighbors(n_neighbors=self.n_neighbors)
        neighbors_structure.fit(logits)

        nearest_neighbors = neighbors_structure.kneighbors(logits)[1]
        nearest_labels = expanded_y[nearest_neighbors]
        average_labels = torch.mean(nearest_labels.float(), dim=1)

        return (1 - self.alpha) * expanded_y + self.alpha * (
                    (1 - self.beta) * average_labels + (self.beta / self.num_classes) * (torch.ones_like(expanded_y)))


label_smoothers.register_builder("knn", KNNLabelSmoother)
