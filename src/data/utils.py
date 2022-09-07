import copy

import numpy as np


def add_label_noise(labels, noise):
    if noise == 0:
        return labels

    labels = copy.deepcopy(labels)

    num_classes = max(labels) + 1
    num_noisy_samples = int(len(labels) * noise)
    noisy_indices = np.random.choice(np.arange(len(labels)), num_noisy_samples, replace=False)

    for noisy_index in noisy_indices:
        labels[noisy_index] = np.random.choice(np.setdiff1d(np.arange(num_classes), labels[noisy_index]))

    return labels
