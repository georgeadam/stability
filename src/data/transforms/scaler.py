import numpy as np

from .creation import transforms


class Scaler:
    def __init__(self):
        super(Scaler, self).__init__()
        self._mean = None
        self._std = None

    def fit(self, x):
        self._mean = np.mean(x, axis=0)
        self._std = np.std(x, axis=0)

    def __call__(self, x):
        x = (x - self._mean) / self._std

        return x


transforms.register_builder("scaler", Scaler)