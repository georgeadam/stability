import torch
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR as SklearnSVR

from .creation import mappers
from .mapper import MapperInterface


class SVR(MapperInterface):
    def __init__(self, kernel):
        self.model = MultiOutputRegressor(SklearnSVR(kernel=kernel))

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return torch.tensor(self.model.predict(x)).float()

    def __call__(self, x):
        return self.predict(x)


mappers.register_builder("svr", SVR)
