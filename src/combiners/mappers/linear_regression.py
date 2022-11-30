import torch
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from .creation import mappers
from .mapper import MapperInterface


class LinearRegression(MapperInterface):
    def __init__(self):
        self.model = SklearnLinearRegression()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return torch.tensor(self.model.predict(x))

    def __call__(self, x):
        return self.predict(x)


mappers.register_builder("linear_regression", LinearRegression)
