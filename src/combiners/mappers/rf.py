import torch
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

from .creation import mappers
from .mapper import MapperInterface


class RandomForestRegressor(MapperInterface):
    def __init__(self, n_estimators):
        self.model = SklearnRandomForestRegressor(n_estimators=n_estimators)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return torch.tensor(self.model.predict(x))

    def __call__(self, x):
        return self.predict(x)


mappers.register_builder("rf", RandomForestRegressor)
