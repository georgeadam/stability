import torch
from sklearn.ensemble import GradientBoostingRegressor as SklearnGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from .creation import mappers
from .mapper import MapperInterface


class GradientBoostingRegressor(MapperInterface):
    def __init__(self, n_estimators):
        self.model = MultiOutputRegressor(SklearnGradientBoostingRegressor(n_estimators=n_estimators))

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return torch.tensor(self.model.predict(x))

    def __call__(self, x):
        return self.predict(x)


mappers.register_builder("gradient_boosting", GradientBoostingRegressor)
