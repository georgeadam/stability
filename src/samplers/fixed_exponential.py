import math
import torch
from torch.utils.data import Sampler as PytorchSampler

from .creation import samplers
from .sampler import Sampler


class FixedExponentialSampler(Sampler, PytorchSampler[int]):
    def __init__(self, dataset_size, start_percentage, increase_amount):
        super().__init__(dataset_size)
        self.start_percentage = start_percentage
        self.percentage = start_percentage
        self.increase_amount = increase_amount

    @property
    def indices(self):
        size = int(min(self.percentage, 1) * self.dataset_size)

        return torch.arange(size)

    @property
    def epochs_until_full(self):
        factor = 1 / self.start_percentage

        return math.log(factor) / math.log(self.increase_amount)

    def update(self, *args):
        self.percentage *= self.increase_amount

    def __iter__(self):
        for i in torch.randperm(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


samplers.register_builder("fixed_exponential", FixedExponentialSampler)
