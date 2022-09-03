import torch
from torch.utils.data import Sampler as PytorchSampler

from .creation import samplers
from .sampler import Sampler


class StandardSampler(Sampler, PytorchSampler[int]):
    def __init__(self, dataset_size, *args, **kwargs):
        super().__init__(dataset_size)

    @property
    def indices(self):
        return torch.arange(self.dataset_size)

    @property
    def epochs_until_full(self):
        return 0

    def update(self, *args):
        pass

    def __iter__(self):
        for i in torch.randperm(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


samplers.register_builder("standard", StandardSampler)
