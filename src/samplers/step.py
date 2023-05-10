import torch
from torch.utils.data import Sampler as PytorchSampler

from .creation import samplers
from .sampler import Sampler


class StepSampler(Sampler, PytorchSampler[int]):
    def __init__(self, dataset_size, original_size, update_size, correct_percentage, threshold, *args, **kwargs):
        super().__init__(dataset_size)

        self.original_size = original_size
        self.update_size = update_size
        self.perfect_count = 0
        self.correct_percentage = correct_percentage
        self.threshold = threshold
        self.done = False

    @property
    def indices(self):
        if self.done:
            return torch.arange(self.original_size + self.update_size)
        else:
            return torch.arange(int(self.original_size * self.correct_percentage))

    @property
    def epochs_until_full(self):
        if self.done:
            return 0
        else:
            return float("inf")

    def update(self, acc):
        if not self.done:
            if acc == 1.0:
                self.perfect_count += 1
            else:
                self.perfect_count = 0

            if self.perfect_count == self.threshold:
                self.done = True

    def __iter__(self):
        for i in torch.randperm(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


samplers.register_builder("step", StepSampler)
