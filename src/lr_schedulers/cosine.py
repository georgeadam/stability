from torch.optim.lr_scheduler import CosineAnnealingLR

from .creation import lr_schedulers


def cosine_annealing_wrapper(optimizer, T_max, **args):
    return CosineAnnealingLR(optimizer, T_max)


lr_schedulers.register_builder("cosine", cosine_annealing_wrapper)