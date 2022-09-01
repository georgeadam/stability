from torch.optim.lr_scheduler import ConstantLR

from .creation import lr_schedulers


def constant_wrapper(optimizer, **kwargs):
    return ConstantLR(optimizer, factor=1.0)


lr_schedulers.register_builder("constant", constant_wrapper)
