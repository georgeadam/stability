from torch.optim import Adam

from .creation import optimizers

def adam_wrapper(parameters, lr, weight_decay=0.0, **args):
    return Adam(parameters, lr=lr, weight_decay=weight_decay)


optimizers.register_builder("adam", adam_wrapper)