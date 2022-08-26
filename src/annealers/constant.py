from .annealer import Annealer
from .creation import annealers


class ConstantAnnealer(Annealer):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, initial, *args):
        return initial


annealers.register_builder("constant", ConstantAnnealer)
